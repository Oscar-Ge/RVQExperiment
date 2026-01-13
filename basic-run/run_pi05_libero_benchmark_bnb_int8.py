"""
使用 BitandBytes INT8 量化 PaliGemma，评测 pi05-libero 在 LIBERO 上的表现

本脚本在 PyTorch 转换的基础上，使用 bitsandbytes 对 PaliGemma (2B VLM) 进行 INT8 量化，
同时保持 Action Expert (300M) 为 bfloat16 精度，以快速验证量化的可行性。

前置要求：
    1. 安装依赖：uv sync
    2. 安装 bitsandbytes：pip install bitsandbytes
    3. 已有转换好的 PyTorch checkpoint（使用 run_pi05_libero_benchmark_pytorch.py）

使用示例：
    # 默认：INT8 量化 PaliGemma 并评测
    python run_pi05_libero_benchmark_bnb_int8.py --task_suite libero_spatial --num_episodes 10

    # 使用自定义 checkpoint
    python run_pi05_libero_benchmark_bnb_int8.py \\
        --pytorch_checkpoint_dir ~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch

    # 不量化（baseline 对比）
    python run_pi05_libero_benchmark_bnb_int8.py --no-quantize

    # 检查环境
    python run_pi05_libero_benchmark_bnb_int8.py --check-env
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add openpi to path if needed
OPENPI_PATH = Path(__file__).parent / "openpi" / "src"
if OPENPI_PATH.exists():
    sys.path.insert(0, str(OPENPI_PATH))

# Try to add LIBERO to path if it exists locally
LIBERO_LOCAL_PATH = Path(__file__).parent / "LIBERO"
if LIBERO_LOCAL_PATH.exists():
    sys.path.insert(0, str(LIBERO_LOCAL_PATH))

# openpi dependency check
OPENPI_AVAILABLE = False
OPENPI_IMPORT_ERROR = None
try:
    from openpi.shared import download
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as train_config
    from openpi_client import image_tools
    import openpi.models.pi0_config
    OPENPI_AVAILABLE = True
except ImportError as e:
    OPENPI_IMPORT_ERROR = str(e)

# bitsandbytes availability check
BNB_AVAILABLE = False
BNB_IMPORT_ERROR = None
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError as e:
    BNB_IMPORT_ERROR = str(e)

# robosuite dependency check
ROBOSUITE_AVAILABLE = False
ROBOSUITE_IMPORT_ERROR = None
ROBOSUITE_VERSION = None
try:
    import robosuite  # type: ignore
    ROBOSUITE_AVAILABLE = True
    ROBOSUITE_VERSION = getattr(robosuite, "__version__", "unknown")
except ImportError as e:
    ROBOSUITE_IMPORT_ERROR = str(e)

# Fix for PyTorch 2.6+ torch.load security change
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
except Exception as e:
    print(f"Warning: Could not configure torch safe globals: {e}")

# LIBERO imports
LIBERO_AVAILABLE = False
LIBERO_IMPORT_ERROR = None

try:
    import libero.libero.benchmark as benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError as e:
    LIBERO_IMPORT_ERROR = str(e)
    print("⚠️  Warning: LIBERO import failed!")
    print(f"  Error: {e}")
    print(f"  LIBERO local path: {LIBERO_LOCAL_PATH} (exists: {LIBERO_LOCAL_PATH.exists()})")
    print(f"  Python path: {sys.path[:3]}")
    print("  Install with: pip install libero")
    print("  If robosuite missing, install: pip install 'robosuite==1.4.0'")
    if "single_arm_env" in LIBERO_IMPORT_ERROR:
        print("  Hint: robosuite 版本不兼容，请安装: pip install 'robosuite==1.3.0'")
    print()


# Task suite configurations
TASK_SUITES = {
    "libero_spatial": {"num_tasks": 10, "max_steps": 220},
    "libero_object": {"num_tasks": 10, "max_steps": 280},
    "libero_goal": {"num_tasks": 10, "max_steps": 300},
    "libero_10": {"num_tasks": 10, "max_steps": 520},
    "libero_90": {"num_tasks": 90, "max_steps": 400},
}

LIBERO_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0])
LIBERO_ENV_RESOLUTION = 256  # Resolution used in training
LIBERO_RESIZE_SIZE = 224  # Resize for model input


def quantize_paligemma_int8(model):
    """
    Apply INT8 quantization to PaliGemma using bitsandbytes.
    Strategy based on AutoQVLA paper & technical stability:
      1. Quantize Language Model (Safe, saves most memory)
      2. Skip Vision Encoder (Fixes torch.compile crash, preserves visual features)
      3. Skip Projector (Paper explicitly warns this is highly sensitive)
    """
    if not BNB_AVAILABLE:
        raise ImportError(f"bitsandbytes not available: {BNB_IMPORT_ERROR}")

    print("\n" + "=" * 80)
    print("QUANTIZING PALIGEMMA (SMART MIXED-PRECISION)")
    print("=" * 80)

    # Get PaliGemma component
    paligemma = model.paligemma_with_expert.paligemma

    # Quantize linear layers in Language Model ONLY
    quantized_layers = 0
    original_params = 0
    quantized_params = 0
    skipped_params = 0

    for name, module in paligemma.named_modules():
        is_vision = "vision_tower" in name or "siglip" in name
        is_projector = "multi_modal_projector" in name
        
        # 如果是 Vision Encoder 或 Projector，跳过量化
        if is_vision or is_projector:
            if isinstance(module, torch.nn.Linear):
                skipped_params += module.weight.numel()
                if module.bias is not None:
                    skipped_params += module.bias.numel()
            continue

        if isinstance(module, torch.nn.Linear):
            # Calculate parameter count
            param_count = module.weight.numel()
            if module.bias is not None:
                param_count += module.bias.numel()
            original_params += param_count

            # Replace with INT8 quantized linear
            quantized_linear = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,  # Use INT8 weights
                threshold=6.0,  # Outlier threshold
            )

            # Copy weights and bias
            quantized_linear.weight = bnb.nn.Int8Params(
                module.weight.data,
                requires_grad=False,
                has_fp16_weights=False
            )
            if module.bias is not None:
                quantized_linear.bias = module.bias

            # Replace the module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(paligemma.named_modules())[parent_name]
            else:
                parent = paligemma
            setattr(parent, child_name, quantized_linear)

            quantized_layers += 1
            # INT8 weights + FP32 bias if exists
            quantized_params += param_count // 4 + (module.bias.numel() if module.bias is not None else 0)

    compression_ratio = original_params / quantized_params if quantized_params > 0 else 1.0

    print(f"✓ Quantized {quantized_layers} linear layers in Language Module")
    print(f"  Skipped Vision/Projector params: {skipped_params:,} (Kept as BFloat16)")
    print(f"  Quantized LLM parameters: {quantized_params:,}")
    print(f"  Compression ratio (on quantized parts): {compression_ratio:.2f}x")
    print("=" * 80)

    return model


def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite and OpenPI LIBERO example.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def load_pi05_libero_policy(
    checkpoint_dir: str = "~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch",
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    quantize_paligemma: bool = True,
):
    """
    Load pi05-libero policy from PyTorch checkpoint with optional quantization.

    Args:
        checkpoint_dir: Path to PyTorch checkpoint directory
        default_prompt: Default prompt to use if not provided in observations
        pytorch_device: Device for PyTorch models ("cuda", "cpu", etc.)
        quantize_paligemma: If True, apply INT8 quantization to PaliGemma

    Returns:
        Loaded Policy object ready for inference
    """
    if not OPENPI_AVAILABLE:
        raise ImportError(f"openpi not available: {OPENPI_IMPORT_ERROR}")

    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    print(f"\nLoading pi05-libero policy from: {checkpoint_dir}")

    # Get pi05_libero config
    config = train_config.get_config("pi05_libero")

    print("Creating policy with pi05_libero configuration...")

    # Create policy
    policy = _policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )

    # Apply quantization if requested
    if quantize_paligemma:
        policy._model = quantize_paligemma_int8(policy._model)

    print("✓ Policy loaded successfully!")

    # Print memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    return policy


def evaluate_task(
    policy,
    env: "OffScreenRenderEnv",
    task_description: str,
    initial_states: list,
    num_episodes: int,
    max_steps: int,
    num_steps_wait: int = 10,
    replan_steps: int = 5,
    resize_size: int = LIBERO_RESIZE_SIZE,
    verbose: bool = True,
):
    """Evaluate policy on a single LIBERO task."""
    import collections

    successes = []
    episode_lengths = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for episode_idx in range(num_episodes):
        # Reset with cycled initial state
        init_state_idx = episode_idx % len(initial_states)
        env.reset()
        obs = env.set_init_state(initial_states[init_state_idx])

        success = False
        action_plan = collections.deque()
        t = 0

        while t < max_steps + num_steps_wait:
            try:
                # Do nothing for the first few timesteps
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Preprocess images (rotate 180 degrees)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Resize
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, resize_size, resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
                )

                # Prepare state
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                if not action_plan:
                    # Query model
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }

                    policy_output = policy.infer(element)
                    action_chunk = policy_output["actions"]
                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()

                # Execute action
                obs, reward, done, info = env.step(action.tolist())

                if done:
                    success = True
                    episode_lengths.append(t - num_steps_wait + 1)
                    break

                t += 1

            except Exception as e:
                if verbose:
                    print(f"    Error during episode: {e}")
                    import traceback
                    traceback.print_exc()
                break

        if not success:
            episode_lengths.append(max_steps)

        successes.append(success)

        if verbose:
            status = "✓ SUCCESS" if success else "✗ FAILURE"
            print(f"    Episode {episode_idx + 1}/{num_episodes}: {status} "
                  f"(steps: {episode_lengths[-1]})")

    success_rate = 100.0 * sum(successes) / len(successes)
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    [Peak VRAM during inference]: {peak_mem:.2f} GB")
    return {
        "success_rate": success_rate,
        "num_successes": sum(successes),
        "num_episodes": num_episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "episode_lengths": episode_lengths,
    }


def run_benchmark(
    task_suite_name: str,
    num_episodes: int = 10,
    checkpoint_dir: str = "~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch",
    verbose: bool = True,
    seed: int = 42,
    quantize_paligemma: bool = True,
):
    """Run full benchmark on a LIBERO task suite."""
    if not OPENPI_AVAILABLE:
        raise ImportError(f"openpi is not available. Error: {OPENPI_IMPORT_ERROR}")
    if not LIBERO_AVAILABLE:
        error_msg = "LIBERO is not available."
        if LIBERO_IMPORT_ERROR:
            error_msg += f"\nImport error: {LIBERO_IMPORT_ERROR}"
        error_msg += "\nPlease install it with: pip install libero"
        raise ImportError(error_msg)

    # Set random seed
    np.random.seed(seed)

    # Load policy
    print("\n" + "=" * 80)
    print("LOADING PI0.5-LIBERO POLICY" + (" (INT8 QUANTIZED)" if quantize_paligemma else ""))
    print("=" * 80)
    policy = load_pi05_libero_policy(
        checkpoint_dir=checkpoint_dir,
        quantize_paligemma=quantize_paligemma,
    )

    # Load benchmark
    print("\n" + "=" * 80)
    print(f"LOADING LIBERO BENCHMARK: {task_suite_name.upper()}")
    print("=" * 80)

    if task_suite_name not in TASK_SUITES:
        raise ValueError(f"Unknown task suite: {task_suite_name}. "
                        f"Available: {list(TASK_SUITES.keys())}")

    suite_config = TASK_SUITES[task_suite_name]
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    print(f"Task Suite: {task_suite_name}")
    print(f"Number of Tasks: {suite_config['num_tasks']}")
    print(f"Max Steps: {suite_config['max_steps']}")
    print(f"Episodes per Task: {num_episodes}")

    # Evaluate all tasks
    all_results = []

    for task_id in range(suite_config["num_tasks"]):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        print(f"\n{'=' * 80}")
        print(f"TASK {task_id + 1}/{suite_config['num_tasks']}: {task_name}")
        print(f"{'=' * 80}")
        print(f"Description: {task_description}")
        print(f"Initial States: {len(initial_states)}")

        # Create environment
        from libero.libero import get_libero_path
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        # Evaluate
        results = evaluate_task(
            policy=policy,
            env=env,
            task_description=task_description,
            initial_states=initial_states,
            num_episodes=num_episodes,
            max_steps=suite_config["max_steps"],
            verbose=verbose,
        )

        results["task_id"] = task_id
        results["task_name"] = task_name
        results["task_description"] = task_description

        all_results.append(results)

        print(f"\n  Results:")
        print(f"    Success Rate: {results['success_rate']:.1f}%")
        print(f"    Successes: {results['num_successes']}/{results['num_episodes']}")
        print(f"    Avg Episode Length: {results['avg_episode_length']:.1f}")

        env.close()

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    overall_success_rate = np.mean([r["success_rate"] for r in all_results])
    total_successes = sum([r["num_successes"] for r in all_results])
    total_episodes = sum([r["num_episodes"] for r in all_results])

    print(f"\nTask Suite: {task_suite_name}")
    print(f"Quantization: {'PaliGemma INT8, Action Expert bfloat16' if quantize_paligemma else 'No quantization (baseline)'}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"Total Successes: {total_successes}/{total_episodes}")

    print(f"\nPer-Task Results:")
    print(f"{'ID':<4} {'Task Name':<45} {'Success Rate':<15} {'Successes'}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['task_id']:<4} {r['task_name']:<45} "
              f"{r['success_rate']:>6.1f}%        {r['num_successes']:>2}/{r['num_episodes']}")

    print("=" * 80)

    return all_results


def check_environment(verbose: bool = True):
    """Check if all dependencies are properly installed."""
    results = {
        "openpi_available": OPENPI_AVAILABLE,
        "openpi_error": OPENPI_IMPORT_ERROR,
        "bnb_available": BNB_AVAILABLE,
        "bnb_error": BNB_IMPORT_ERROR,
        "libero_available": LIBERO_AVAILABLE,
        "libero_error": LIBERO_IMPORT_ERROR,
        "robosuite_available": ROBOSUITE_AVAILABLE,
        "robosuite_error": ROBOSUITE_IMPORT_ERROR,
        "robosuite_version": ROBOSUITE_VERSION,
        "python_version": sys.version,
    }

    if verbose:
        print("=" * 80)
        print("ENVIRONMENT CHECK")
        print("=" * 80)
        print(f"Python: {sys.version.split()[0]}")
        print(f"openpi available: {'✓ YES' if OPENPI_AVAILABLE else '✗ NO'}")
        if not OPENPI_AVAILABLE:
            print(f"  Error: {OPENPI_IMPORT_ERROR}")

        print(f"bitsandbytes available: {'✓ YES' if BNB_AVAILABLE else '✗ NO'}")
        if not BNB_AVAILABLE:
            print(f"  Error: {BNB_IMPORT_ERROR}")
            print(f"  Install with: pip install bitsandbytes")
        else:
            import bitsandbytes
            print(f"  Version: {bitsandbytes.__version__}")

        print(f"LIBERO available: {'✓ YES' if LIBERO_AVAILABLE else '✗ NO'}")
        if not LIBERO_AVAILABLE:
            print(f"  Error: {LIBERO_IMPORT_ERROR}")
        print(f"robosuite available: {'✓ YES' if ROBOSUITE_AVAILABLE else '✗ NO'}")
        if ROBOSUITE_AVAILABLE:
            print(f"  robosuite version: {ROBOSUITE_VERSION}")
        else:
            print(f"  Error: {ROBOSUITE_IMPORT_ERROR}")
        print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pi05-libero with bitsandbytes INT8 quantization on LIBERO tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: INT8 quantize PaliGemma and benchmark
  python run_pi05_libero_benchmark_bnb_int8.py --task_suite libero_spatial --num_episodes 10

  # Run without quantization (baseline)
  python run_pi05_libero_benchmark_bnb_int8.py --no-quantize

  # Use custom PyTorch checkpoint
  python run_pi05_libero_benchmark_bnb_int8.py --pytorch_checkpoint_dir /path/to/checkpoint

  # Check environment
  python run_pi05_libero_benchmark_bnb_int8.py --check-env
        """
    )

    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        choices=list(TASK_SUITES.keys()),
        help="LIBERO task suite to evaluate (default: libero_spatial)",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes per task (default: 10)",
    )

    parser.add_argument(
        "--pytorch_checkpoint_dir",
        type=str,
        default="~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch",
        help="PyTorch checkpoint directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check if all dependencies are installed and exit",
    )

    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization (run baseline for comparison)",
    )

    args = parser.parse_args()

    # Check environment if requested
    if args.check_env:
        check_environment(verbose=True)
        return

    results = run_benchmark(
        task_suite_name=args.task_suite,
        num_episodes=args.num_episodes,
        checkpoint_dir=args.pytorch_checkpoint_dir,
        seed=args.seed,
        verbose=not args.quiet,
        quantize_paligemma=not args.no_quantize,
    )

    # Print final message
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)
    print(f"Task Suite: {args.task_suite}")
    print(f"Episodes per Task: {args.num_episodes}")
    print(f"Quantization: {'PaliGemma INT8' if not args.no_quantize else 'Disabled (baseline)'}")
    print(f"Checkpoint Directory: {args.pytorch_checkpoint_dir}")
    print(f"Seed: {args.seed}")
    print("Results:")
    for r in results:
        print(f"  Task {r['task_id']}: {r['success_rate']:.1f}% success "
              f"({r['num_successes']}/{r['num_episodes']}), "
              f"avg. length: {r['avg_episode_length']:.1f} steps")
    print("=" * 80)


if __name__ == "__main__":
    main()
