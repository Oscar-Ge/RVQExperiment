"""
使用 PyTorch 自动转换评测 pi05-libero 在 LIBERO 上的表现

本脚本自动将 pi05_libero JAX checkpoint 转换为 PyTorch 格式，然后运行 LIBERO 评测。
转换结果会被缓存，后续运行会直接使用已转换的 checkpoint，除非指定 --force_conversion。

前置要求：
    1. 安装依赖：uv sync
    2. 验证 transformers 版本：uv pip show transformers
    3. 应用 transformers 补丁：
       cp -r ./openpi/src/openpi/models_pytorch/transformers_replace/* \\
           .venv/lib/python3.XX/site-packages/transformers/

使用示例：
    # 默认：自动转换并使用 PyTorch 进行评测
    python run_pi05_libero_benchmark_pytorch.py --task_suite libero_spatial --num_episodes 10

    # 使用已转换的 PyTorch checkpoint
    python run_pi05_libero_benchmark_pytorch.py \\
        --pytorch_checkpoint_dir ~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch

    # 强制重新转换
    python run_pi05_libero_benchmark_pytorch.py --force_conversion

    # 使用 JAX 模式（不转换）
    python run_pi05_libero_benchmark_pytorch.py --use-jax

    # 检查环境和依赖
    python run_pi05_libero_benchmark_pytorch.py --check-env
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Add openpi to path if needed
OPENPI_PATH = Path(__file__).parent / "openpi" / "src"
if OPENPI_PATH.exists():
    sys.path.insert(0, str(OPENPI_PATH))

# Try to add LIBERO to path if it exists locally
LIBERO_LOCAL_PATH = Path(__file__).parent / "LIBERO"
if LIBERO_LOCAL_PATH.exists():
    sys.path.insert(0, str(LIBERO_LOCAL_PATH))

# Add examples directory to path for conversion script
EXAMPLES_PATH = Path(__file__).parent / "openpi" / "examples"
if EXAMPLES_PATH.exists():
    sys.path.insert(0, str(EXAMPLES_PATH))

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

# Conversion script import
CONVERSION_AVAILABLE = False
CONVERSION_IMPORT_ERROR = None
try:
    from convert_jax_model_to_pytorch import convert_pi0_checkpoint
    CONVERSION_AVAILABLE = True
except ImportError as e:
    CONVERSION_IMPORT_ERROR = str(e)

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
# LIBERO uses torch.load to load numpy objects, so we need to add them to safe globals
try:
    import torch
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


def check_transformers_patches():
    """
    Check if transformers patches have been applied.

    Returns:
        tuple: (patches_applied: bool, message: str)
    """
    try:
        import transformers

        # Find transformers installation path
        transformers_path = Path(transformers.__file__).parent

        # Check for key patched files
        patch_indicators = [
            transformers_path / "models" / "gemma" / "modeling_gemma.py",
            transformers_path / "models" / "paligemma" / "modeling_paligemma.py",
        ]

        # Simple heuristic: check if files exist and contain "AdaRMS" keyword
        for patch_file in patch_indicators:
            if not patch_file.exists():
                return False, f"Patch file not found: {patch_file}"

            # Check for AdaRMS as indicator of patches
            # try:
            #     content = patch_file.read_text()
            #     if "AdaRMS" not in content and "ada" not in content.lower():
            #         return False, f"Patch may not be applied to: {patch_file}"
            # except Exception as e:
            #     return False, f"Could not read patch file: {patch_file}, error: {e}"

        return True, "Transformers patches appear to be applied"

    except Exception as e:
        return False, f"Could not verify patches: {e}"


def check_disk_space(output_path: Path, required_gb: float = 5.0):
    """
    Check if sufficient disk space is available.

    Args:
        output_path: Path where checkpoint will be saved
        required_gb: Required space in GB

    Returns:
        tuple: (sufficient_space: bool, message: str)
    """
    try:
        stat = shutil.disk_usage(output_path.parent if output_path.exists() else output_path.parent.parent)
        available_gb = stat.free / (1024 ** 3)

        if available_gb < required_gb:
            return False, f"Insufficient disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
        return True, f"Sufficient disk space: {available_gb:.1f}GB available"

    except Exception as e:
        # Non-fatal: return warning
        return True, f"Could not check disk space: {e}"


def get_converted_checkpoint_path(
    checkpoint_dir: str,
    converted_output_dir: str | None = None
):
    """
    Determine the path for the converted PyTorch checkpoint.

    Args:
        checkpoint_dir: Original JAX checkpoint directory (can be GCS URL)
        converted_output_dir: Optional custom output directory

    Returns:
        Path to converted checkpoint directory
    """
    if converted_output_dir:
        return Path(converted_output_dir).resolve()

    # Use cache directory with predictable naming
    cache_dir = download.get_cache_dir()
    converted_base = cache_dir / "converted_checkpoints"

    # Extract checkpoint name from path/URL
    if checkpoint_dir.startswith("gs://"):
        # gs://openpi-assets/checkpoints/pi05_libero -> pi05_libero
        checkpoint_name = checkpoint_dir.rstrip("/").split("/")[-1]
    else:
        checkpoint_name = Path(checkpoint_dir).name

    return converted_base / f"{checkpoint_name}_pytorch"


def convert_jax_to_pytorch(
    jax_checkpoint_dir: str,
    pytorch_output_dir: Path,
    config_name: str = "pi05_libero",
    precision: str = "bfloat16",
    force: bool = False,
    converted_output_dir: str | None = None,
):
    """
    Convert JAX checkpoint to PyTorch format.

    Args:
        jax_checkpoint_dir: Path to JAX checkpoint (can be GCS URL)
        pytorch_output_dir: Where to save converted checkpoint
        config_name: Model config name
        precision: Model precision
        force: Force conversion even if output exists
        converted_output_dir: Custom output directory passed from CLI

    Returns:
        Path to converted checkpoint

    Raises:
        ImportError: If conversion module not available
        RuntimeError: If conversion fails
    """
    if not CONVERSION_AVAILABLE:
        raise ImportError(
            f"Conversion module not available: {CONVERSION_IMPORT_ERROR}\n"
            f"Make sure you're running from the openpi repository root."
        )

    # Determine final output path
    if converted_output_dir:
        pytorch_output_dir = Path(converted_output_dir)

    # Check if conversion already exists
    output_marker = pytorch_output_dir / "model.safetensors"
    if output_marker.exists() and not force:
        print(f"✓ PyTorch checkpoint already exists: {pytorch_output_dir}")
        print("  Use --force_conversion to re-convert")
        return pytorch_output_dir

    print("\n" + "=" * 80)
    print("CONVERTING JAX CHECKPOINT TO PYTORCH")
    print("=" * 80)
    print(f"Source (JAX): {jax_checkpoint_dir}")
    print(f"Target (PyTorch): {pytorch_output_dir}")
    print(f"Config: {config_name}")
    print(f"Precision: {precision}")

    # Pre-flight checks
    print("\nRunning pre-flight checks...")

    # 1. Check transformers patches
    patches_ok, patch_msg = check_transformers_patches()
    if not patches_ok:
        raise RuntimeError(
            f"Transformers patches not applied: {patch_msg}\n\n"
            f"Please apply patches with:\n"
            f"  cp -r ./openpi/src/openpi/models_pytorch/transformers_replace/* "
            f".venv/lib/python3.XX/site-packages/transformers/\n\n"
            f"See openpi/README.md for details."
        )
    print(f"✓ {patch_msg}")

    # 2. Check disk space
    space_ok, space_msg = check_disk_space(pytorch_output_dir)
    if not space_ok:
        raise RuntimeError(space_msg)
    print(f"✓ {space_msg}")

    # 3. Download JAX checkpoint if needed
    print(f"\nResolving JAX checkpoint...")
    jax_checkpoint_path = download.maybe_download(jax_checkpoint_dir)
    print(f"✓ JAX checkpoint ready: {jax_checkpoint_path}")

    # 4. Get model config
    print(f"\nLoading model config...")
    model_config = train_config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    print(f"✓ Model config loaded: {config_name}")

    # 5. Run conversion
    print(f"\nStarting conversion (this may take several minutes)...")
    try:
        pytorch_output_dir.mkdir(parents=True, exist_ok=True)
        convert_pi0_checkpoint(
            checkpoint_dir=str(jax_checkpoint_path),
            precision=precision,
            output_path=str(pytorch_output_dir),
            model_config=model_config
        )
        src_assets = Path(jax_checkpoint_path) / "assets"
        dst_assets = pytorch_output_dir / "assets"
        
        if src_assets.exists():
            print(f"Copying assets from {src_assets} to {dst_assets}...")
            if dst_assets.exists():
                shutil.rmtree(dst_assets)
            shutil.copytree(src_assets, dst_assets)
            print(f"✓ Assets copied successfully.")
        else:
            print(f"⚠️ Warning: No assets folder found at {src_assets}")
    except Exception as e:
        # Cleanup partial conversion on failure
        if pytorch_output_dir.exists():
            print(f"\nCleaning up partial conversion...")
            shutil.rmtree(pytorch_output_dir, ignore_errors=True)
        raise RuntimeError(f"Conversion failed: {e}") from e

    # 6. Verify conversion
    if not output_marker.exists():
        raise RuntimeError(
            f"Conversion completed but output file not found: {output_marker}"
        )

    print("\n" + "=" * 80)
    print("✓ CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"PyTorch checkpoint saved to: {pytorch_output_dir}")
    try:
        total_size = sum(f.stat().st_size for f in pytorch_output_dir.rglob('*') if f.is_file()) / (1024**3)
        print(f"Size: {total_size:.2f} GB")
    except:
        pass

    return pytorch_output_dir


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
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero",
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    use_pytorch: bool = True,
    pytorch_checkpoint_dir: str | None = None,
    force_conversion: bool = False,
    converted_output_dir: str | None = None,
):
    """
    Load pi05-libero policy from checkpoint.

    Args:
        checkpoint_dir: Path to JAX checkpoint directory (can be GCS URL or local path)
        default_prompt: Default prompt to use if not provided in observations
        pytorch_device: Device for PyTorch models ("cuda", "cpu", etc.)
        use_pytorch: If True, convert to PyTorch and use PyTorch model
        pytorch_checkpoint_dir: If provided, use this PyTorch checkpoint instead of converting
        force_conversion: Force re-conversion even if PyTorch checkpoint exists
        converted_output_dir: Custom output directory for converted checkpoint

    Returns:
        Loaded Policy object ready for inference
    """
    # Determine which checkpoint to load
    if pytorch_checkpoint_dir:
        # User provided pre-converted PyTorch checkpoint
        print(f"Using pre-converted PyTorch checkpoint: {pytorch_checkpoint_dir}")
        checkpoint_to_load = pytorch_checkpoint_dir
    elif use_pytorch:
        # Convert JAX to PyTorch
        pytorch_output_dir = get_converted_checkpoint_path(checkpoint_dir, converted_output_dir)
        checkpoint_to_load = convert_jax_to_pytorch(
            jax_checkpoint_dir=checkpoint_dir,
            pytorch_output_dir=pytorch_output_dir,
            force=force_conversion,
            converted_output_dir=converted_output_dir,
        )
    else:
        # Use JAX checkpoint directly
        print(f"Using JAX checkpoint: {checkpoint_dir}")
        checkpoint_to_load = checkpoint_dir

    print(f"\nLoading pi05-libero policy from: {checkpoint_to_load}")

    # Get pi05_libero config
    config = train_config.get_config("pi05_libero")

    print("Creating policy with pi05_libero configuration...")

    # Create policy using policy_config.create_trained_policy
    # This handles:
    # - Downloading from GCS (if needed)
    # - Loading model weights (auto-detects JAX vs PyTorch)
    # - Setting up transforms
    # - Loading normalization stats
    policy = _policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_to_load,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )

    print("✓ Policy loaded successfully!")
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    return policy


def evaluate_task(
    policy,  # Policy object from openpi.policies.policy
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
    """
    Evaluate policy on a single LIBERO task.

    Args:
        policy: Loaded OpenPI policy
        env: LIBERO environment
        task_description: Natural language task description
        initial_states: List of initial states for the task
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        num_steps_wait: Number of steps to wait for objects to stabilize
        replan_steps: Number of steps before replanning
        resize_size: Size to resize images to
        verbose: Whether to print episode-level results

    Returns:
        Dictionary with evaluation results
    """
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
                # IMPORTANT: Do nothing for the first few timesteps because
                # the simulator drops objects and we need to wait for them to fall
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Preprocess images
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Resize and convert to uint8
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, resize_size, resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
                )

                # Prepare state observation (eef pos + axis angle + gripper)
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }

                    # Query model to get action
                    policy_output = policy.infer(element)
                    action_chunk = policy_output["actions"]

                    assert (
                        len(action_chunk) >= replan_steps
                    ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."

                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()

                # Execute action in environment
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
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero",
    verbose: bool = True,
    seed: int = 42,
    use_pytorch: bool = True,
    pytorch_checkpoint_dir: str | None = None,
    force_conversion: bool = False,
    converted_output_dir: str | None = None,
):
    """
    Run full benchmark on a LIBERO task suite.

    Args:
        task_suite_name: Name of LIBERO task suite to evaluate
        num_episodes: Number of episodes to run per task
        checkpoint_dir: Path to JAX checkpoint directory
        verbose: Whether to print detailed output
        seed: Random seed
        use_pytorch: If True, convert to PyTorch and use PyTorch model
        pytorch_checkpoint_dir: If provided, use this PyTorch checkpoint instead of converting
        force_conversion: Force re-conversion even if PyTorch checkpoint exists
        converted_output_dir: Custom output directory for converted checkpoint

    Returns:
        List of per-task results
    """
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
    print("LOADING PI0.5-LIBERO POLICY")
    print("=" * 80)
    policy = load_pi05_libero_policy(
        checkpoint_dir=checkpoint_dir,
        use_pytorch=use_pytorch,
        pytorch_checkpoint_dir=pytorch_checkpoint_dir,
        force_conversion=force_conversion,
        converted_output_dir=converted_output_dir,
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
        env.seed(seed)  # Seed affects object positions even with fixed initial state

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
        "libero_available": LIBERO_AVAILABLE,
        "libero_error": LIBERO_IMPORT_ERROR,
        "robosuite_available": ROBOSUITE_AVAILABLE,
        "robosuite_error": ROBOSUITE_IMPORT_ERROR,
        "robosuite_version": ROBOSUITE_VERSION,
        "python_version": sys.version,
        "sys_path": sys.path,
    }

    if verbose:
        print("=" * 80)
        print("ENVIRONMENT CHECK")
        print("=" * 80)
        print(f"Python: {sys.version.split()[0]}")
        print(f"openpi available: {'✓ YES' if OPENPI_AVAILABLE else '✗ NO'}")
        if not OPENPI_AVAILABLE:
            print(f"  Error: {OPENPI_IMPORT_ERROR}")
        else:
            try:
                cfg = train_config.get_config("pi05_libero")
                print(f"  pi05_libero config found: ✓")
            except Exception as e:
                print(f"  pi05_libero config error: {e}")

        # Check conversion availability
        print(f"Conversion available: {'✓ YES' if CONVERSION_AVAILABLE else '✗ NO'}")
        if not CONVERSION_AVAILABLE:
            print(f"  Error: {CONVERSION_IMPORT_ERROR}")
            print(f"  Note: Conversion requires running from openpi repository root")

        # Check transformers patches
        if CONVERSION_AVAILABLE:
            patches_ok, patch_msg = check_transformers_patches()
            print(f"Transformers patches: {'✓ YES' if patches_ok else '✗ NO'}")
            if not patches_ok:
                print(f"  {patch_msg}")
                print(f"  Apply with:")
                print(f"    cp -r ./openpi/src/openpi/models_pytorch/transformers_replace/* \\")
                print(f"        .venv/lib/python3.XX/site-packages/transformers/")

        print(f"LIBERO available: {'✓ YES' if LIBERO_AVAILABLE else '✗ NO'}")
        if not LIBERO_AVAILABLE:
            print(f"  Error: {LIBERO_IMPORT_ERROR}")
        print(f"robosuite available: {'✓ YES' if ROBOSUITE_AVAILABLE else '✗ NO'}")
        if ROBOSUITE_AVAILABLE:
            print(f"  robosuite version: {ROBOSUITE_VERSION}")
        else:
            print(f"  Error: {ROBOSUITE_IMPORT_ERROR}")
            if ROBOSUITE_IMPORT_ERROR and "single_arm_env" in ROBOSUITE_IMPORT_ERROR:
                print("  Hint: robosuite 版本不兼容，试试: pip install 'robosuite==1.3.0'")
            else:
                print("  Try: pip install 'robosuite==1.4.0'")
        print(f"\nPython path (first 3):")
        for i, p in enumerate(sys.path[:3]):
            print(f"  {i+1}. {p}")
        if LIBERO_LOCAL_PATH.exists():
            print(f"\nLIBERO local found at: {LIBERO_LOCAL_PATH}")
        print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert pi05-libero to PyTorch and benchmark on LIBERO tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Convert JAX to PyTorch and benchmark on libero_spatial
  python run_pi05_libero_benchmark_pytorch.py --task_suite libero_spatial --num_episodes 10

  # Use pre-converted PyTorch checkpoint
  python run_pi05_libero_benchmark_pytorch.py \\
      --pytorch_checkpoint_dir ~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch

  # Force re-conversion of checkpoint
  python run_pi05_libero_benchmark_pytorch.py --force_conversion

  # Run with JAX model (no conversion)
  python run_pi05_libero_benchmark_pytorch.py --use-jax

  # Custom conversion output location
  python run_pi05_libero_benchmark_pytorch.py \\
      --converted_output_dir ./my_converted_checkpoints

  # Check environment and dependencies
  python run_pi05_libero_benchmark_pytorch.py --check-env
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
        "--checkpoint_dir",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_libero",
        help="Checkpoint directory (GCS URL or local path)",
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

    # PyTorch conversion arguments
    parser.add_argument(
        "--use-jax",
        action="store_true",
        help="Use JAX model instead of converting to PyTorch (default: use PyTorch)",
    )

    parser.add_argument(
        "--pytorch_checkpoint_dir",
        type=str,
        default=None,
        help="Path to pre-converted PyTorch checkpoint (skips conversion)",
    )

    parser.add_argument(
        "--force_conversion",
        action="store_true",
        help="Force re-conversion even if PyTorch checkpoint exists",
    )

    parser.add_argument(
        "--converted_output_dir",
        type=str,
        default=None,
        help="Custom output directory for converted checkpoint (default: ~/.cache/openpi/converted_checkpoints/)",
    )

    args = parser.parse_args()

    # Check environment if requested
    if args.check_env:
        check_environment(verbose=True)
        return

    results = run_benchmark(
        task_suite_name=args.task_suite,
        num_episodes=args.num_episodes,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        verbose=not args.quiet,
        use_pytorch=not args.use_jax,
        pytorch_checkpoint_dir=args.pytorch_checkpoint_dir,
        force_conversion=args.force_conversion,
        converted_output_dir=args.converted_output_dir,
    )


    # Print final message
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)
    print(f"Task Suite: {args.task_suite}")
    print(f"Episodes per Task: {args.num_episodes}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Seed: {args.seed}")
    print(f"Verbose: {not args.quiet}")
    print("Results:")
    for r in results:
        print(f"  Task {r['task_id']}: {r['success_rate']:.1f}% success "
              f"({r['num_successes']}/{r['num_episodes']}), "
              f"avg. length: {r['avg_episode_length']:.1f} steps")
    print("=" * 80)


if __name__ == "__main__":
    main()
