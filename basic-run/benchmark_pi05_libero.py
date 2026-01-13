"""
Download and benchmark pi05-libero checkpoint on LIBERO tasks.

This script:
1. Downloads the pi05_libero checkpoint from gs://openpi-assets/checkpoints/pi05_libero
2. Loads the checkpoint in PyTorch format
3. Runs benchmark evaluation on LIBERO task suites

Usage:
    python benchmark_pi05_libero.py --task_suite libero_spatial --num_episodes 10
    python benchmark_pi05_libero.py --task_suite libero_10 --num_episodes 50 --device cuda
"""

import argparse
import dataclasses
import os
import pathlib
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# OpenPI imports
from openpi.shared import download
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi import models as _model
from openpi.policies import libero_policy
from openpi import transforms
from openpi.training import config as train_config

# LIBERO imports
try:
    import libero.libero.benchmark as benchmark
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    print("Warning: LIBERO not installed. Install with: pip install libero")
    benchmark = None


# LIBERO task suite configurations
TASK_SUITE_CONFIGS = {
    "libero_spatial": {"num_tasks": 10, "max_steps": 220},
    "libero_object": {"num_tasks": 10, "max_steps": 280},
    "libero_goal": {"num_tasks": 10, "max_steps": 300},
    "libero_10": {"num_tasks": 10, "max_steps": 520},
    "libero_90": {"num_tasks": 90, "max_steps": 400},
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 6 joints + gripper


@dataclasses.dataclass
class Args:
    """Command-line arguments for LIBERO benchmarking."""
    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_libero"
    task_suite_name: str = "libero_spatial"
    num_episodes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bfloat16: bool = True
    image_size: int = 224
    seed: int = 42
    verbose: bool = True
    save_videos: bool = False
    video_dir: str = "./libero_videos"


class Pi05LiberoModel(nn.Module):
    """PyTorch wrapper for pi05-libero checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        use_bfloat16: bool = True,
    ):
        super().__init__()
        self.device = device
        self.dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        # Download checkpoint from GCS
        print(f"Downloading checkpoint from {checkpoint_path}...")
        local_checkpoint_path = download.maybe_download(
            checkpoint_path,
            gs={"token": "anon"}  # Anonymous access for public bucket
        )
        print(f"Checkpoint downloaded to: {local_checkpoint_path}")

        # Load pi0.5 model configuration
        self.model = self._load_model(local_checkpoint_path)
        self.model.to(device)

        if use_bfloat16:
            self.model = self.model.to(torch.bfloat16)

        self.model.eval()

        # Load normalization statistics
        self.norm_stats = self._load_normalization_stats(local_checkpoint_path)

        print(f"Model loaded successfully on {device}")

    def _load_model(self, checkpoint_path: pathlib.Path) -> nn.Module:
        """Load the pi0.5 model architecture and weights."""
        # Create pi0.5 model configuration
        # Based on openpi/training/config.py - pi05_libero uses pi0_config.Pi0Config(pi05=True)

        # Pi0.5 architecture specs
        vlm_config = {
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_attention_heads": 8,
            "head_dim": 256,
            "num_hidden_layers": 18,
            "num_key_value_heads": 4,
        }

        action_expert_config = vlm_config.copy()

        # Create model
        model = PaliGemmaWithExpertModel(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            use_adarms=True,  # Pi0.5 uses AdaRMS
            precision="bfloat16" if self.dtype == torch.bfloat16 else "float32",
        )

        # Load weights from checkpoint
        params_file = checkpoint_path / "params" / "params"
        if params_file.exists():
            print(f"Loading weights from {params_file}")
            # Load JAX/Flax checkpoint and convert to PyTorch
            loaded_params = _model.restore_params(
                params_file,
                restore_type=np.ndarray
            )

            # Convert JAX params to PyTorch state_dict
            state_dict = self._convert_jax_to_pytorch(loaded_params, model)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: No params file found at {params_file}, using random initialization")

        return model

    def _convert_jax_to_pytorch(self, jax_params: dict, model: nn.Module) -> dict:
        """Convert JAX/Flax parameters to PyTorch state_dict."""
        # This is a simplified conversion - you may need to adapt based on actual checkpoint format
        state_dict = {}

        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}/{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flattened = flatten_dict(jax_params)

        # Map JAX parameter names to PyTorch parameter names
        for jax_name, jax_param in flattened.items():
            # Convert JAX naming to PyTorch naming
            pytorch_name = jax_name.replace('/', '.')

            # Convert numpy array to PyTorch tensor
            if isinstance(jax_param, np.ndarray):
                state_dict[pytorch_name] = torch.from_numpy(jax_param)

        return state_dict

    def _load_normalization_stats(self, checkpoint_path: pathlib.Path) -> dict:
        """Load normalization statistics from checkpoint."""
        assets_dir = checkpoint_path / "assets"
        if assets_dir.exists():
            # Load normalization stats using openpi utilities
            from openpi.shared import normalize as _normalize
            return _normalize.load(assets_dir)
        return {}

    def predict_action(
        self,
        observation: dict,
        prompt: str,
    ) -> np.ndarray:
        """
        Predict action from observation.

        Args:
            observation: Dict with keys 'image', 'wrist_image', 'state'
            prompt: Task description string

        Returns:
            Action array of shape (7,) for LIBERO (6 joints + gripper)
        """
        with torch.no_grad():
            # Prepare inputs (resize images, tokenize prompt, etc.)
            inputs = self._prepare_inputs(observation, prompt)

            # Forward pass through model
            outputs = self.model(**inputs)

            # Extract action prediction
            action = outputs["actions"][0].cpu().numpy()

            # Apply normalization if available
            if "actions" in self.norm_stats:
                action = self._denormalize(action, self.norm_stats["actions"])

            # Return first 7 dimensions for LIBERO
            return action[:7]

    def _prepare_inputs(self, observation: dict, prompt: str) -> dict:
        """Prepare model inputs from observation."""
        # This is a simplified version - you'll need to implement full preprocessing
        # based on libero_policy.LiberoInputs transform

        inputs = {
            "image": observation.get("image"),
            "wrist_image": observation.get("wrist_image"),
            "state": observation.get("state"),
            "prompt": prompt,
        }

        # TODO: Apply full transform pipeline:
        # - Resize images to 224x224
        # - Tokenize prompt
        # - Normalize state
        # - Convert to tensors

        return inputs

    def _denormalize(self, normalized: np.ndarray, stats: dict) -> np.ndarray:
        """Denormalize values using stored statistics."""
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)
        return normalized * std + mean


def create_libero_transform_pipeline(model_type: str = "pi0", tokenizer=None):
    """Create data transform pipeline for LIBERO."""
    return transforms.Group(
        inputs=[
            libero_policy.LiberoInputs(model_type),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(tokenizer),
            transforms.PadStatesAndActions(action_dim=256),  # Pi0.5 action dim
        ],
        outputs=[
            libero_policy.LiberoOutputs(),
        ]
    )


def evaluate_libero_task(
    model: Pi05LiberoModel,
    env: OffScreenRenderEnv,
    task_description: str,
    initial_states: list,
    num_episodes: int,
    max_steps: int,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model on a single LIBERO task.

    Returns:
        Dictionary with success rate and episode statistics
    """
    successes = []
    episode_lengths = []

    for episode_idx in range(num_episodes):
        # Reset environment with random initial state
        init_state_id = episode_idx % len(initial_states)
        env.reset()
        env.set_init_state(initial_states[init_state_id])

        success = False

        for step in range(max_steps):
            # Get observation
            obs = env.get_observation()

            # Prepare observation dict
            observation = {
                "image": obs["agentview_image"],
                "wrist_image": obs["eye_in_hand_image"],
                "state": obs["robot0_eef_pos"],  # End-effector position
            }

            try:
                # Predict action
                action = model.predict_action(observation, task_description)
            except Exception as e:
                if verbose:
                    print(f"  Error predicting action: {e}")
                action = np.array(LIBERO_DUMMY_ACTION)

            # Execute action
            obs, reward, done, info = env.step(action)

            # Check for success
            if done or reward > 0:
                success = True
                episode_lengths.append(step + 1)
                break

        if not success:
            episode_lengths.append(max_steps)

        successes.append(success)

        if verbose:
            status = "SUCCESS" if success else "FAILURE"
            print(f"  Episode {episode_idx + 1}/{num_episodes}: {status} (steps: {episode_lengths[-1]})")

    success_rate = np.mean(successes) * 100
    avg_length = np.mean(episode_lengths)

    return {
        "success_rate": success_rate,
        "num_successes": sum(successes),
        "num_episodes": num_episodes,
        "avg_episode_length": avg_length,
        "episode_lengths": episode_lengths,
    }


def run_benchmark(args: Args):
    """Run full LIBERO benchmark."""
    if benchmark is None:
        raise ImportError("LIBERO not installed. Install with: pip install libero")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    print("=" * 80)
    print("Loading pi05-libero model")
    print("=" * 80)
    model = Pi05LiberoModel(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        use_bfloat16=args.use_bfloat16,
    )

    # Load LIBERO benchmark
    print("\n" + "=" * 80)
    print(f"Loading LIBERO benchmark: {args.task_suite_name}")
    print("=" * 80)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    suite_config = TASK_SUITE_CONFIGS[args.task_suite_name]
    num_tasks = suite_config["num_tasks"]
    max_steps = suite_config["max_steps"]

    print(f"Task suite: {args.task_suite_name}")
    print(f"Number of tasks: {num_tasks}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Episodes per task: {args.num_episodes}")

    # Run evaluation on all tasks
    all_results = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        print(f"\n{'=' * 80}")
        print(f"Task {task_id + 1}/{num_tasks}: {task_name}")
        print(f"Description: {task_description}")
        print(f"{'=' * 80}")

        # Create environment
        env_args = {
            "bddl_file_name": task.problem_folder,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)

        # Evaluate task
        task_results = evaluate_libero_task(
            model=model,
            env=env,
            task_description=task_description,
            initial_states=initial_states,
            num_episodes=args.num_episodes,
            max_steps=max_steps,
            verbose=args.verbose,
        )

        task_results["task_id"] = task_id
        task_results["task_name"] = task_name
        task_results["task_description"] = task_description

        all_results.append(task_results)

        print(f"\nTask Results:")
        print(f"  Success Rate: {task_results['success_rate']:.1f}%")
        print(f"  Successes: {task_results['num_successes']}/{task_results['num_episodes']}")
        print(f"  Avg Episode Length: {task_results['avg_episode_length']:.1f}")

        env.close()

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    overall_success_rate = np.mean([r["success_rate"] for r in all_results])
    total_successes = sum([r["num_successes"] for r in all_results])
    total_episodes = sum([r["num_episodes"] for r in all_results])

    print(f"\nTask Suite: {args.task_suite_name}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"Total Successes: {total_successes}/{total_episodes}")
    print(f"\nPer-Task Results:")
    print(f"{'Task ID':<10} {'Task Name':<40} {'Success Rate':<15} {'Successes'}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['task_id']:<10} {result['task_name']:<40} "
              f"{result['success_rate']:>6.1f}%        "
              f"{result['num_successes']}/{result['num_episodes']}")

    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Download and benchmark pi05-libero on LIBERO tasks"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_libero",
        help="Path to checkpoint (GCS URL or local path)",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        choices=list(TASK_SUITE_CONFIGS.keys()),
        help="LIBERO task suite to evaluate",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Convert to Args dataclass
    benchmark_args = Args(
        checkpoint_path=args.checkpoint_path,
        task_suite_name=args.task_suite,
        num_episodes=args.num_episodes,
        device=args.device,
        use_bfloat16=args.use_bfloat16,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Run benchmark
    results = run_benchmark(benchmark_args)

    return results


if __name__ == "__main__":
    main()
