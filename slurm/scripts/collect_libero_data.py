"""
Collect LIBERO Action Data for Phase 1 RFSQ Training

This script collects action trajectories from LIBERO using a random or scripted policy.
The collected data will be used to train the RFSQ AutoEncoder in Phase 1.

Usage:
    python collect_libero_data.py --task-suite libero_spatial --num-episodes 100 --output-path ../data/libero_actions_normalized.pt
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def normalize_actions(actions, min_val=-1.0, max_val=1.0):
    """Normalize actions to [min_val, max_val] range."""
    actions_min = actions.min(axis=0, keepdims=True)
    actions_max = actions.max(axis=0, keepdims=True)

    # Avoid division by zero
    range_vals = actions_max - actions_min
    range_vals[range_vals < 1e-6] = 1.0

    # Normalize to [0, 1] then scale to [min_val, max_val]
    normalized = (actions - actions_min) / range_vals
    normalized = normalized * (max_val - min_val) + min_val

    return normalized, actions_min, actions_max


def collect_random_actions(task, env, num_episodes, max_steps=220, chunk_size=8):
    """
    Collect action trajectories using random policy.

    Args:
        task: LIBERO task
        env: LIBERO environment
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        chunk_size: Action chunk size for RFS Q

    Returns:
        all_chunks: List of action chunks [chunk_size, 7]
    """
    all_chunks = []

    print(f"\nCollecting {num_episodes} episodes of random actions...")

    for ep in tqdm(range(num_episodes)):
        # Reset environment
        env.reset()
        init_states = task.get_init_states()
        env.set_init_state(init_states[0])

        episode_actions = []

        for step in range(max_steps):
            # Random action in [-1, 1]
            action = np.random.uniform(-1, 1, size=7).astype(np.float32)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_actions.append(action)

            if done:
                break

        # Convert to numpy array
        episode_actions = np.array(episode_actions)  # [T, 7]

        # Split into chunks
        num_chunks = len(episode_actions) // chunk_size
        for i in range(num_chunks):
            chunk = episode_actions[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) == chunk_size:
                all_chunks.append(chunk)

    print(f"✅ Collected {len(all_chunks)} action chunks")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Collect LIBERO action data for RFSQ training")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        help="LIBERO task suite")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID within suite")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=220, help="Max steps per episode")
    parser.add_argument("--chunk-size", type=int, default=8, help="Action chunk size")
    parser.add_argument("--output-path", type=str, default=None, help="Output file path (default: data/libero_actions_normalized.pt)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set output path
    if args.output_path is None:
        data_dir = os.getenv('RVQ_DATA_DIR', str(Path(__file__).parent.parent.parent / 'data'))
        args.output_path = os.path.join(data_dir, 'libero_actions_normalized.pt')

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("LIBERO Action Data Collection for Phase 1")
    print("=" * 80)
    print(f"Task Suite: {args.task_suite}")
    print(f"Task ID: {args.task_id}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Output: {args.output_path}")
    print("=" * 80)

    # Load LIBERO task suite
    print("\n[1/3] Loading LIBERO task suite...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    tasks = task_suite.get_task(args.task_id)

    if not isinstance(tasks, list):
        tasks = [tasks]

    task = tasks[0]
    print(f"✅ Loaded task: {task.name}")

    # Create environment
    print("\n[2/3] Creating LIBERO environment...")
    env_args = {
        "bddl_file_name": task.bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)
    print("✅ Environment created")

    # Collect data
    print("\n[3/3] Collecting action data...")
    all_chunks = collect_random_actions(
        task=task,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        chunk_size=args.chunk_size,
    )

    # Convert to numpy array
    all_chunks = np.array(all_chunks)  # [N, chunk_size, 7]
    print(f"\nCollected data shape: {all_chunks.shape}")
    print(f"  Number of chunks: {len(all_chunks)}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Action dim: 7")

    # Normalize actions
    print("\nNormalizing actions to [-1, 1]...")
    all_chunks_flat = all_chunks.reshape(-1, 7)
    normalized_flat, data_min, data_max = normalize_actions(all_chunks_flat)
    normalized_chunks = normalized_flat.reshape(all_chunks.shape)

    print(f"  Original range: [{all_chunks_flat.min():.3f}, {all_chunks_flat.max():.3f}]")
    print(f"  Normalized range: [{normalized_flat.min():.3f}, {normalized_flat.max():.3f}]")

    # Save data
    print(f"\nSaving to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    torch.save({
        'actions': torch.from_numpy(normalized_chunks).float(),
        'normalization': {
            'min': torch.from_numpy(data_min).float(),
            'max': torch.from_numpy(data_max).float(),
        },
        'metadata': {
            'task_suite': args.task_suite,
            'task_id': args.task_id,
            'num_episodes': args.num_episodes,
            'chunk_size': args.chunk_size,
            'num_chunks': len(all_chunks),
            'seed': args.seed,
        }
    }, args.output_path)

    print("✅ Data saved successfully!")
    print("\n" + "=" * 80)
    print("Collection Complete!")
    print("=" * 80)
    print(f"Output: {args.output_path}")
    print(f"Total chunks: {len(all_chunks)}")
    print("\nNext step: Train Phase 1 RFSQ AutoEncoder")
    print("  sbatch slurm/jobs/1_phase1_rfsq.sbatch")
    print("=" * 80)


if __name__ == "__main__":
    main()
