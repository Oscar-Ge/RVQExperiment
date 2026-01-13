"""
Analyze π0.5's actions on LIBERO and test DCT compression.

This script:
1. Runs π0.5 on a few LIBERO tasks
2. Collects all action chunks
3. Tests DCT compression with different settings
4. Plots MSE vs. compression ratio
5. Analyzes compression across different task phases

Usage:
    python analyze_libero_actions.py --num_episodes 20 --task_suite libero_spatial
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add paths for imports
BASIC_RUN_PATH = Path(__file__).parent / "basic-run"
if BASIC_RUN_PATH.exists():
    sys.path.insert(0, str(BASIC_RUN_PATH))

from minimal_dct_tokenizer import MinimalDCTTokenizer

# Import from existing baseline code
try:
    from run_pi05_libero_benchmark_pytorch import (
        load_pi05_libero_policy,
        TASK_SUITES,
        LIBERO_ENV_RESOLUTION,
        LIBERO_RESIZE_SIZE,
        LIBERO_DUMMY_ACTION,
    )
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.benchmark as benchmark
    from openpi_client import image_tools
    import torch
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please make sure you have installed all dependencies and LIBERO is available.")
    print("See basic-run/QUICK_START.md for installation instructions.")
    sys.exit(1)


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation (from baseline code)."""
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    # Extract angle
    angle = 2 * np.arccos(np.clip(quat[0], -1, 1))
    # Extract axis
    s = np.sqrt(1 - quat[0] ** 2)
    if s < 1e-8:
        # Angle is 0, axis doesn't matter
        axis = np.array([1, 0, 0])
    else:
        axis = quat[1:] / s
    # Return axis * angle
    return axis * angle


def collect_pi05_actions(
    task_suite_name="libero_spatial",
    task_id=0,
    num_episodes=20,
    max_steps=220,
    pytorch_device="cuda",
    verbose=True,
):
    """
    Run π0.5 on LIBERO and collect all action chunks.

    Args:
        task_suite_name: LIBERO task suite (e.g., "libero_spatial")
        task_id: Which task to run (0-9 for most suites)
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        pytorch_device: Device for PyTorch ("cuda" or "cpu")
        verbose: Print progress

    Returns:
        all_action_chunks: List of [16, 7] numpy arrays
        metadata: Dictionary with task info and statistics
    """
    if verbose:
        print("=" * 80)
        print("COLLECTING π0.5 ACTIONS ON LIBERO")
        print("=" * 80)

    # Load policy
    if verbose:
        print("\n[1/4] Loading π0.5 policy...")
    policy = load_pi05_libero_policy(pytorch_device=pytorch_device)

    # Load task
    if verbose:
        print(f"\n[2/4] Loading LIBERO task: {task_suite_name} - Task {task_id}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_description = task.language
    initial_states = task_suite.get_task_init_states(task_id)

    if verbose:
        print(f"  Task: {task.name}")
        print(f"  Description: {task_description}")
        print(f"  Initial states: {len(initial_states)}")

    # Create environment
    if verbose:
        print("\n[3/4] Creating LIBERO environment...")

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": LIBERO_ENV_RESOLUTION,
        "camera_widths": LIBERO_ENV_RESOLUTION,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(42)

    # Collect actions
    if verbose:
        print(f"\n[4/4] Collecting actions from {num_episodes} episodes...")

    all_action_chunks = []
    episode_successes = []
    episode_lengths = []
    filtered_chunks = 0  # Track chunks with incorrect shape

    for episode in range(num_episodes):
        # Reset environment
        init_state_idx = episode % len(initial_states)
        env.reset()
        obs = env.set_init_state(initial_states[init_state_idx])

        done = False
        step = 0
        episode_actions = []

        while not done and step < max_steps:
            # Skip first few steps (objects settling)
            if step < 10:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                step += 1
                continue

            # Preprocess images
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, LIBERO_RESIZE_SIZE, LIBERO_RESIZE_SIZE)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, LIBERO_RESIZE_SIZE, LIBERO_RESIZE_SIZE)
            )

            # Prepare state
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            # Get action chunk from policy
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(task_description),
            }

            policy_output = policy.infer(element)
            action_chunk = policy_output["actions"]  # Expected [16, 7]

            # Debug: print shape on first occurrence
            if len(all_action_chunks) == 0 and filtered_chunks == 0:
                print(f"\n  [DEBUG] First action_chunk shape: {action_chunk.shape}")
                print(f"  [DEBUG] action_chunk dtype: {action_chunk.dtype}")
                print(f"  [DEBUG] action_chunk type: {type(action_chunk)}")

            # Accept all valid action chunks from the policy
            # π0.5-libero outputs 10-step chunks, not 16-step
            if hasattr(action_chunk, 'shape'):
                if len(action_chunk.shape) == 2 and action_chunk.shape[1] == 7:
                    # Accept any reasonable chunk size (8-16 steps)
                    if 8 <= action_chunk.shape[0] <= 16:
                        all_action_chunks.append(action_chunk.copy())
                    else:
                        # Skip chunks with unexpected length
                        filtered_chunks += 1
                        if filtered_chunks == 1:
                            print(f"  [WARNING] Unexpected chunk length: {action_chunk.shape[0]}")
                else:
                    filtered_chunks += 1
                    if filtered_chunks == 1:
                        print(f"  [DEBUG] Unexpected shape: {action_chunk.shape}")

            episode_actions.append(action_chunk[0])

            # Execute first action
            obs, reward, done, info = env.step(action_chunk[0].tolist())
            step += 1

        episode_successes.append(done)
        episode_lengths.append(step)

        if verbose:
            status = "✅" if done else "❌"
            print(f"  Episode {episode + 1}/{num_episodes}: {status} (steps: {step})")

    env.close()

    # Compute metadata
    metadata = {
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "task_description": task_description,
        "num_episodes": num_episodes,
        "num_chunks": len(all_action_chunks),
        "success_rate": np.mean(episode_successes) * 100,
        "avg_episode_length": np.mean(episode_lengths),
    }

    if verbose:
        print(f"\n✅ Collected {len(all_action_chunks)} action chunks")
        if filtered_chunks > 0:
            print(f"   (Filtered {filtered_chunks} chunks with incorrect shape)")
        print(f"   Success rate: {metadata['success_rate']:.1f}%")
        print(f"   Avg episode length: {metadata['avg_episode_length']:.1f} steps")

    return all_action_chunks, metadata


def analyze_compression(action_chunks, dct_keep_values=None, verbose=True):
    """
    Test different DCT compression settings.

    Args:
        action_chunks: List of [N, 7] numpy arrays (N determined by policy)
        dct_keep_values: List of num_dct_keep values to test
        verbose: Print detailed results

    Returns:
        results: List of dictionaries with compression metrics
    """
    # Detect chunk size from first action chunk
    if len(action_chunks) == 0:
        raise ValueError("No action chunks to analyze!")

    chunk_size = action_chunks[0].shape[0]
    if verbose:
        print(f"Detected action chunk size: {chunk_size} (from policy output)")

    if dct_keep_values is None:
        # Adjust default values based on chunk size
        if chunk_size <= 10:
            dct_keep_values = [2, 3, 4, 6, 8, 10]
        else:
            dct_keep_values = [2, 4, 6, 8, 12, 16]

    if verbose:
        print("\n" + "=" * 80)
        print("ANALYZING COMPRESSION WITH DIFFERENT SETTINGS")
        print("=" * 80)

    results = []

    for num_dct_keep in dct_keep_values:
        tokenizer = MinimalDCTTokenizer(chunk_size=chunk_size, num_dct_keep=num_dct_keep)
        tokenizer.fit(action_chunks)

        mses = []
        token_lengths = []

        for actions in action_chunks:
            tokens = tokenizer.encode(actions)
            reconstructed = tokenizer.decode(tokens)

            mse = np.mean((actions - reconstructed) ** 2)
            mses.append(mse)
            token_lengths.append(len(tokens))

        avg_mse = np.mean(mses)
        std_mse = np.std(mses)
        avg_token_len = np.mean(token_lengths)
        compression_ratio = (chunk_size * 7) / avg_token_len

        if verbose:
            print(f"\nDCT keep={num_dct_keep:2d}:")
            print(f"  MSE: {avg_mse:.6f} ± {std_mse:.6f}")
            print(f"  Tokens: {avg_token_len:.1f}")
            print(f"  Compression: {compression_ratio:.2f}x")

        results.append({
            'dct_keep': num_dct_keep,
            'mse': avg_mse,
            'mse_std': std_mse,
            'token_len': avg_token_len,
            'compression': compression_ratio,
        })

    return results


def plot_results(results, metadata, output_path="dct_compression_analysis.png"):
    """Plot MSE vs. compression ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MSE vs DCT coefficients
    dct_keeps = [r['dct_keep'] for r in results]
    mses = [r['mse'] for r in results]
    mse_stds = [r['mse_std'] for r in results]

    ax1.errorbar(dct_keeps, mses, yerr=mse_stds, marker='o', linewidth=2, capsize=5)
    ax1.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='Target MSE=0.01')
    ax1.axhline(y=0.001, color='g', linestyle='--', linewidth=2, label='Excellent MSE=0.001')
    ax1.set_xlabel('Number of DCT Coefficients Kept', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Reconstruction Error vs. Compression', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Compression ratio vs DCT coefficients
    compressions = [r['compression'] for r in results]
    ax2.plot(dct_keeps, compressions, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of DCT Coefficients Kept', fontsize=12)
    ax2.set_ylabel('Compression Ratio', fontsize=12)
    ax2.set_title('Compression Achieved', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add metadata to plot
    info_text = (
        f"Task: {metadata['task_name']}\n"
        f"Episodes: {metadata['num_episodes']}\n"
        f"Action chunks: {metadata['num_chunks']}\n"
        f"Success rate: {metadata['success_rate']:.1f}%"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DCT compression on π0.5 LIBERO actions")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=list(TASK_SUITES.keys()),
                        help="LIBERO task suite to use")
    parser.add_argument("--task_id", type=int, default=0,
                        help="Task ID within the suite (0-9 for most suites)")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes to collect")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="dct_compression_analysis.png",
                        help="Output plot filename")

    args = parser.parse_args()

    print("=" * 80)
    print("π0-FAST DCT COMPRESSION ANALYSIS")
    print("=" * 80)

    # Step 1: Collect actions
    action_chunks, metadata = collect_pi05_actions(
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        num_episodes=args.num_episodes,
        max_steps=TASK_SUITES[args.task_suite]["max_steps"],
        pytorch_device=args.device,
        verbose=True,
    )

    # Step 2: Analyze compression
    results = analyze_compression(action_chunks, verbose=True)

    # Step 3: Visualize
    plot_results(results, metadata, output_path=args.output)

    # Step 4: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find optimal setting (closest to MSE=0.01)
    best = min(results, key=lambda r: abs(r['mse'] - 0.01))
    print(f"\n✅ Optimal setting: Keep {best['dct_keep']} DCT coefficients")
    print(f"   MSE: {best['mse']:.6f} ± {best['mse_std']:.6f}")
    print(f"   Compression: {best['compression']:.2f}x")
    print(f"   Tokens per chunk: {best['token_len']:.1f}")

    # Analysis
    excellent_settings = [r for r in results if r['mse'] < 0.01]
    if excellent_settings:
        best_compression = max(excellent_settings, key=lambda r: r['compression'])
        print(f"\n🎯 Best compression with excellent MSE (<0.01):")
        print(f"   Keep {best_compression['dct_keep']} coefficients")
        print(f"   MSE: {best_compression['mse']:.6f}")
        print(f"   Compression: {best_compression['compression']:.2f}x")
        print(f"   → Autoregressive decoding could be {best_compression['compression']:.1f}x faster!")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("✅ DCT compression proof of concept complete!")
    print("\nPotential next steps:")
    print("  1. Add BPE (Byte-Pair Encoding) to further compress tokens")
    print("  2. Train an autoregressive model to predict these tokens")
    print("  3. Implement Residual Speculative Decoding")
    print("  4. Analyze task-aware compression (different settings for reach vs grasp)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
