"""
Analyze RVQ compression on trained tokenizer.

This script:
1. Loads a trained RVQ tokenizer
2. Tests reconstruction with different numbers of layers (1-8)
3. Plots MSE vs. number of layers (similar to DCT analysis)

Usage:
    python analyze_rvq_compression.py --model rvq_tokenizer.pt --task_suite libero_spatial --num_episodes 20
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add paths
BASIC_RUN_PATH = Path(__file__).parent / "basic-run"
if BASIC_RUN_PATH.exists():
    sys.path.insert(0, str(BASIC_RUN_PATH))

from rvq_tokenizer import RVQTokenizer
from analyze_libero_actions import collect_pi05_actions, TASK_SUITES


def load_rvq_tokenizer(model_path, device="cuda"):
    """
    Load trained RVQ tokenizer from checkpoint.

    Args:
        model_path: Path to saved model
        device: Device to load model on

    Returns:
        model: Loaded RVQTokenizer
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = RVQTokenizer(
        action_dim=config['action_dim'],
        chunk_size=config['chunk_size'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        num_embeddings=config['num_embeddings'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded RVQ tokenizer from {model_path}")
    print(f"  Config: {config}")

    return model


def analyze_rvq_compression(model, action_chunks, device="cuda", verbose=True):
    """
    Test RVQ compression with different numbers of layers.

    Args:
        model: Trained RVQTokenizer
        action_chunks: List of [chunk_size, action_dim] numpy arrays
        device: Device to run on
        verbose: Print detailed results

    Returns:
        results: List of dictionaries with compression metrics
    """
    if verbose:
        print("\n" + "=" * 80)
        print("ANALYZING RVQ COMPRESSION")
        print("=" * 80)

    results = []

    # Test from 1 to num_layers
    for num_layers in range(1, model.num_layers + 1):
        mses = []
        token_counts = []

        for action in action_chunks:
            # Encode with only first `num_layers` layers
            tokens = model.encode(action, num_layers=num_layers)

            # Decode
            reconstructed = model.decode(tokens)

            # Compute MSE
            mse = np.mean((action - reconstructed) ** 2)
            mses.append(mse)

            # Count tokens
            total_tokens = sum(len(t) for t in tokens)
            token_counts.append(total_tokens)

        avg_mse = np.mean(mses)
        std_mse = np.std(mses)
        avg_tokens = np.mean(token_counts)
        compression_ratio = model.get_compression_ratio(num_layers)

        if verbose:
            print(f"\nLayers={num_layers}:")
            print(f"  MSE: {avg_mse:.6f} ± {std_mse:.6f}")
            print(f"  Tokens: {avg_tokens:.1f}")
            print(f"  Compression: {compression_ratio:.2f}x")

        results.append({
            'num_layers': num_layers,
            'mse': avg_mse,
            'mse_std': std_mse,
            'token_count': avg_tokens,
            'compression': compression_ratio,
        })

    return results


def plot_rvq_results(results, metadata, output_path="rvq_compression_analysis.png"):
    """
    Plot MSE vs. number of RVQ layers.

    Args:
        results: List of result dictionaries
        metadata: Task metadata
        output_path: Output path for plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MSE vs layers
    num_layers = [r['num_layers'] for r in results]
    mses = [r['mse'] for r in results]
    mse_stds = [r['mse_std'] for r in results]

    ax1.errorbar(num_layers, mses, yerr=mse_stds, marker='o', linewidth=2, capsize=5)
    ax1.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='Target MSE=0.01')
    ax1.axhline(y=0.001, color='g', linestyle='--', linewidth=2, label='Excellent MSE=0.001')
    ax1.set_xlabel('Number of RVQ Layers', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('RVQ Reconstruction Error', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xticks(num_layers)

    # Compression ratio vs layers
    compressions = [r['compression'] for r in results]
    ax2.plot(num_layers, compressions, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of RVQ Layers', fontsize=12)
    ax2.set_ylabel('Compression Ratio', fontsize=12)
    ax2.set_title('Compression Achieved', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(num_layers)

    # Add metadata
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
    parser = argparse.ArgumentParser(description="Analyze RVQ compression")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained RVQ model")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=list(TASK_SUITES.keys()),
                        help="LIBERO task suite")
    parser.add_argument("--task_id", type=int, default=0,
                        help="Task ID")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="rvq_compression_analysis.png",
                        help="Output plot path")

    args = parser.parse_args()

    print("=" * 80)
    print("RVQ COMPRESSION ANALYSIS")
    print("=" * 80)

    # Step 1: Load model
    print("\n[1/3] Loading trained RVQ model...")
    model = load_rvq_tokenizer(args.model, device=args.device)

    # Step 2: Collect test data
    print("\n[2/3] Collecting test actions...")
    action_chunks, metadata = collect_pi05_actions(
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        num_episodes=args.num_episodes,
        max_steps=TASK_SUITES[args.task_suite]["max_steps"],
        pytorch_device=args.device,
        verbose=True,
    )

    # Step 3: Analyze compression
    print("\n[3/3] Analyzing compression...")
    results = analyze_rvq_compression(model, action_chunks, device=args.device, verbose=True)

    # Step 4: Visualize
    plot_rvq_results(results, metadata, output_path=args.output)

    # Step 5: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find optimal setting
    best = min(results, key=lambda r: abs(r['mse'] - 0.01))
    print(f"\n✅ Optimal setting: {best['num_layers']} RVQ layers")
    print(f"   MSE: {best['mse']:.6f} ± {best['mse_std']:.6f}")
    print(f"   Compression: {best['compression']:.2f}x")
    print(f"   Tokens per chunk: {best['token_count']:.1f}")

    # Find excellent settings
    excellent_settings = [r for r in results if r['mse'] < 0.01]
    if excellent_settings:
        best_compression = max(excellent_settings, key=lambda r: r['compression'])
        print(f"\n🎯 Best compression with excellent MSE (<0.01):")
        print(f"   {best_compression['num_layers']} layers")
        print(f"   MSE: {best_compression['mse']:.6f}")
        print(f"   Compression: {best_compression['compression']:.2f}x")
        print(f"   → Coarse layers (1-{best_compression['num_layers']}) capture key motion!")

    # Compare layers
    if len(results) >= 2:
        layer2 = results[1]  # 2 layers
        layer8 = results[-1]  # 8 layers
        print(f"\n📊 Layer comparison:")
        print(f"   Layer 1-2 (coarse): MSE={layer2['mse']:.6f}")
        print(f"   Layer 1-8 (fine):   MSE={layer8['mse']:.6f}")
        print(f"   → Improvement from fine layers: {(layer2['mse'] - layer8['mse']) / layer2['mse'] * 100:.1f}%")

        if layer2['mse'] < 0.01:
            print(f"\n✅ HYPOTHESIS VALIDATED!")
            print(f"   Layer 1-2 alone achieve MSE < 0.01")
            print(f"   → Can use coarse layers for simple motions")
            print(f"   → Only activate fine layers for complex tasks")
        else:
            print(f"\n⚠️  HYPOTHESIS NEEDS REVISION")
            print(f"   Layer 1-2 MSE = {layer2['mse']:.6f} > 0.01")
            print(f"   → May need more coarse layers or task-specific analysis")

    # Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("✅ RVQ compression analysis complete!")
    print("\nBased on results:")

    if excellent_settings and excellent_settings[0]['num_layers'] <= 3:
        print("  ✅ Ready for Phase 2: Train VLA policy to predict RVQ tokens")
        print("  ✅ Ready for Phase 3: Implement adaptive inference")
        print(f"     - Use layers 1-{excellent_settings[0]['num_layers']} for coarse prediction")
        print("     - Activate all layers when uncertainty is high")
    elif excellent_settings:
        print(f"  ⚠️  Need {excellent_settings[0]['num_layers']} layers for good MSE")
        print("  → Consider:")
        print("     1. Increasing codebook size")
        print("     2. Training longer")
        print("     3. Using larger hidden_dim")
    else:
        print("  ⚠️  No configuration achieves MSE < 0.01")
        print("  → Debug:")
        print("     1. Check training curves")
        print("     2. Increase model capacity")
        print("     3. Collect more training data")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
