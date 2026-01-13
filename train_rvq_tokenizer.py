"""
Train RVQ tokenizer on LIBERO actions collected from π0.5.

This script:
1. Collects actions from π0.5 on LIBERO
2. Trains an RVQ tokenizer
3. Saves the trained model

Usage:
    python train_rvq_tokenizer.py --task_suite libero_spatial --num_episodes 50 --epochs 100
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Add paths
BASIC_RUN_PATH = Path(__file__).parent / "basic-run"
if BASIC_RUN_PATH.exists():
    sys.path.insert(0, str(BASIC_RUN_PATH))

from rvq_tokenizer import RVQTokenizer

# Import from analyze_libero_actions.py
try:
    from analyze_libero_actions import collect_pi05_actions, TASK_SUITES
except ImportError as e:
    print(f"Error importing: {e}")
    print("Make sure analyze_libero_actions.py is in the same directory")
    sys.exit(1)


class ActionDataset(Dataset):
    """Simple dataset for action chunks."""

    def __init__(self, action_chunks):
        """
        Args:
            action_chunks: List of [chunk_size, action_dim] numpy arrays
        """
        self.actions = [torch.from_numpy(a).float() for a in action_chunks]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.actions[idx]


def train_rvq_tokenizer(
    action_chunks,
    num_layers=8,
    hidden_dim=64,
    num_embeddings=256,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    residual_dropout_prob=0.1,
    device="cuda",
    save_path="rvq_tokenizer.pt",
    verbose=True,
):
    """
    Train RVQ tokenizer on action data.

    Args:
        action_chunks: List of [chunk_size, action_dim] numpy arrays
        num_layers: Number of RVQ layers
        hidden_dim: Hidden dimension
        num_embeddings: Codebook size
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        residual_dropout_prob: Dropout probability for residual layers
        device: Device to train on
        save_path: Where to save model
        verbose: Print training progress

    Returns:
        trained_model: Trained RVQTokenizer
        history: Training history
    """
    # Detect chunk size from data
    chunk_size = action_chunks[0].shape[0]
    action_dim = action_chunks[0].shape[1]

    if verbose:
        print("=" * 80)
        print("TRAINING RVQ TOKENIZER")
        print("=" * 80)
        print(f"\nDataset:")
        print(f"  Action chunks: {len(action_chunks)}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Action dim: {action_dim}")
        print(f"\nModel config:")
        print(f"  Num layers: {num_layers}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Codebook size: {num_embeddings}")
        print(f"  Residual dropout: {residual_dropout_prob}")
        print(f"\nTraining config:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {device}")

    # Create model
    model = RVQTokenizer(
        action_dim=action_dim,
        chunk_size=chunk_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_embeddings=num_embeddings,
    ).to(device)

    # Fit normalization
    model.fit(action_chunks)

    # Create dataset and dataloader
    dataset = ActionDataset(action_chunks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (only for encoder/decoder, VQ uses EMA)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'recon_loss': [],
        'vq_loss': [],
        'total_loss': [],
    }

    # Training loop
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING PROGRESS")
        print("=" * 80)

    model.train()
    for epoch in range(epochs):
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            reconstructed, vq_loss, _ = model(
                batch,
                residual_dropout_prob=residual_dropout_prob
            )

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, batch)

            # Total loss
            loss = recon_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            num_batches += 1

        # Average losses
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        avg_total_loss = avg_recon_loss + avg_vq_loss

        # Record history
        history['recon_loss'].append(avg_recon_loss)
        history['vq_loss'].append(avg_vq_loss)
        history['total_loss'].append(avg_total_loss)

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Recon Loss: {avg_recon_loss:.6f}")
            print(f"  VQ Loss: {avg_vq_loss:.6f}")
            print(f"  Total Loss: {avg_total_loss:.6f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'action_dim': action_dim,
            'chunk_size': chunk_size,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'num_embeddings': num_embeddings,
        },
        'history': history,
    }, save_path)

    if verbose:
        print(f"\n✅ Model saved to {save_path}")

    return model, history


def plot_training_history(history, output_path="training_history.png"):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['recon_loss']) + 1)

    # Reconstruction loss
    ax1.plot(epochs, history['recon_loss'], 'b-', linewidth=2, label='Recon Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # VQ loss
    ax2.plot(epochs, history['vq_loss'], 'r-', linewidth=2, label='VQ Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('VQ Loss', fontsize=12)
    ax2.set_title('Vector Quantization Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RVQ tokenizer on mixed π0.5 LIBERO actions")
    parser.add_argument("--num_episodes_per_task", type=int, default=5,
                        help="Number of episodes per task")
    parser.add_argument("--max_tasks_per_suite", type=int, default=10,
                        help="Maximum task IDs to sample from each suite (0 to N-1)")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of RVQ layers")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension")
    parser.add_argument("--codebook_size", type=int, default=256,
                        help="Codebook size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--residual_dropout", type=float, default=0.1,
                        help="Residual dropout probability")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="rvq_tokenizer_mixed.pt",
                        help="Output model path")
    parser.add_argument("--target_suites", type=str, nargs="+",
                        default=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="Target task suites to mix")
    parser.add_argument("--skip_collect", action="store_true",
                        help="Skip data collection (use existing data)")

    args = parser.parse_args()

    print("=" * 80)
    print("RVQ TOKENIZER TRAINING (MULTI-DATASET)")
    print("=" * 80)

    # Step 1: Collect actions from multiple suites and tasks
    all_action_chunks = []
    all_metadata = []
    
    if args.skip_collect:
        print("\n⚠️  Skipping data collection (--skip_collect flag)")
        print("Please load existing action data manually")
        sys.exit(1)
    else:
        print(f"\n📊 Multi-dataset collection configuration:")
        print(f"  Target suites: {args.target_suites}")
        print(f"  Episodes per task: {args.num_episodes_per_task}")
        print(f"  Max tasks per suite: {args.max_tasks_per_suite}")
        print(f"  Device: {args.device}")
        print("\n" + "=" * 80)
        print("COLLECTING DATA FROM MULTIPLE DATASETS")
        print("=" * 80)

        for suite_name in args.target_suites:
            if suite_name not in TASK_SUITES:
                print(f"⚠️  Skipping unknown suite: {suite_name}")
                continue

            suite_config = TASK_SUITES[suite_name]
            max_steps = suite_config["max_steps"]
            num_tasks = suite_config.get("num_tasks", args.max_tasks_per_suite)
            
            print(f"\n📦 Suite: {suite_name}")
            print(f"   Total tasks in suite: {num_tasks}")
            print(f"   Tasks to sample: {min(args.max_tasks_per_suite, num_tasks)}")

            # Collect from each task in the suite
            for task_id in range(min(args.max_tasks_per_suite, num_tasks)):
                print(f"   Collecting Task {task_id}...", end=" ", flush=True)
                
                try:
                    action_chunks, metadata = collect_pi05_actions(
                        task_suite_name=suite_name,
                        task_id=task_id,
                        num_episodes=args.num_episodes_per_task,
                        max_steps=max_steps,
                        pytorch_device=args.device,
                        verbose=False,
                    )
                    
                    all_action_chunks.extend(action_chunks)
                    all_metadata.append({
                        'suite': suite_name,
                        'task_id': task_id,
                        'num_chunks': len(action_chunks),
                        'metadata': metadata,
                    })
                    
                    print(f"✅ ({len(action_chunks)} chunks)")
                    
                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    continue

        print("\n" + "=" * 80)
        print("DATA COLLECTION SUMMARY")
        print("=" * 80)
        print(f"✅ Total action chunks collected: {len(all_action_chunks)}")
        print(f"✅ Total tasks sampled: {len(all_metadata)}")
        
        for meta in all_metadata:
            print(f"   - {meta['suite']} (Task {meta['task_id']}): {meta['num_chunks']} chunks")

    # Step 2: Train RVQ tokenizer on combined dataset
    model, history = train_rvq_tokenizer(
        action_chunks=all_action_chunks,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_embeddings=args.codebook_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        residual_dropout_prob=args.residual_dropout,
        device=args.device,
        save_path=args.output,
        verbose=True,
    )

    # Step 3: Plot training curves
    plot_training_history(history, output_path="training_history_mixed.png")

    # Step 4: Quick validation
    print("\n" + "=" * 80)
    print("QUICK VALIDATION")
    print("=" * 80)

    model.eval()
    test_action = all_action_chunks[0]
    test_action_tensor = torch.from_numpy(test_action).unsqueeze(0).float().to(args.device)

    with torch.no_grad():
        reconstructed, _, _ = model(test_action_tensor)
        reconstructed_np = reconstructed[0].cpu().numpy()

    mse = np.mean((test_action - reconstructed_np) ** 2)
    print(f"\nTest reconstruction MSE: {mse:.6f}")

    if mse < 0.01:
        print("✅ Validation passed! MSE < 0.01")
    else:
        print("⚠️  Warning: MSE higher than expected")

    # Step 5: Summary
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("✅ Multi-dataset RVQ tokenizer trained successfully!")
    print(f"\nModel saved to: {args.output}")
    print(f"Training curves saved to: training_history_mixed.png")
    print("\nNext:")
    print(f"  1. Evaluate on held-out test tasks: python evaluate_generalization.py --model {args.output}")
    print("  2. Compare with single-suite baseline")
    print("  3. Analyze compression ratios across suites")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
