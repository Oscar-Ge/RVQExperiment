#!/usr/bin/env python3
"""
Phase 1 Training Script: Train Robust RFSQ AutoEncoder

Converts modal_train_rfsq_robust.py to standalone SLURM-compatible script.
Trains an improved RFSQ AutoEncoder using the LayerNorm strategy.

Expected Results:
- Naive RFSQ MSE: ~0.018
- Robust RFSQ MSE: ~0.010 (-44% improvement)
- Target MSE: < 0.012

Usage:
    python train_phase1_rfsq.py --epochs 100 --batch_size 64 --learning_rate 1e-3
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import local modules
from models.rfsq_models import ActionRFSQAE
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from experiment_logger import ExperimentLogger
from checkpoint_manager import CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Phase 1 Robust RFSQ AutoEncoder")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of RFSQ layers')
    parser.add_argument('--num_levels', type=int, default=7, help='Number of quantization levels')
    parser.add_argument('--use_layernorm', type=bool, default=True, help='Use LayerNorm strategy')
    parser.add_argument('--entropy_weight', type=float, default=0.05, help='Entropy regularization weight')

    # Paths (from environment variables or defaults)
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--model_dir', type=str, default=None, help='Model output directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')

    # Other options
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (10 epochs only)')

    return parser.parse_args()


def load_training_data(data_path):
    """Load and prepare training data."""
    print(f"\n📦 Loading training data from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")

    data = torch.load(data_path, weights_only=False)

    # Handle different data formats
    if isinstance(data, dict):
        if 'actions' in data:
            actions = data['actions']
        elif 'data' in data:
            actions = data['data']
        else:
            actions = list(data.values())[0]
    else:
        actions = data

    # Ensure correct shape [N, action_dim]
    if len(actions.shape) == 3:
        # [N, chunk, action_dim] -> [N*chunk, action_dim]
        actions = actions.view(-1, actions.shape[-1])

    action_dim = actions.shape[-1]
    print(f"✅ Loaded {len(actions)} action samples with dim={action_dim}")

    return actions, action_dim


def create_dataloader(actions, batch_size):
    """Create PyTorch DataLoader."""
    dataset = TensorDataset(actions.float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def train_epoch(model, dataloader, optimizer, device, num_levels, entropy_weight):
    """Train for one epoch."""
    model.train()
    total_mse = 0.0
    total_entropy = 0.0
    num_batches = 0

    for batch_idx, (x,) in enumerate(dataloader):
        x = x.to(device)

        # Add sequence dimension if needed [B, A] -> [B, 1, A]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Forward pass
        x_recon, codes = model(x)

        # Reconstruction loss (MSE)
        loss_mse = F.mse_loss(x_recon, x)

        # Entropy regularization (prevent codebook collapse)
        codes_flat = codes.view(-1)
        one_hot = F.one_hot(codes_flat, num_classes=num_levels).float()
        avg_probs = torch.mean(one_hot, dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        loss_entropy = -entropy  # Maximize entropy

        # Total loss
        loss = loss_mse + entropy_weight * loss_entropy

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse += loss_mse.item()
        total_entropy += loss_entropy.item()
        num_batches += 1

    avg_mse = total_mse / num_batches
    avg_entropy = total_entropy / num_batches

    return avg_mse, avg_entropy


def main():
    """Main training function."""
    args = parse_args()

    # Debug mode
    if args.debug:
        print("🔧 Debug mode: Running 10 epochs only")
        args.epochs = 10

    # Get paths from environment variables or args
    data_dir = args.data_dir or os.getenv('RVQ_DATA_DIR', './data')
    model_dir = args.model_dir or os.getenv('RVQ_MODEL_DIR', './models')
    log_dir = args.log_dir or os.getenv('RVQ_LOG_DIR', './logs')

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"📊 Training Robust RFSQ with LayerNorm={args.use_layernorm}")

    # Initialize experiment logger
    logger = ExperimentLogger(
        name="phase1_robust_rfsq_training",
        log_dir=log_dir,
        config={
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_levels": args.num_levels,
            "use_layernorm": args.use_layernorm,
            "entropy_weight": args.entropy_weight,
            "device": str(device),
            "target_mse": 0.012,
        }
    )

    # Load training data
    data_path = os.path.join(data_dir, "libero_actions_normalized.pt")
    actions, action_dim = load_training_data(data_path)
    logger.log_text(f"Loaded {len(actions)} samples, action_dim={action_dim}")

    # Create dataloader
    dataloader = create_dataloader(actions, args.batch_size)

    # Initialize model
    print(f"\n📐 Creating RFSQ AutoEncoder...")
    model = ActionRFSQAE(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_levels=args.num_levels,
        use_layernorm=args.use_layernorm,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters")
    print(f"   - Action dim: {action_dim}")
    print(f"   - Hidden dim: {args.hidden_dim}")
    print(f"   - Num layers: {args.num_layers}")
    print(f"   - Num levels: {args.num_levels}")
    print(f"   - Use LayerNorm: {args.use_layernorm}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize checkpoint manager
    checkpoint_dir = os.path.join(model_dir, "phase1_rfsq_checkpoints")
    ckpt_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_n_best=3,
        metric_name="mse",
        mode="min"
    )

    # Training loop
    print(f"\n🔥 Starting training for {args.epochs} epochs...")
    best_mse = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train one epoch
        avg_mse, avg_entropy = train_epoch(
            model, dataloader, optimizer, device,
            args.num_levels, args.entropy_weight
        )

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - start_time

        # Log metrics
        logger.log({
            'mse': avg_mse,
            'entropy_loss': avg_entropy,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
        }, step=epoch)

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"MSE: {avg_mse:.6f} | "
              f"Entropy: {avg_entropy:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_epoch = epoch + 1

            print(f"   ✅ New best model! MSE: {avg_mse:.6f}")
            logger.log_text(f"New best model at epoch {epoch+1}, MSE={avg_mse:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or avg_mse == best_mse:
            ckpt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=(epoch + 1) * len(dataloader),
                metrics={'mse': avg_mse, 'entropy_loss': avg_entropy},
                extra_state={
                    'config': {
                        'action_dim': action_dim,
                        'hidden_dim': args.hidden_dim,
                        'num_layers': args.num_layers,
                        'num_levels': args.num_levels,
                        'use_layernorm': args.use_layernorm,
                    }
                }
            )

    # Save final best model with simple name
    best_model_path = os.path.join(model_dir, "rfsq_robust_best.pt")
    best_checkpoint_path = ckpt_manager.get_best_checkpoint_path()
    if best_checkpoint_path:
        best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        torch.save(best_checkpoint, best_model_path)
        print(f"\n✅ Best model saved to: {best_model_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"   Best MSE: {best_mse:.6f} (at epoch {best_epoch})")
    print(f"   Target MSE: < 0.012")
    print(f"   Status: {'✅ TARGET MET!' if best_mse < 0.012 else '⚠️ Target not met'}")
    print(f"   Checkpoint: {best_model_path}")
    print("=" * 60)

    # Log final results
    logger.log({
        'final_best_mse': best_mse,
        'best_epoch': best_epoch,
        'target_met': best_mse < 0.012,
    }, step=args.epochs)

    logger.log_text(f"Training complete! Best MSE: {best_mse:.6f} at epoch {best_epoch}")

    if best_mse < 0.012:
        logger.finish('completed')
    else:
        logger.log_text(f"Warning: Target MSE < 0.012 not met. Best: {best_mse:.6f}")
        logger.finish('completed')

    return best_mse, best_epoch


if __name__ == "__main__":
    main()
