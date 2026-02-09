#!/usr/bin/env python3
"""
Phase 2 Conditional Training Script: Train Draft Model + Conditional Main Model

Converts modal_train_phase2_conditional.py to standalone SLURM-compatible script.
Trains TWO models:
1. Draft Model: Predicts coarse tokens (L0-L2) from image features
2. Conditional Main Model: Predicts all layers (L0-L7) conditioned on L0-L2 tokens

Usage:
    # Train both models
    python train_phase2_conditional.py --data_path /path/to/phase2_training_data.pt --epochs 50

    # Train only draft model
    python train_phase2_conditional.py --train_draft_only --epochs 50

    # Train only conditional model
    python train_phase2_conditional.py --train_conditional_only --epochs 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import local modules
from models.rfsq_models import RFSQDraftModelWithProjection, ConditionedRFSQHead
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from experiment_logger import ExperimentLogger
from checkpoint_manager import CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Phase 2 Conditional RFSQ Models")

    # Data path
    parser.add_argument('--data_path', type=str, required=True, help='Path to phase2_training_data.pt')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')

    # Draft model hyperparameters
    parser.add_argument('--draft_hidden_dim', type=int, default=512, help='Draft model hidden dimension')
    parser.add_argument('--draft_input_dim', type=int, default=4096, help='Draft model input dimension')

    # Conditional model hyperparameters
    parser.add_argument('--cond_hidden_dim', type=int, default=1024, help='Conditional model hidden dimension')
    parser.add_argument('--cond_input_dim', type=int, default=4096, help='Conditional model input dimension')
    parser.add_argument('--token_embed_dim', type=int, default=64, help='Token embedding dimension')

    # Model architecture hyperparameters
    parser.add_argument('--num_coarse_layers', type=int, default=3, help='Number of coarse layers (L0-L2)')
    parser.add_argument('--num_layers', type=int, default=8, help='Total number of RFSQ layers')
    parser.add_argument('--chunk_len', type=int, default=8, help='Action chunk length')
    parser.add_argument('--action_hidden_dim', type=int, default=16, help='RFSQ latent dimension')
    parser.add_argument('--grid_size', type=int, default=7, help='Quantization levels')

    # Paths (from environment variables or defaults)
    parser.add_argument('--model_dir', type=str, default=None, help='Model output directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')

    # Training options
    parser.add_argument('--train_draft_only', action='store_true', help='Train only draft model')
    parser.add_argument('--train_conditional_only', action='store_true', help='Train only conditional model')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    return parser.parse_args()


# ============================================================
# Dataset Classes
# ============================================================

class DraftDataset(Dataset):
    """Dataset for Draft Model training (predicts L0-L2)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Extract L0-L2 tokens
        coarse_tokens = sample['rfsq_tokens'][:, :, :3]  # [8, 16, 3]
        tokens_flat = coarse_tokens.reshape(-1, 3)  # [128, 3]

        return {
            'hidden': sample['hidden_state'],  # [4096]
            'tokens': tokens_flat,  # [128, 3]
        }


class ConditionedRFSQHeadDataset(Dataset):
    """Dataset for Conditional RFSQ Head training (predicts all 8 layers conditioned on L0-L2)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # sample['rfsq_tokens']: [Chunk=8, Hidden=16, Layers=8]

        # Extract L0-L2 for conditioning (Teacher Forcing)
        condition_tokens = sample['rfsq_tokens'][:, :, 0:3]  # [8, 16, 3]

        # All layers as labels
        label_tokens = sample['rfsq_tokens']  # [8, 16, 8]
        label_tokens_flat = label_tokens.reshape(-1, 8)  # [128, 8]

        return {
            'hidden': sample['hidden_state'],        # [4096]
            'condition': condition_tokens.long(),    # [8, 16, 3]
            'labels': label_tokens_flat.long(),      # [128, 8]
        }


# ============================================================
# Training Functions
# ============================================================

def train_draft_epoch(model, dataloader, optimizer, device):
    """Train draft model for one epoch."""
    model.train()
    total_loss = 0.0
    total_accs = [0.0, 0.0, 0.0]
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training Draft", leave=False):
        hidden = batch['hidden'].to(device)  # [B, 4096]
        targets = batch['tokens'].to(device)  # [B, 128, 3]

        # Forward pass
        logits = model(hidden)  # [B, 3, 128, 7]

        # Reshape for cross-entropy loss
        logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)  # [B*128*3, 7]
        targets_flat = targets.reshape(-1)  # [B*128*3]

        loss = F.cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Calculate per-layer accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)  # [B, 3, 128]
            for layer_idx in range(3):
                acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                total_accs[layer_idx] += acc.item()

    avg_loss = total_loss / num_batches
    avg_accs = [acc / num_batches for acc in total_accs]

    return avg_loss, avg_accs


def validate_draft(model, dataloader, device):
    """Validate draft model."""
    model.eval()
    total_loss = 0.0
    total_accs = [0.0, 0.0, 0.0]
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Draft", leave=False):
            hidden = batch['hidden'].to(device)
            targets = batch['tokens'].to(device)

            logits = model(hidden)

            logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            for layer_idx in range(3):
                acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                total_accs[layer_idx] += acc.item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accs = [acc / num_batches for acc in total_accs]

    return avg_loss, avg_accs


def train_conditional_epoch(model, dataloader, optimizer, device):
    """Train conditional model for one epoch."""
    model.train()
    total_loss = 0.0
    total_accs = [0.0] * 8
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training Conditional", leave=False):
        hidden = batch['hidden'].to(device)          # [B, 4096]
        condition = batch['condition'].to(device)     # [B, 8, 16, 3]
        targets = batch['labels'].to(device)          # [B, 128, 8]

        # Forward pass with conditioning
        logits = model(hidden, condition)  # [B, 8, 128, 7]

        # Reshape for cross-entropy loss
        logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)  # [B*128*8, 7]
        targets_flat = targets.reshape(-1)  # [B*128*8]

        loss = F.cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Calculate per-layer accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)  # [B, 8, 128]
            for layer_idx in range(8):
                acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                total_accs[layer_idx] += acc.item()

    avg_loss = total_loss / num_batches
    avg_accs = [acc / num_batches for acc in total_accs]

    return avg_loss, avg_accs


def validate_conditional(model, dataloader, device):
    """Validate conditional model."""
    model.eval()
    total_loss = 0.0
    total_accs = [0.0] * 8
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Conditional", leave=False):
            hidden = batch['hidden'].to(device)
            condition = batch['condition'].to(device)
            targets = batch['labels'].to(device)

            logits = model(hidden, condition)

            logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            for layer_idx in range(8):
                acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                total_accs[layer_idx] += acc.item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accs = [acc / num_batches for acc in total_accs]

    return avg_loss, avg_accs


# ============================================================
# Main Training Functions
# ============================================================

def train_draft_model(args, all_data, device, model_dir, log_dir):
    """Train Draft Model."""
    print("\n" + "=" * 80)
    print("🚀 Training Draft Model (L0-L2 Prediction)")
    print("=" * 80)

    # Initialize experiment logger
    logger = ExperimentLogger(
        name="phase2_draft_model_training",
        log_dir=log_dir,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "input_dim": args.draft_input_dim,
            "hidden_dim": args.draft_hidden_dim,
            "num_coarse_layers": args.num_coarse_layers,
            "device": str(device),
        }
    )

    # Create dataset
    dataset = DraftDataset(all_data)
    val_size = min(5000, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create model
    print(f"\n📐 Creating Draft Model...")
    model = RFSQDraftModelWithProjection(
        input_dim=args.draft_input_dim,
        hidden_dim=args.draft_hidden_dim,
        num_coarse_layers=args.num_coarse_layers,
        chunk_len=args.chunk_len,
        action_hidden_dim=args.action_hidden_dim,
        grid_size=args.grid_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params / 1e6:.1f}M parameters")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 10)

    # Checkpoint manager
    checkpoint_dir = os.path.join(model_dir, "draft_model_checkpoints")
    ckpt_manager = CheckpointManager(checkpoint_dir, keep_n_best=3, metric_name="val_acc", mode="max")

    # Training loop
    print(f"\n🔥 Training for {args.epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_accs = train_draft_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_accs = validate_draft(model, val_loader, device)

        scheduler.step()
        epoch_time = time.time() - start_time

        avg_val_acc = sum(val_accs) / 3

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: L0={train_accs[0]:.3f} L1={train_accs[1]:.3f} L2={train_accs[2]:.3f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: L0={val_accs[0]:.3f} L1={val_accs[1]:.3f} L2={val_accs[2]:.3f} | Avg={avg_val_acc:.3f}")

        # Log metrics
        logger.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc_L0': val_accs[0],
            'val_acc_L1': val_accs[1],
            'val_acc_L2': val_accs[2],
            'val_acc_avg': avg_val_acc,
            'learning_rate': scheduler.get_last_lr()[0],
        }, step=epoch)

        # Save checkpoint
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"  ✅ Best model saved: {avg_val_acc:.3f}")
            logger.log_text(f"New best model at epoch {epoch + 1} with accuracy {avg_val_acc:.3f}")

        if (epoch + 1) % args.save_every == 0 or avg_val_acc == best_val_acc:
            ckpt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=(epoch + 1) * len(train_loader),
                metrics={'val_acc': avg_val_acc, 'val_loss': val_loss},
                extra_state={
                    'val_accuracies_per_layer': val_accs,
                    'config': {
                        'input_dim': args.draft_input_dim,
                        'hidden_dim': args.draft_hidden_dim,
                        'num_coarse_layers': args.num_coarse_layers,
                        'chunk_len': args.chunk_len,
                        'action_hidden_dim': args.action_hidden_dim,
                        'grid_size': args.grid_size,
                    }
                }
            )

    # Save final best model
    draft_model_path = os.path.join(model_dir, "best_draft_with_projection.pt")
    best_checkpoint_path = ckpt_manager.get_best_checkpoint_path()
    if best_checkpoint_path:
        best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        torch.save(best_checkpoint, draft_model_path)
        print(f"\n✅ Best Draft Model saved to: {draft_model_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"🎉 Draft Model Training Complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.3f}")
    print(f"   Status: {'✅ TARGET MET (>90%)' if best_val_acc > 0.90 else '⚠️ Below target'}")
    print("=" * 80)

    logger.log_text(f"Training complete. Best accuracy: {best_val_acc:.3f}")
    logger.finish('completed')

    return best_val_acc


def train_conditional_model(args, all_data, device, model_dir, log_dir):
    """Train Conditional Main Model."""
    print("\n" + "=" * 80)
    print("🚀 Training CONDITIONAL Main Model (L0-L7 with Mode Locking)")
    print("=" * 80)

    # Initialize experiment logger
    logger = ExperimentLogger(
        name="phase2_conditional_rfsq_head_training",
        log_dir=log_dir,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "input_dim": args.cond_input_dim,
            "hidden_dim": args.cond_hidden_dim,
            "num_layers": args.num_layers,
            "condition_layers": args.num_coarse_layers,
            "token_embed_dim": args.token_embed_dim,
            "training_mode": "conditional",
            "device": str(device),
        }
    )

    # Create dataset
    dataset = ConditionedRFSQHeadDataset(all_data)
    val_size = min(5000, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create model
    print(f"\n📐 Creating Conditional RFSQ Head...")
    model = ConditionedRFSQHead(
        input_dim=args.cond_input_dim,
        hidden_dim=args.cond_hidden_dim,
        num_layers=args.num_layers,
        chunk_len=args.chunk_len,
        action_hidden_dim=args.action_hidden_dim,
        grid_size=args.grid_size,
        condition_layers=args.num_coarse_layers,
        token_embed_dim=args.token_embed_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params / 1e6:.1f}M parameters")
    print(f"   📊 Architecture: Conditional (Mode Locking)")
    print(f"   🔧 Condition layers: L0-L2 ({args.num_coarse_layers} layers)")
    print(f"   🔧 Token embedding dim: {args.token_embed_dim}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 10)

    # Checkpoint manager
    checkpoint_dir = os.path.join(model_dir, "openvla_rfsq_conditional_checkpoints")
    ckpt_manager = CheckpointManager(checkpoint_dir, keep_n_best=3, metric_name="val_acc", mode="max")

    # Training loop
    print(f"\n🔥 Training for {args.epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_accs = train_conditional_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_accs = validate_conditional(model, val_loader, device)

        scheduler.step()
        epoch_time = time.time() - start_time

        avg_val_acc = sum(val_accs) / 8

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc (avg): {avg_val_acc:.3f}")
        print(f"  Per-layer: L0={val_accs[0]:.3f} L1={val_accs[1]:.3f} L2={val_accs[2]:.3f} L3={val_accs[3]:.3f}")
        print(f"             L4={val_accs[4]:.3f} L5={val_accs[5]:.3f} L6={val_accs[6]:.3f} L7={val_accs[7]:.3f}")

        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc_avg': avg_val_acc,
            'learning_rate': scheduler.get_last_lr()[0],
        }
        for i in range(8):
            metrics[f'val_acc_L{i}'] = val_accs[i]
        logger.log(metrics, step=epoch)

        # Save checkpoint
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"  ✅ Best model saved: {avg_val_acc:.3f}")
            logger.log_text(f"New best conditional model at epoch {epoch + 1} with accuracy {avg_val_acc:.3f}")

        if (epoch + 1) % args.save_every == 0 or avg_val_acc == best_val_acc:
            ckpt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=(epoch + 1) * len(train_loader),
                metrics={'val_acc': avg_val_acc, 'val_loss': val_loss},
                extra_state={
                    'val_accuracies_per_layer': val_accs,
                    'config': {
                        'input_dim': args.cond_input_dim,
                        'hidden_dim': args.cond_hidden_dim,
                        'num_layers': args.num_layers,
                        'chunk_len': args.chunk_len,
                        'action_hidden_dim': args.action_hidden_dim,
                        'grid_size': args.grid_size,
                        'condition_layers': args.num_coarse_layers,
                        'token_embed_dim': args.token_embed_dim,
                        'training_mode': 'conditional',
                    }
                }
            )

    # Save final best model
    os.makedirs(os.path.join(model_dir, "openvla_rfsq_conditional"), exist_ok=True)
    cond_model_path = os.path.join(model_dir, "openvla_rfsq_conditional", "best_rfsq_head.pt")
    best_checkpoint_path = ckpt_manager.get_best_checkpoint_path()
    if best_checkpoint_path:
        best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        torch.save(best_checkpoint, cond_model_path)
        print(f"\n✅ Best Conditional Model saved to: {cond_model_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"🎉 Conditional RFSQ Head Training Complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.3f}")
    print(f"   Mode Locking: ✅ Enabled")
    print(f"   Status: {'✅ TARGET MET (>92%)' if best_val_acc > 0.92 else '⚠️ Below target'}")
    print("=" * 80)

    logger.log_text(f"Conditional training complete. Best accuracy: {best_val_acc:.3f}")
    logger.finish('completed')

    return best_val_acc


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main training function."""
    args = parse_args()

    # Get paths from environment variables or args
    model_dir = args.model_dir or os.getenv('RVQ_MODEL_DIR', './models')
    log_dir = args.log_dir or os.getenv('RVQ_LOG_DIR', './logs')

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load training data
    print(f"\n📦 Loading training data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Training data not found at {args.data_path}")

    all_data = torch.load(args.data_path, weights_only=False)
    print(f"✅ Loaded {len(all_data)} samples")

    results = {}

    # Train Draft Model
    if not args.train_conditional_only:
        draft_acc = train_draft_model(args, all_data, device, model_dir, log_dir)
        results['draft_accuracy'] = draft_acc
    else:
        print("\n⏭️  Skipping Draft Model training")

    # Train Conditional Model
    if not args.train_draft_only:
        cond_acc = train_conditional_model(args, all_data, device, model_dir, log_dir)
        results['conditional_accuracy'] = cond_acc
    else:
        print("\n⏭️  Skipping Conditional Model training")

    # Final Summary
    print("\n" + "=" * 80)
    print("🎉 Phase 2 CONDITIONAL Training Pipeline Complete!")
    print("=" * 80)

    if 'draft_accuracy' in results:
        status = "✅" if results['draft_accuracy'] > 0.90 else "⚠️"
        print(f"   {status} Draft Model Accuracy: {results['draft_accuracy']:.3f} (target: >90%)")

    if 'conditional_accuracy' in results:
        status = "✅" if results['conditional_accuracy'] > 0.92 else "⚠️"
        print(f"   {status} Conditional RFSQ Head Accuracy: {results['conditional_accuracy']:.3f} (target: >92%)")

    print("\n📁 Output files:")
    print(f"   - {os.path.join(model_dir, 'best_draft_with_projection.pt')}")
    print(f"   - {os.path.join(model_dir, 'openvla_rfsq_conditional', 'best_rfsq_head.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
