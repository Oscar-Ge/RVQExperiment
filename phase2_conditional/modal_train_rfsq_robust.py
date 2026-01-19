"""
Phase 1 Improved: Train Robust RFSQ with LayerNorm Strategy

This script trains an improved RFSQ AutoEncoder using the LayerNorm strategy
from the latest RFSQ paper. The key improvement is normalizing residual signals
before quantization to make all 8 layers effective (vs only 3 effective layers
in the naive approach).

Expected Results:
- Naive RFSQ MSE: ~0.018
- Robust RFSQ MSE: ~0.010 (-44% improvement)
- Target MSE: < 0.012

Output:
- /models/rfsq_robust_best.pt - Best model checkpoint

Usage:
    modal run modal_train_rfsq_robust.py
"""

import os
import sys
import modal

# ============================================================
# Modal App Setup
# ============================================================
app = modal.App("phase1-robust-rfsq-training")

# Create/get volumes
data_volume = modal.Volume.from_name("rsd-libero-data", create_if_missing=True)
models_volume = modal.Volume.from_name("rsd-models", create_if_missing=True)

# Get SDK path for Orchestra experiment tracking
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

# Build the training image
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch torchvision numpy tqdm matplotlib pandas requests"
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)


# ============================================================
# Robust RFSQ Model Definition (with LayerNorm)
# ============================================================
def define_models():
    """Define the Robust RFSQ model classes inside the Modal function."""
    import torch
    import torch.nn as nn
    
    class RobustSTEQuantizer(nn.Module):
        """
        Improved STE Quantizer with LayerNorm strategy.
        
        Key improvement: Normalizes residual signals before quantization,
        then denormalizes after, making all layers effective.
        """
        def __init__(self, num_levels=7, use_layernorm=True):
            super().__init__()
            self.num_levels = num_levels
            self.use_layernorm = use_layernorm
            self.register_buffer('boundaries', torch.linspace(-1, 1, num_levels))

        def forward(self, z):
            if self.use_layernorm:
                # Save original scale
                original_mean = z.mean(dim=-1, keepdim=True)
                original_std = z.std(dim=-1, keepdim=True) + 1e-5
                
                # Normalize
                z_norm = (z - original_mean) / original_std
                
                # Quantize in normalized space
                dist = torch.abs(z_norm.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
                indices = torch.argmin(dist, dim=-1)
                z_q_norm = self.boundaries[indices]
                
                # Denormalize
                z_q = z_q_norm * original_std + original_mean
            else:
                # Original logic (no LayerNorm)
                dist = torch.abs(z.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
                indices = torch.argmin(dist, dim=-1)
                z_q = self.boundaries[indices]

            # Straight-Through Estimator
            z_q_out = z + (z_q - z).detach()
            return z_q_out, indices

    class RobustRFSQBlock(nn.Module):
        """Multi-layer residual quantization with LayerNorm."""
        def __init__(self, num_layers=8, num_levels=7, use_layernorm=True):
            super().__init__()
            self.num_layers = num_layers
            self.num_levels = num_levels
            self.use_layernorm = use_layernorm
            self.layers = nn.ModuleList([
                RobustSTEQuantizer(num_levels=num_levels, use_layernorm=use_layernorm)
                for _ in range(num_layers)
            ])

        def forward(self, z):
            residual = z
            quantized_sum = 0
            all_indices = []

            for layer in self.layers:
                z_q, indices = layer(residual)
                quantized_sum = quantized_sum + z_q
                residual = residual - z_q
                all_indices.append(indices)

            codes = torch.stack(all_indices, dim=-1)
            return quantized_sum, codes

        def decode_from_indices(self, indices):
            batch_size, seq_len, dim, num_layers = indices.shape
            reconstruction = torch.zeros(batch_size, seq_len, dim, device=indices.device)
            for layer_idx in range(num_layers):
                layer_indices = indices[:, :, :, layer_idx]
                layer_values = self.layers[layer_idx].boundaries[layer_indices]
                reconstruction = reconstruction + layer_values
            return reconstruction

    class ActionRFSQAE(nn.Module):
        """Improved Action RFSQ AutoEncoder with LayerNorm strategy."""
        def __init__(self, action_dim=7, hidden_dim=16, num_layers=8, num_levels=7, use_layernorm=True):
            super().__init__()
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_levels = num_levels
            self.use_layernorm = use_layernorm

            self.encoder = nn.Sequential(
                nn.Linear(action_dim, 64),
                nn.Mish(),
                nn.Linear(64, hidden_dim),
                nn.Tanh()
            )

            self.rfsq = RobustRFSQBlock(
                num_layers=num_layers,
                num_levels=num_levels,
                use_layernorm=use_layernorm,
            )

            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.Mish(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            z_quantized, codes = self.rfsq(z)
            x_recon = self.decoder(z_quantized)
            return x_recon, codes

        def encode(self, x):
            z = self.encoder(x)
            _, codes = self.rfsq(z)
            return codes

        def decode_from_indices(self, indices):
            batch_size, chunk_len, hidden_dim, num_layers = indices.shape
            z_reconstructed = self.rfsq.decode_from_indices(indices)
            z_flat = z_reconstructed.view(-1, self.hidden_dim)
            actions_flat = self.decoder(z_flat)
            actions = actions_flat.view(batch_size, chunk_len, -1)
            return actions

    return ActionRFSQAE


# ============================================================
# Training Function
# ============================================================
@app.function(
    image=training_image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={
        "/data": data_volume,
        "/models": models_volume,
    },
    secrets=[
        modal.Secret.from_name("orchestra-supabase"),
    ],
)
def train_robust_rfsq(
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim: int = 16,
    num_layers: int = 8,
    num_levels: int = 7,
    use_layernorm: bool = True,
    entropy_weight: float = 0.05,
    save_every: int = 10,
):
    """
    Train Robust RFSQ AutoEncoder with LayerNorm strategy.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        hidden_dim: Latent dimension
        num_layers: Number of RFSQ layers
        num_levels: Number of quantization levels per layer
        use_layernorm: Whether to use LayerNorm strategy (True for Robust)
        entropy_weight: Weight for entropy regularization loss
        save_every: Save checkpoint every N epochs
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    import time
    from datetime import datetime
    
    # Setup experiment tracking
    sys.path.insert(0, "/root")
    from src.orchestra_sdk.experiment import Experiment
    
    exp = Experiment.init(
        name="Phase 1: Robust RFSQ Training with LayerNorm",
        description="Training improved RFSQ AutoEncoder with LayerNorm strategy. "
                    "Target: MSE < 0.012 (vs Naive RFSQ MSE ~0.018). "
                    "Expected improvement: ~44% reduction in reconstruction error.",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_levels": num_levels,
            "use_layernorm": use_layernorm,
            "entropy_weight": entropy_weight,
            "gpu_type": "a100",
            "gpu_count": 1,
            "model_type": "Robust RFSQ" if use_layernorm else "Naive RFSQ",
            "target_mse": 0.012,
        }
    )
    
    exp.add_tags(['phase1', 'rfsq', 'robust', 'layernorm', 'a100'])
    exp.set_metadata({
        'platform': 'Modal',
        'gpu_spec': '1x A100',
        'improvement': 'LayerNorm strategy for all 8 layers effective',
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"📊 Training Robust RFSQ with LayerNorm={use_layernorm}")
    
    # Load training data
    print("\n📦 Loading training data...")
    data_path = "/data/libero_actions_normalized.pt"
    
    if not os.path.exists(data_path):
        exp.log_text(f"Data file not found: {data_path}", level='error')
        exp.finish('failed')
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    data = torch.load(data_path)
    
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
    exp.log_text(f"Loaded {len(actions)} samples, action_dim={action_dim}")
    
    # Create dataloader
    dataset = TensorDataset(actions.float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model
    ActionRFSQAE = define_models()
    model = ActionRFSQAE(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_levels=num_levels,
        use_layernorm=use_layernorm,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Model parameters: {total_params:,}")
    print(f"   - Action dim: {action_dim}")
    print(f"   - Hidden dim: {hidden_dim}")
    print(f"   - Num layers: {num_layers}")
    print(f"   - Num levels: {num_levels}")
    print(f"   - Use LayerNorm: {use_layernorm}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print(f"\n🔥 Starting training for {num_epochs} epochs...")
    best_mse = float('inf')
    best_epoch = 0
    
    model.train()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        total_mse = 0
        total_entropy = 0
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
        
        # Epoch statistics
        avg_mse = total_mse / num_batches
        avg_entropy = total_entropy / num_batches
        epoch_time = time.time() - start_time
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        exp.log({
            'mse': avg_mse,
            'entropy_loss': avg_entropy,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
        }, step=epoch)
        
        # Update progress
        progress = int((epoch + 1) / num_epochs * 100)
        exp.set_progress(progress)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"MSE: {avg_mse:.6f} | "
              f"Entropy: {avg_entropy:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_epoch = epoch + 1
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'train_mse': avg_mse,
                'config': {
                    'action_dim': action_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_levels': num_levels,
                    'use_layernorm': use_layernorm,
                },
            }
            
            save_path = "/models/rfsq_robust_best.pt"
            torch.save(checkpoint, save_path)
            models_volume.commit()
            
            print(f"   ✅ New best model saved! MSE: {avg_mse:.6f}")
            exp.log_text(f"New best model at epoch {epoch+1}, MSE={avg_mse:.6f}")
        
        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'train_mse': avg_mse,
                'config': {
                    'action_dim': action_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_levels': num_levels,
                    'use_layernorm': use_layernorm,
                },
            }
            save_path = f"/models/rfsq_robust_ep{epoch+1}.pt"
            torch.save(checkpoint, save_path)
            models_volume.commit()
            print(f"   📁 Checkpoint saved: {save_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"   Best MSE: {best_mse:.6f} (at epoch {best_epoch})")
    print(f"   Target MSE: < 0.012")
    print(f"   Status: {'✅ TARGET MET!' if best_mse < 0.012 else '⚠️ Target not met'}")
    print(f"   Checkpoint: /models/rfsq_robust_best.pt")
    print("=" * 60)
    
    # Log final results
    exp.log({
        'final_best_mse': best_mse,
        'best_epoch': best_epoch,
        'target_met': best_mse < 0.012,
    }, step=num_epochs)
    
    exp.log_text(f"Training complete! Best MSE: {best_mse:.6f} at epoch {best_epoch}")
    
    if best_mse < 0.012:
        exp.finish('completed')
    else:
        exp.log_text(f"Warning: Target MSE < 0.012 not met. Best: {best_mse:.6f}", level='warning')
        exp.finish('completed')  # Still mark as completed, just didn't hit target
    
    return {
        'best_mse': best_mse,
        'best_epoch': best_epoch,
        'target_met': best_mse < 0.012,
        'checkpoint_path': '/models/rfsq_robust_best.pt',
    }


# ============================================================
# Local Entrypoint
# ============================================================
@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    debug: bool = False,
):
    """
    Train Robust RFSQ with LayerNorm strategy.
    
    Args:
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 64)
        lr: Learning rate (default: 1e-3)
        debug: If True, run quick debug mode (10 epochs)
    """
    if debug:
        print("🔧 Debug mode: Running 10 epochs only")
        epochs = 10
    
    print(f"🚀 Launching Robust RFSQ training on Modal A100")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Use LayerNorm: True (Robust RFSQ)")
    
    result = train_robust_rfsq.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        use_layernorm=True,
    )
    
    print("\n📊 Training Results:")
    print(f"   Best MSE: {result['best_mse']:.6f}")
    print(f"   Best Epoch: {result['best_epoch']}")
    print(f"   Target Met: {result['target_met']}")
    print(f"   Checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
