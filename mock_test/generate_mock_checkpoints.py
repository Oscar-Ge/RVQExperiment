"""
生成Mock Checkpoints用于测试

这个脚本生成所有阶段的假checkpoint，模拟训练完成的状态。
用于在便宜GPU上测试整个pipeline的集成，避免在A100上发现问题。

Usage:
    python mock_test/generate_mock_checkpoints.py --output-dir ./mock_models
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# Phase 1: Robust RFSQ AutoEncoder
# ============================================================

class RobustSTEQuantizer(nn.Module):
    """Mock Robust STE Quantizer"""
    def __init__(self, num_levels=7, use_layernorm=True):
        super().__init__()
        self.num_levels = num_levels
        self.use_layernorm = use_layernorm
        self.register_buffer('boundaries', torch.linspace(-1, 1, num_levels))

    def forward(self, z):
        if self.use_layernorm:
            original_mean = z.mean(dim=-1, keepdim=True)
            original_std = z.std(dim=-1, keepdim=True) + 1e-5
            z_norm = (z - original_mean) / original_std
            dist = torch.abs(z_norm.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)
            z_q_norm = self.boundaries[indices]
            z_q = z_q_norm * original_std + original_mean
        else:
            dist = torch.abs(z.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)
            z_q = self.boundaries[indices]

        z_q_out = z + (z_q - z).detach()
        return z_q_out, indices


class RobustRFSQBlock(nn.Module):
    """Mock Robust RFSQ Block"""
    def __init__(self, num_layers=8, num_levels=7, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
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
        batch_size, chunk_len, hidden_dim, num_layers = indices.shape
        reconstruction = torch.zeros(batch_size, chunk_len, hidden_dim, device=indices.device)
        for layer_idx in range(num_layers):
            layer_indices = indices[:, :, :, layer_idx]
            layer_values = self.layers[layer_idx].boundaries[layer_indices]
            reconstruction = reconstruction + layer_values
        return reconstruction


class ActionRFSQAE(nn.Module):
    """Mock Action RFSQ AutoEncoder"""
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
        self.rfsq = RobustRFSQBlock(num_layers=num_layers, num_levels=num_levels, use_layernorm=use_layernorm)
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


# ============================================================
# Phase 2: Draft Model with Projection
# ============================================================

class DraftTransformerDecoder(nn.Module):
    """Mock Draft Transformer Decoder"""
    def __init__(self, hidden_dim=512, num_heads=8, feedforward_dim=2048, max_seq_length=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.position_encoding = nn.Parameter(torch.randn(1, max_seq_length, hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        pos_enc = self.position_encoding[:, :seq_len, :]
        hidden_states = hidden_states + pos_enc
        output = self.decoder_layer(hidden_states, hidden_states)
        return self.output_norm(output)


class RFSQDraftModelWithProjection(nn.Module):
    """Mock Draft Model with Projection"""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=512,
        num_coarse_layers=3,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_coarse_layers = num_coarse_layers
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.decoder = DraftTransformerDecoder(hidden_dim=hidden_dim)

        output_size_per_head = chunk_len * action_hidden_dim * grid_size
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_size_per_head),
            )
            for _ in range(num_coarse_layers)
        ])

    def forward(self, openvla_hidden_states):
        batch_size = openvla_hidden_states.shape[0]
        projected = self.input_projection(openvla_hidden_states)
        x = projected.unsqueeze(1)
        decoder_output = self.decoder(x)
        decoder_output = decoder_output.squeeze(1)

        layer_outputs = []
        for head in self.classification_heads:
            logits = head(decoder_output)
            logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
            layer_outputs.append(logits)

        return torch.stack(layer_outputs, dim=1)


# ============================================================
# Phase 2: RFSQ Classification Head (for Main Model)
# ============================================================

class RFSQClassificationHead(nn.Module):
    """Mock RFSQ Classification Head"""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=2048,
        num_layers=8,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size

        output_size_per_head = chunk_len * action_hidden_dim * grid_size

        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_size_per_head),
            )
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        layer_outputs = []
        for head in self.classification_heads:
            logits = head(hidden_states)
            logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
            layer_outputs.append(logits)
        return torch.stack(layer_outputs, dim=1)


# ============================================================
# Mock Checkpoint Generator
# ============================================================

def generate_phase1_checkpoint(output_dir: Path, use_layernorm: bool = True):
    """生成Phase 1 RFSQ checkpoint"""
    print(f"\n{'='*60}")
    print(f"📦 Generating Phase 1 {'Robust' if use_layernorm else 'Naive'} RFSQ Checkpoint")
    print(f"{'='*60}")

    model = ActionRFSQAE(
        action_dim=7,
        hidden_dim=16,
        num_layers=8,
        num_levels=7,
        use_layernorm=use_layernorm,
    )

    # Initialize with reasonable values
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.ones_(param)  # For 1D weights like LayerNorm
        elif 'bias' in name:
            nn.init.zeros_(param)

    checkpoint = {
        'model': model.state_dict(),
        'model_state_dict': model.state_dict(),  # Compatibility
        'epoch': 100,
        'mse': 0.010 if use_layernorm else 0.018,
        'config': {
            'action_dim': 7,
            'hidden_dim': 16,
            'num_layers': 8,
            'num_levels': 7,
            'use_layernorm': use_layernorm,
        },
    }

    filename = 'rfsq_robust_best.pt' if use_layernorm else 'rfsq_best.pt'
    save_path = output_dir / filename
    torch.save(checkpoint, save_path)

    print(f"   ✅ Saved to: {save_path}")
    print(f"   📊 Mock MSE: {checkpoint['mse']:.6f}")
    print(f"   🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return save_path


def generate_phase2_draft_checkpoint(output_dir: Path):
    """生成Phase 2 Draft Model checkpoint"""
    print(f"\n{'='*60}")
    print(f"📦 Generating Phase 2 Draft Model Checkpoint")
    print(f"{'='*60}")

    model = RFSQDraftModelWithProjection(
        input_dim=4096,
        hidden_dim=512,
        num_coarse_layers=3,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
    )

    # Initialize
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.ones_(param)  # For 1D weights like LayerNorm
        elif 'bias' in name:
            nn.init.zeros_(param)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Not needed for inference
        'epoch': 50,
        'val_accuracy': 0.915,  # Mock 91.5% accuracy
        'val_accuracies_per_layer': [0.920, 0.915, 0.910],
        'config': {
            'input_dim': 4096,
            'hidden_dim': 512,
            'num_coarse_layers': 3,
            'chunk_len': 8,
            'action_hidden_dim': 16,
            'grid_size': 7,
        },
    }

    save_path = output_dir / 'best_draft_with_projection.pt'
    torch.save(checkpoint, save_path)

    print(f"   ✅ Saved to: {save_path}")
    print(f"   📊 Mock Accuracy: {checkpoint['val_accuracy']:.3f}")
    print(f"   📊 Per-layer: L0={checkpoint['val_accuracies_per_layer'][0]:.3f}, "
          f"L1={checkpoint['val_accuracies_per_layer'][1]:.3f}, "
          f"L2={checkpoint['val_accuracies_per_layer'][2]:.3f}")
    print(f"   🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return save_path


def generate_phase2_main_checkpoint(output_dir: Path):
    """生成Phase 2 Main Model (RFSQ Head) checkpoint"""
    print(f"\n{'='*60}")
    print(f"📦 Generating Phase 2 Main Model (RFSQ Head) Checkpoint")
    print(f"{'='*60}")

    model = RFSQClassificationHead(
        input_dim=4096,
        hidden_dim=2048,
        num_layers=8,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
    )

    # Initialize
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.ones_(param)  # For 1D weights like LayerNorm
        elif 'bias' in name:
            nn.init.zeros_(param)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 10,
        'val_accuracy': 0.925,  # Mock 92.5% accuracy
        'config': {
            'input_dim': 4096,
            'hidden_dim': 2048,
            'num_layers': 8,
            'chunk_len': 8,
            'action_hidden_dim': 16,
            'grid_size': 7,
        },
    }

    # Create subdirectory
    rfsq_head_dir = output_dir / 'openvla_rfsq_robust'
    rfsq_head_dir.mkdir(parents=True, exist_ok=True)

    save_path = rfsq_head_dir / 'best_rfsq_head.pt'
    torch.save(checkpoint, save_path)

    print(f"   ✅ Saved to: {save_path}")
    print(f"   📊 Mock Accuracy: {checkpoint['val_accuracy']:.3f}")
    print(f"   🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description='Generate mock checkpoints for testing')
    parser.add_argument('--output-dir', type=str, default='./mock_models',
                        help='Directory to save mock checkpoints')
    parser.add_argument('--generate-naive', action='store_true',
                        help='Also generate naive RFSQ checkpoint for comparison')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 Mock Checkpoint Generator")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")

    # Generate all checkpoints
    checkpoints = {}

    # Phase 1: Robust RFSQ
    checkpoints['rfsq_robust'] = generate_phase1_checkpoint(output_dir, use_layernorm=True)

    # Phase 1: Naive RFSQ (optional)
    if args.generate_naive:
        checkpoints['rfsq_naive'] = generate_phase1_checkpoint(output_dir, use_layernorm=False)

    # Phase 2: Draft Model
    checkpoints['draft_model'] = generate_phase2_draft_checkpoint(output_dir)

    # Phase 2: Main Model (RFSQ Head)
    checkpoints['rfsq_head'] = generate_phase2_main_checkpoint(output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("✅ All Mock Checkpoints Generated!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for name, path in checkpoints.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   • {name:20s}: {path.name:40s} ({size_mb:.2f} MB)")

    print(f"\n📁 Total size: {sum(p.stat().st_size for p in checkpoints.values()) / (1024 * 1024):.2f} MB")
    print(f"\n🎯 Next step: Run integration test")
    print(f"   python mock_test/test_phase3_integration.py --models-dir {output_dir}")


if __name__ == "__main__":
    main()
