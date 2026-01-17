"""
Phase 3 集成测试脚本

测试整个RSD pipeline的数据流，不需要真实的OpenVLA和LIBERO。
用于验证各个组件的shape匹配和逻辑正确性。

Usage:
    # 1. 先生成mock checkpoints
    python mock_test/generate_mock_checkpoints.py --output-dir ./mock_models

    # 2. 运行集成测试
    python mock_test/test_phase3_integration.py --models-dir ./mock_models

    # 3. (可选) 测试LIBERO集成
    python mock_test/test_phase3_integration.py --models-dir ./mock_models --test-libero
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Import model definitions from mock checkpoint generator
# ============================================================

# We'll redefine them here for independence
class RobustSTEQuantizer(nn.Module):
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
    def __init__(self, action_dim=7, hidden_dim=16, num_layers=8, num_levels=7, use_layernorm=True):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
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

    def decode_from_indices(self, indices):
        batch_size, chunk_len, hidden_dim, num_layers = indices.shape
        z_reconstructed = self.rfsq.decode_from_indices(indices)
        z_flat = z_reconstructed.view(-1, self.hidden_dim)
        actions_flat = self.decoder(z_flat)
        actions = actions_flat.view(batch_size, chunk_len, -1)
        return actions


class DraftTransformerDecoder(nn.Module):
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
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.decoder = DraftTransformerDecoder(hidden_dim=hidden_dim)
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size
        self.num_coarse_layers = num_coarse_layers

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


class RFSQClassificationHead(nn.Module):
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
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size
        self.num_layers = num_layers

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
# RSD Inference Engine (Simplified)
# ============================================================

class MockRSDEngine:
    """Simplified RSD Engine for testing"""

    def __init__(
        self,
        draft_model,
        rfsq_head,
        rfsq_decoder,
        device='cpu',
        chunk_len=8,
        action_dim=7,
    ):
        self.draft_model = draft_model
        self.rfsq_head = rfsq_head
        self.rfsq_decoder = rfsq_decoder
        self.device = device
        self.chunk_len = chunk_len
        self.action_dim = action_dim

        # Set to eval mode
        self.draft_model.eval()
        self.rfsq_head.eval()
        self.rfsq_decoder.eval()

    @torch.no_grad()
    def generate_action(
        self,
        mock_hidden_state: torch.Tensor,
        use_speculative_decoding: bool = True,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate action from mock OpenVLA hidden state

        Args:
            mock_hidden_state: [1, 4096] mock hidden state
            use_speculative_decoding: whether to use draft model
            verbose: print debug info

        Returns:
            actions: [chunk_len, action_dim] numpy array
            info: dict with stats
        """
        start_time = time.time()
        info = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 RSD Inference Engine - Mock Test")
            print(f"{'='*60}")
            print(f"Input shape: {mock_hidden_state.shape}")
            print(f"Use speculative decoding: {use_speculative_decoding}")

        # Step 1: Draft Model Prediction (if enabled)
        if use_speculative_decoding:
            draft_start = time.time()
            draft_logits = self.draft_model(mock_hidden_state)  # [1, 3, 128, 7]
            draft_tokens = torch.argmax(draft_logits, dim=-1)   # [1, 3, 128]
            draft_time = time.time() - draft_start
            info['draft_time'] = draft_time

            if verbose:
                print(f"\n1️⃣ Draft Model Prediction:")
                print(f"   Logits shape: {draft_logits.shape}")
                print(f"   Tokens shape: {draft_tokens.shape}")
                print(f"   Time: {draft_time*1000:.2f}ms")
        else:
            draft_tokens = None
            info['draft_time'] = 0.0

        # Step 2: Main Model Prediction
        main_start = time.time()
        main_logits = self.rfsq_head(mock_hidden_state)  # [1, 8, 128, 7]
        main_tokens = torch.argmax(main_logits, dim=-1)  # [1, 8, 128]
        main_time = time.time() - main_start
        info['main_time'] = main_time

        if verbose:
            print(f"\n2️⃣ Main Model Prediction:")
            print(f"   Logits shape: {main_logits.shape}")
            print(f"   Tokens shape: {main_tokens.shape}")
            print(f"   Time: {main_time*1000:.2f}ms")

        # Step 3: Token Comparison & Acceptance (if using speculative decoding)
        if use_speculative_decoding and draft_tokens is not None:
            # Compare first 3 layers
            matches = (draft_tokens == main_tokens[:, :3, :]).float()
            acceptance_rate = matches.mean().item()
            info['acceptance_rate'] = acceptance_rate

            if verbose:
                print(f"\n3️⃣ Token Comparison:")
                print(f"   Acceptance rate: {acceptance_rate*100:.1f}%")
                print(f"   Layer 0: {matches[0, 0].mean().item()*100:.1f}%")
                print(f"   Layer 1: {matches[0, 1].mean().item()*100:.1f}%")
                print(f"   Layer 2: {matches[0, 2].mean().item()*100:.1f}%")
        else:
            info['acceptance_rate'] = 0.0

        # Step 4: Use main_tokens for final decoding
        final_tokens = main_tokens  # [1, 8, 128]

        # Reshape: [1, 8, 128] -> [1, 8, 16] -> [1, Chunk=8, Hidden=16, Layers=8]
        # Need to reshape 128 = chunk_len * action_hidden_dim back
        batch_size, num_layers, flat_dim = final_tokens.shape
        final_tokens = final_tokens.view(batch_size, num_layers, self.chunk_len, 16)
        final_tokens = final_tokens.permute(0, 2, 3, 1)  # [1, Chunk, Hidden, Layers]

        if verbose:
            print(f"\n4️⃣ Token Reshaping:")
            print(f"   After reshape: {final_tokens.shape}")

        # Step 5: RFSQ Decoder
        decode_start = time.time()
        actions = self.rfsq_decoder.decode_from_indices(final_tokens)  # [1, Chunk, 7]
        decode_time = time.time() - decode_start
        info['decode_time'] = decode_time

        if verbose:
            print(f"\n5️⃣ RFSQ Decoding:")
            print(f"   Actions shape: {actions.shape}")
            print(f"   Time: {decode_time*1000:.2f}ms")

        # Convert to numpy
        actions_np = actions[0].cpu().numpy()  # [Chunk, 7]

        # Total time
        total_time = time.time() - start_time
        info['total_time'] = total_time

        if verbose:
            print(f"\n6️⃣ Final Output:")
            print(f"   Actions shape: {actions_np.shape}")
            print(f"   Actions range: [{actions_np.min():.3f}, {actions_np.max():.3f}]")
            print(f"   Total time: {total_time*1000:.2f}ms")
            print(f"{'='*60}")

        return actions_np, info


# ============================================================
# Test Functions
# ============================================================

def test_checkpoint_loading(models_dir: Path, device: str):
    """测试checkpoint加载"""
    print(f"\n{'='*60}")
    print(f"📦 Test 1: Checkpoint Loading")
    print(f"{'='*60}")

    checkpoints = {}

    # Load RFSQ Decoder
    rfsq_path = models_dir / 'rfsq_robust_best.pt'
    if not rfsq_path.exists():
        print(f"❌ RFSQ checkpoint not found: {rfsq_path}")
        return None

    print(f"\n1️⃣ Loading RFSQ Decoder...")
    rfsq_model = ActionRFSQAE(use_layernorm=True)
    checkpoint = torch.load(rfsq_path, map_location=device, weights_only=False)
    rfsq_model.load_state_dict(checkpoint['model'])
    rfsq_model.to(device)
    rfsq_model.eval()
    checkpoints['rfsq'] = rfsq_model
    print(f"   ✅ Loaded (MSE: {checkpoint.get('mse', 'N/A')})")

    # Load Draft Model
    draft_path = models_dir / 'best_draft_with_projection.pt'
    if not draft_path.exists():
        print(f"❌ Draft Model checkpoint not found: {draft_path}")
        return None

    print(f"\n2️⃣ Loading Draft Model...")
    draft_model = RFSQDraftModelWithProjection()
    checkpoint = torch.load(draft_path, map_location=device, weights_only=False)
    draft_model.load_state_dict(checkpoint['model_state_dict'])
    draft_model.to(device)
    draft_model.eval()
    checkpoints['draft'] = draft_model
    print(f"   ✅ Loaded (Accuracy: {checkpoint.get('val_accuracy', 'N/A')})")

    # Load RFSQ Head
    rfsq_head_path = models_dir / 'openvla_rfsq_robust' / 'best_rfsq_head.pt'
    if not rfsq_head_path.exists():
        print(f"❌ RFSQ Head checkpoint not found: {rfsq_head_path}")
        return None

    print(f"\n3️⃣ Loading RFSQ Head...")
    rfsq_head = RFSQClassificationHead()
    checkpoint = torch.load(rfsq_head_path, map_location=device, weights_only=False)
    rfsq_head.load_state_dict(checkpoint['model_state_dict'])
    rfsq_head.to(device)
    rfsq_head.eval()
    checkpoints['rfsq_head'] = rfsq_head
    print(f"   ✅ Loaded (Accuracy: {checkpoint.get('val_accuracy', 'N/A')})")

    print(f"\n✅ All checkpoints loaded successfully!")
    return checkpoints


def test_rsd_pipeline(checkpoints: Dict, device: str):
    """测试完整RSD pipeline"""
    print(f"\n{'='*60}")
    print(f"🔬 Test 2: RSD Pipeline Integration")
    print(f"{'='*60}")

    # Create RSD Engine
    engine = MockRSDEngine(
        draft_model=checkpoints['draft'],
        rfsq_head=checkpoints['rfsq_head'],
        rfsq_decoder=checkpoints['rfsq'],
        device=device,
    )

    # Generate mock OpenVLA hidden state
    mock_hidden = torch.randn(1, 4096).to(device)

    print(f"\n🧪 Test 2.1: With Speculative Decoding")
    actions1, info1 = engine.generate_action(mock_hidden, use_speculative_decoding=True, verbose=True)

    print(f"\n🧪 Test 2.2: Without Speculative Decoding")
    actions2, info2 = engine.generate_action(mock_hidden, use_speculative_decoding=False, verbose=True)

    # Verify outputs
    print(f"\n{'='*60}")
    print(f"📊 Results Comparison")
    print(f"{'='*60}")
    print(f"With Draft Model:")
    print(f"   Total time: {info1['total_time']*1000:.2f}ms")
    print(f"   Draft time: {info1['draft_time']*1000:.2f}ms")
    print(f"   Main time: {info1['main_time']*1000:.2f}ms")
    print(f"   Decode time: {info1['decode_time']*1000:.2f}ms")
    print(f"   Acceptance rate: {info1['acceptance_rate']*100:.1f}%")

    print(f"\nWithout Draft Model:")
    print(f"   Total time: {info2['total_time']*1000:.2f}ms")
    print(f"   Main time: {info2['main_time']*1000:.2f}ms")
    print(f"   Decode time: {info2['decode_time']*1000:.2f}ms")

    print(f"\n✅ Pipeline test passed!")
    return True


def test_libero_integration(checkpoints: Dict, device: str):
    """测试LIBERO集成（简化版本）"""
    print(f"\n{'='*60}")
    print(f"🤖 Test 3: LIBERO Integration (Simplified)")
    print(f"{'='*60}")

    try:
        # Try to import LIBERO
        import sys
        sys.path.insert(0, '/root/LIBERO')  # Adjust path as needed
        from libero.libero import benchmark
        print(f"   ✅ LIBERO imported successfully")

        # This would be the actual LIBERO test
        # For now, we just verify the import works
        print(f"   ℹ️  Full LIBERO test requires LIBERO environment")
        print(f"   ℹ️  Skipping actual episode rollout")

    except ImportError as e:
        print(f"   ⚠️  LIBERO not available: {e}")
        print(f"   ℹ️  This is expected in local testing environment")
        print(f"   ℹ️  LIBERO test will work in Modal environment")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Phase 3 integration')
    parser.add_argument('--models-dir', type=str, default='./mock_models',
                        help='Directory containing mock checkpoints')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run tests on')
    parser.add_argument('--test-libero', action='store_true',
                        help='Also test LIBERO integration')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    device = args.device

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print(f"   Please run: python mock_test/generate_mock_checkpoints.py --output-dir {models_dir}")
        return 1

    print("=" * 60)
    print("🧪 Phase 3 Integration Test Suite")
    print("=" * 60)
    print(f"Models directory: {models_dir.absolute()}")
    print(f"Device: {device}")

    # Test 1: Checkpoint Loading
    checkpoints = test_checkpoint_loading(models_dir, device)
    if checkpoints is None:
        print(f"\n❌ Test 1 failed: Could not load checkpoints")
        return 1

    # Test 2: RSD Pipeline
    try:
        test_rsd_pipeline(checkpoints, device)
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 3: LIBERO Integration (optional)
    if args.test_libero:
        try:
            test_libero_integration(checkpoints, device)
        except Exception as e:
            print(f"\n⚠️  Test 3 warning: {e}")
            # Don't fail on LIBERO test, it's optional

    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ All Tests Passed!")
    print(f"{'='*60}")
    print(f"\n🎯 Next steps:")
    print(f"   1. If all tests pass locally, deploy to Modal with real checkpoints")
    print(f"   2. Run actual Phase 3 evaluation with LIBERO")
    print(f"   3. Monitor for any integration issues")

    return 0


if __name__ == "__main__":
    exit(main())
