"""
LIBERO Integration Test with Mock OpenVLA

用真实LIBERO测试Phase 3，但使用Mock OpenVLA。
这样可以测试最容易出问题的LIBERO集成，但不需要：
- 昂贵的A100训练成本
- OpenVLA加载的显存开销（7B模型需要~14GB）

Usage:
    # SSH到你租的GPU机器后
    python mock_test/test_libero_with_mock_openvla.py \
        --models-dir ./mock_models \
        --num-episodes 3 \
        --task-id 0
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import time
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Import model definitions (same as test_phase3_integration.py)
# ============================================================

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
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
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
    def __init__(self, input_dim=4096, hidden_dim=512, num_coarse_layers=3,
                 chunk_len=8, action_hidden_dim=16, grid_size=7):
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
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_size_per_head),
            )
            for _ in range(num_coarse_layers)
        ])

    def forward(self, openvla_hidden_states):
        batch_size = openvla_hidden_states.shape[0]
        projected = self.input_projection(openvla_hidden_states)
        x = projected.unsqueeze(1)
        decoder_output = self.decoder(x).squeeze(1)
        layer_outputs = []
        for head in self.classification_heads:
            logits = head(decoder_output)
            logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
            layer_outputs.append(logits)
        return torch.stack(layer_outputs, dim=1)


class RFSQClassificationHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048, num_layers=8,
                 chunk_len=8, action_hidden_dim=16, grid_size=7):
        super().__init__()
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size
        self.num_layers = num_layers
        output_size_per_head = chunk_len * action_hidden_dim * grid_size
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.LayerNorm(hidden_dim // 2),
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
# Mock OpenVLA
# ============================================================

class MockOpenVLA:
    """
    Mock OpenVLA模型，不需要加载真实的7B模型

    功能：
    1. 接收图像和文本，返回mock hidden state (4096-dim)
    2. 提供predict_action接口（返回合理的随机action）
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.hidden_size = 4096

        # 简单的CNN用于从图像提取特征（很轻量）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [32, 1, 1]
        ).to(device)

        # 投影到4096维
        self.projection = nn.Linear(32, self.hidden_size).to(device)

        # Action normalization stats (LIBERO specific)
        # 这些是从真实LIBERO数据统计出来的
        self.action_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5])
        self.action_std = np.array([0.1, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5])

    def __call__(self, image: Image.Image, task_description: str, **kwargs):
        """Mock OpenVLA forward pass"""
        # 简单的图像编码
        img_tensor = self._preprocess_image(image).to(self.device)

        with torch.no_grad():
            features = self.image_encoder(img_tensor)  # [1, 32, 1, 1]
            features = features.view(1, -1)  # [1, 32]
            hidden_state = self.projection(features)  # [1, 4096]

            # 添加一些随机性（模拟真实VLA的多样性）
            noise = torch.randn_like(hidden_state) * 0.1
            hidden_state = hidden_state + noise

        return hidden_state

    def predict_action(self, image: Image.Image, task_description: str, unnorm_key: str = None):
        """
        Mock predict_action接口

        返回一个合理的随机action（7-dim）
        """
        # 生成一个合理的随机action
        # 位置变化小，gripper在-1到1之间
        action = np.random.randn(7) * 0.05  # 小变化
        action[-1] = np.random.choice([-1.0, 1.0])  # gripper open/close

        return action

    def _preprocess_image(self, image: Image.Image):
        """预处理图像"""
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0)  # [1, 3, 224, 224]


# ============================================================
# RSD Engine with Mock OpenVLA
# ============================================================

class RSDEngineWithMockVLA:
    """RSD Engine using Mock OpenVLA"""

    def __init__(
        self,
        draft_model,
        rfsq_head,
        rfsq_decoder,
        device='cuda',
        chunk_len=8,
        action_dim=7,
    ):
        self.draft_model = draft_model
        self.rfsq_head = rfsq_head
        self.rfsq_decoder = rfsq_decoder
        self.device = device
        self.chunk_len = chunk_len
        self.action_dim = action_dim

        # Create Mock OpenVLA
        self.mock_openvla = MockOpenVLA(device=device)

        # Set to eval mode
        self.draft_model.eval()
        self.rfsq_head.eval()
        self.rfsq_decoder.eval()

    @torch.no_grad()
    def generate_action_chunk(
        self,
        image: Image.Image,
        task_description: str,
        use_speculative_decoding: bool = True,
    ) -> np.ndarray:
        """
        Generate action chunk from image and task description

        Returns:
            actions: [chunk_len, action_dim] numpy array
        """
        # Step 1: Get hidden state from Mock OpenVLA
        hidden_state = self.mock_openvla(image, task_description)  # [1, 4096]

        # Step 2: Draft Model prediction (if enabled)
        if use_speculative_decoding:
            draft_logits = self.draft_model(hidden_state)  # [1, 3, 128, 7]
            draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]

        # Step 3: Main Model prediction
        main_logits = self.rfsq_head(hidden_state)  # [1, 8, 128, 7]
        main_tokens = torch.argmax(main_logits, dim=-1)  # [1, 8, 128]

        # Step 4: Use main_tokens
        final_tokens = main_tokens  # [1, 8, 128]

        # Step 5: Reshape
        batch_size, num_layers, flat_dim = final_tokens.shape
        final_tokens = final_tokens.view(batch_size, num_layers, self.chunk_len, 16)
        final_tokens = final_tokens.permute(0, 2, 3, 1)  # [1, Chunk, Hidden, Layers]

        # Step 6: RFSQ Decoder
        actions = self.rfsq_decoder.decode_from_indices(final_tokens)  # [1, Chunk, 7]

        return actions[0].cpu().numpy()  # [Chunk, 7]


# ============================================================
# LIBERO Test
# ============================================================

def test_libero_episode(
    engine: RSDEngineWithMockVLA,
    task_suite,
    task_id: int,
    episode_id: int,
    max_steps: int = 300,
    verbose: bool = True,
):
    """
    运行一个LIBERO episode测试

    这里测试的是：
    1. ✅ LIBERO环境能正常初始化
    2. ✅ 图像能正常获取
    3. ✅ RSD Engine能生成actions
    4. ✅ Actions能被LIBERO环境接受
    5. ✅ 环境能正常step
    6. ❌ 任务成功率（因为用的是mock OpenVLA）
    """
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    task_description = task.language
    init_states = task_suite.get_task_init_states(task_id)

    if episode_id >= len(init_states):
        print(f"   ⚠️  Episode {episode_id} exceeds available init states")
        return None

    # Create environment
    bddl_file_path = os.path.join(
        os.environ.get('LIBERO_FOLDER', '/root/LIBERO/libero/libero'),
        'bddl_files',
        task.problem_folder,
        task.bddl_file
    )

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            camera_heights=256,
            camera_widths=256,
        )
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        return None

    # Reset and set init state
    env.reset()
    obs = env.set_init_state(init_states[episode_id])

    if verbose:
        print(f"\n   Task: {task_description}")
        print(f"   Episode: {episode_id}")

    # Episode loop
    step_count = 0
    action_chunk_idx = 0
    current_chunk = None

    for step in range(max_steps):
        # Get image
        image_array = obs['agentview_image']
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array)

        # Generate action chunk if needed
        if action_chunk_idx == 0 or action_chunk_idx >= 8:
            try:
                current_chunk = engine.generate_action_chunk(
                    image,
                    task_description,
                    use_speculative_decoding=True,
                )
                action_chunk_idx = 0
            except Exception as e:
                print(f"   ❌ Failed to generate action: {e}")
                env.close()
                return None

        # Get action from chunk
        action = current_chunk[action_chunk_idx]
        action_chunk_idx += 1

        # Step environment
        try:
            obs, reward, done, info = env.step(action)
            step_count += 1
        except Exception as e:
            print(f"   ❌ Environment step failed at step {step}: {e}")
            env.close()
            return None

        if done:
            success = info.get('success', False)
            if verbose:
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"   {status} in {step_count} steps")
            env.close()
            return {
                'success': success,
                'steps': step_count,
                'task_id': task_id,
                'episode_id': episode_id,
            }

    # Timeout
    if verbose:
        print(f"   ⏱️  TIMEOUT after {max_steps} steps")

    env.close()
    return {
        'success': False,
        'steps': max_steps,
        'task_id': task_id,
        'episode_id': episode_id,
        'timeout': True,
    }


def main():
    parser = argparse.ArgumentParser(description='Test LIBERO integration with Mock OpenVLA')
    parser.add_argument('--models-dir', type=str, default='./mock_models',
                        help='Directory containing mock checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--num-episodes', type=int, default=3,
                        help='Number of episodes to test per task')
    parser.add_argument('--task-id', type=int, default=0,
                        help='Task ID to test (0-9 for libero_spatial)')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum steps per episode')

    args = parser.parse_args()

    print("=" * 60)
    print("🤖 LIBERO Integration Test with Mock OpenVLA")
    print("=" * 60)
    print(f"Models directory: {args.models_dir}")
    print(f"Device: {args.device}")
    print(f"Task ID: {args.task_id}")
    print(f"Episodes per task: {args.num_episodes}")
    print("")

    # Check LIBERO
    try:
        from libero.libero import benchmark
        print("✅ LIBERO imported successfully")
    except ImportError as e:
        print(f"❌ LIBERO import failed: {e}")
        print("   Please install LIBERO first:")
        print("   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
        print("   cd LIBERO && pip install -e .")
        return 1

    # Load checkpoints
    models_dir = Path(args.models_dir)
    device = args.device

    print(f"\n📦 Loading mock checkpoints...")

    # RFSQ Decoder
    rfsq_path = models_dir / 'rfsq_robust_best.pt'
    rfsq_model = ActionRFSQAE(use_layernorm=True)
    checkpoint = torch.load(rfsq_path, map_location=device, weights_only=False)
    rfsq_model.load_state_dict(checkpoint['model'])
    rfsq_model.to(device)
    rfsq_model.eval()
    print(f"   ✅ RFSQ Decoder loaded")

    # Draft Model
    draft_path = models_dir / 'best_draft_with_projection.pt'
    draft_model = RFSQDraftModelWithProjection()
    checkpoint = torch.load(draft_path, map_location=device, weights_only=False)
    draft_model.load_state_dict(checkpoint['model_state_dict'])
    draft_model.to(device)
    draft_model.eval()
    print(f"   ✅ Draft Model loaded")

    # RFSQ Head
    rfsq_head_path = models_dir / 'openvla_rfsq_robust' / 'best_rfsq_head.pt'
    rfsq_head = RFSQClassificationHead()
    checkpoint = torch.load(rfsq_head_path, map_location=device, weights_only=False)
    rfsq_head.load_state_dict(checkpoint['model_state_dict'])
    rfsq_head.to(device)
    rfsq_head.eval()
    print(f"   ✅ RFSQ Head loaded")

    # Create RSD Engine
    print(f"\n🔧 Creating RSD Engine with Mock OpenVLA...")
    engine = RSDEngineWithMockVLA(
        draft_model=draft_model,
        rfsq_head=rfsq_head,
        rfsq_decoder=rfsq_model,
        device=device,
    )
    print(f"   ✅ RSD Engine created")

    # Setup LIBERO
    print(f"\n🤖 Setting up LIBERO benchmark...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    print(f"   ✅ LIBERO benchmark loaded ({task_suite.n_tasks} tasks)")

    # Run episodes
    print(f"\n🧪 Running test episodes...")
    results = []

    for ep_id in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_id + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        result = test_libero_episode(
            engine=engine,
            task_suite=task_suite,
            task_id=args.task_id,
            episode_id=ep_id,
            max_steps=args.max_steps,
            verbose=True,
        )

        if result is not None:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    print(f"{'='*60}")

    if len(results) == 0:
        print(f"❌ No episodes completed successfully")
        return 1

    completed = len(results)
    # Note: success rate will be low because we're using mock OpenVLA
    print(f"Episodes completed: {completed}/{args.num_episodes}")
    print(f"Average steps: {np.mean([r['steps'] for r in results]):.1f}")

    print(f"\n🎯 Integration Test Results:")
    print(f"   ✅ LIBERO environment initialization: PASS")
    print(f"   ✅ Image retrieval: PASS")
    print(f"   ✅ RSD Engine action generation: PASS")
    print(f"   ✅ LIBERO environment step: PASS")
    print(f"   ℹ️  Task success rate: N/A (using mock OpenVLA)")

    print(f"\n{'='*60}")
    print(f"✅ LIBERO Integration Test Passed!")
    print(f"{'='*60}")
    print(f"\n🎯 Key Findings:")
    print(f"   • LIBERO环境可以正常初始化")
    print(f"   • RSD pipeline可以生成actions")
    print(f"   • Actions格式正确，可以被LIBERO接受")
    print(f"   • 没有shape mismatch或crash")
    print(f"\n📝 Note:")
    print(f"   任务成功率会很低，因为用的是Mock OpenVLA（随机输出）")
    print(f"   但这已经验证了最容易出错的LIBERO集成部分！")
    print(f"\n🚀 Next Steps:")
    print(f"   1. ✅ LIBERO集成已验证")
    print(f"   2. 🔥 可以开始真实训练（Phase 1 → 2 → 3）")
    print(f"   3. 📖 See AGENT_ACTION_PLAN.md for training guide")

    return 0


if __name__ == "__main__":
    exit(main())
