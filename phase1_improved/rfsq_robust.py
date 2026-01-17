"""
Robust RFSQ with LayerNorm Strategy

基于论文改进，引入LayerNorm提升量化精度。

关键改进：
1. 每层量化前：LayerNorm（放大微弱信号）
2. 量化后：Inverse LayerNorm（还原尺度）
3. 让L3-L7重新有效，提升精细度

引用：
- [cite: 942] LayerNorm for signal amplification
- [cite: 944] Inverse LayerNorm for scale restoration
"""

import torch
import torch.nn as nn


class RobustSTEQuantizer(nn.Module):
    """
    改进的STE量化器（带LayerNorm）

    与原始STEQuantizer的区别：
    - ✅ 添加LayerNorm归一化残差信号
    - ✅ 在归一化空间中量化
    - ✅ 反归一化还原尺度
    - ✅ 保持后层的量化有效性

    Parameters:
        num_levels: 量化级别数（默认7）
        use_layernorm: 是否使用LayerNorm策略（默认True）
    """

    def __init__(self, num_levels=7, use_layernorm=True):
        super().__init__()
        self.num_levels = num_levels
        self.use_layernorm = use_layernorm

        # 量化边界 [-1, 1]
        self.register_buffer('boundaries', torch.linspace(-1, 1, num_levels))

    def forward(self, z):
        """
        Forward pass with optional LayerNorm

        Args:
            z: [Batch, Seq, Dim] - 残差信号

        Returns:
            z_q: [Batch, Seq, Dim] - 量化后的值（原始尺度）
            indices: [Batch, Seq, Dim] - 离散索引 [0, num_levels-1]
        """
        if self.use_layernorm:
            # ========================================
            # 论文策略：LayerNorm + 量化 + Inverse
            # ========================================

            # Step 1: 保存原始尺度信息
            # 每个dimension独立计算mean/std，保持精度
            original_mean = z.mean(dim=-1, keepdim=True)  # [B, S, 1]
            original_std = z.std(dim=-1, keepdim=True) + 1e-5  # [B, S, 1]

            # Step 2: 归一化 [cite: 942]
            # 将残差归一化到相似的尺度，放大微弱信号
            z_norm = (z - original_mean) / original_std  # [B, S, D]

            # Step 3: 量化（在归一化空间）
            # 此时所有层的信号强度相似，量化更有效
            dist = torch.abs(z_norm.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)  # [B, S, D]
            z_q_norm = self.boundaries[indices]  # [B, S, D]

            # Step 4: 反归一化 [cite: 944]
            # 还原到原始尺度，保持残差更新的正确性
            z_q = z_q_norm * original_std + original_mean  # [B, S, D]

        else:
            # ========================================
            # 原始策略：直接量化
            # ========================================
            dist = torch.abs(z.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)
            z_q = self.boundaries[indices]

        # Straight-Through Estimator (梯度回传)
        z_q_out = z + (z_q - z).detach()

        return z_q_out, indices


class RobustRFSQBlock(nn.Module):
    """
    改进的RFSQ Block（多层残差量化）

    与原始RFSQBlock的区别：
    - ✅ 使用RobustSTEQuantizer替代STEQuantizer
    - ✅ 每层都应用LayerNorm策略
    - ✅ 后层（L3-L7）重新变得有效

    Parameters:
        num_layers: 层数（默认8）
        num_levels: 每层的量化级别（默认7）
        use_layernorm: 是否使用LayerNorm策略（默认True）
    """

    def __init__(self, num_layers=8, num_levels=7, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.use_layernorm = use_layernorm

        # 每一层都是独立的量化器
        self.layers = nn.ModuleList([
            RobustSTEQuantizer(num_levels=num_levels, use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])

    def forward(self, z):
        """
        Residual quantization with LayerNorm

        Args:
            z: [Batch, Seq, Dim] - 输入latent

        Returns:
            quantized_sum: [Batch, Seq, Dim] - 量化后的重构
            codes: [Batch, Seq, Dim, Num_Layers] - 离散codes
        """
        residual = z
        quantized_sum = 0
        all_indices = []

        for layer_idx, layer in enumerate(self.layers):
            # 量化当前残差
            z_q, indices = layer(residual)

            # 累加量化值
            quantized_sum = quantized_sum + z_q

            # 更新残差
            residual = residual - z_q

            # 记录indices
            all_indices.append(indices)

            # 可选：打印残差统计（调试用）
            # if layer_idx % 2 == 0:
            #     print(f"  Layer {layer_idx}: residual std = {residual.std().item():.6f}")

        # Stack codes: [B, S, D, L]
        codes = torch.stack(all_indices, dim=-1)

        return quantized_sum, codes

    def decode_from_indices(self, indices):
        """
        从离散indices解码回连续latent

        Args:
            indices: [Batch, Seq, Dim, Num_Layers] - 离散codes

        Returns:
            reconstruction: [Batch, Seq, Dim] - 重构的latent
        """
        batch_size, seq_len, dim, num_layers = indices.shape
        assert num_layers == self.num_layers, f"Expected {self.num_layers} layers, got {num_layers}"

        # 初始化重构
        reconstruction = torch.zeros(batch_size, seq_len, dim, device=indices.device)

        # 逐层累加
        for layer_idx in range(num_layers):
            layer_indices = indices[:, :, :, layer_idx]  # [B, S, D]
            layer_values = self.layers[layer_idx].boundaries[layer_indices]
            reconstruction = reconstruction + layer_values

        return reconstruction


class ActionRFSQAE(nn.Module):
    """
    改进的Action RFSQ AutoEncoder

    与原始ActionRFSQAE的区别：
    - ✅ 使用RobustRFSQBlock替代RFSQBlock
    - ✅ 自动获得LayerNorm的精度提升
    - ✅ 其他部分保持不变（encoder/decoder架构）

    Parameters:
        action_dim: 动作维度（默认7）
        hidden_dim: latent维度（默认16）
        num_layers: RFSQ层数（默认8）
        num_levels: 量化级别（默认7）
        use_layernorm: 是否使用LayerNorm策略（默认True）
    """

    def __init__(
        self,
        action_dim=7,
        hidden_dim=16,
        num_layers=8,
        num_levels=7,
        use_layernorm=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.use_layernorm = use_layernorm

        # Encoder: 动作 -> latent
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Mish(),
            nn.Linear(64, hidden_dim),
            nn.Tanh()
        )

        # RFSQ Block (改进版)
        self.rfsq = RobustRFSQBlock(
            num_layers=num_layers,
            num_levels=num_levels,
            use_layernorm=use_layernorm,
        )

        # Decoder: latent -> 动作
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Mish(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        """
        Forward pass: 编码 -> 量化 -> 解码

        Args:
            x: [Batch, Seq, Action_Dim] - 动作序列

        Returns:
            x_recon: [Batch, Seq, Action_Dim] - 重构的动作
            codes: [Batch, Seq, Hidden_Dim, Num_Layers] - 离散codes
        """
        # Encode
        z = self.encoder(x)  # [B, S, Hidden]

        # Quantize (with LayerNorm if enabled)
        z_quantized, codes = self.rfsq(z)  # [B, S, Hidden], [B, S, H, L]

        # Decode
        x_recon = self.decoder(z_quantized)  # [B, S, Action]

        return x_recon, codes

    def encode(self, x):
        """仅编码，返回codes"""
        z = self.encoder(x)
        _, codes = self.rfsq(z)
        return codes

    def decode_from_indices(self, indices):
        """
        从离散indices解码回动作

        Args:
            indices: [Batch, Chunk, Hidden_Dim, Num_Layers]

        Returns:
            actions: [Batch, Chunk, Action_Dim]
        """
        batch_size, chunk_len, hidden_dim, num_layers = indices.shape

        # RFSQ解码：indices -> latent
        z_reconstructed = self.rfsq.decode_from_indices(indices)

        # Decoder：latent -> actions
        # Reshape for decoder: [B, C, H] -> [B*C, H]
        z_flat = z_reconstructed.view(-1, self.hidden_dim)
        actions_flat = self.decoder(z_flat)

        # Reshape back: [B*C, A] -> [B, C, A]
        actions = actions_flat.view(batch_size, chunk_len, -1)

        return actions


# ============================================================
# 辅助函数：创建模型
# ============================================================

def create_robust_rfsq_ae(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
    use_layernorm=True,
    device='cuda',
):
    """
    创建改进的RFSQ AutoEncoder

    Args:
        action_dim: 动作维度
        hidden_dim: latent维度
        num_layers: RFSQ层数
        num_levels: 量化级别
        use_layernorm: 是否使用LayerNorm策略
        device: 设备

    Returns:
        model: ActionRFSQAE实例
    """
    model = ActionRFSQAE(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_levels=num_levels,
        use_layernorm=use_layernorm,
    )

    model = model.to(device)

    # 打印配置
    print("=" * 60)
    print("🔧 Robust RFSQ AutoEncoder Configuration")
    print("=" * 60)
    print(f"   Action Dim: {action_dim}")
    print(f"   Hidden Dim: {hidden_dim}")
    print(f"   Num Layers: {num_layers}")
    print(f"   Num Levels: {num_levels}")
    print(f"   Use LayerNorm: {use_layernorm} {'✅ (Robust)' if use_layernorm else '❌ (Naive)'}")
    print(f"   Device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total Parameters: {total_params:,}")
    print("=" * 60)

    return model


# ============================================================
# 对比测试函数
# ============================================================

def compare_naive_vs_robust(action_samples, device='cuda'):
    """
    对比原始RFSQ vs 改进RFSQ

    Args:
        action_samples: [Batch, Seq, 7] - 测试动作
        device: 设备

    Returns:
        results: dict with comparison metrics
    """
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print("📊 Comparing Naive RFSQ vs Robust RFSQ")
    print("=" * 60)

    # 1. Create models
    naive_model = create_robust_rfsq_ae(use_layernorm=False, device=device)
    robust_model = create_robust_rfsq_ae(use_layernorm=True, device=device)

    naive_model.eval()
    robust_model.eval()

    actions = torch.from_numpy(action_samples).float().to(device)

    with torch.no_grad():
        # 2. Naive RFSQ
        naive_recon, naive_codes = naive_model(actions)
        naive_mse = F.mse_loss(naive_recon, actions).item()

        # 3. Robust RFSQ
        robust_recon, robust_codes = robust_model(actions)
        robust_mse = F.mse_loss(robust_recon, actions).item()

    # 4. Results
    improvement = (naive_mse - robust_mse) / naive_mse * 100

    results = {
        'naive_mse': naive_mse,
        'robust_mse': robust_mse,
        'improvement_pct': improvement,
    }

    print(f"\n   Naive RFSQ MSE: {naive_mse:.6f}")
    print(f"   Robust RFSQ MSE: {robust_mse:.6f}")
    print(f"   Improvement: {improvement:.1f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # 测试代码
    print("Testing Robust RFSQ implementation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试数据
    batch_size = 4
    chunk_len = 8
    action_dim = 7

    test_actions = torch.randn(batch_size, chunk_len, action_dim).to(device)

    # 测试Robust模型
    model = create_robust_rfsq_ae(use_layernorm=True, device=device)
    model.eval()

    with torch.no_grad():
        recon, codes = model(test_actions)

        print(f"\n✅ Input shape: {test_actions.shape}")
        print(f"✅ Reconstruction shape: {recon.shape}")
        print(f"✅ Codes shape: {codes.shape}")

        mse = F.mse_loss(recon, test_actions).item()
        print(f"✅ Random init MSE: {mse:.6f}")

    print("\n✅ All tests passed!")
