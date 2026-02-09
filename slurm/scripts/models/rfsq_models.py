"""
RFSQ Model Definitions for SLURM Training

This file contains all model classes needed for the RVQ experiment:
1. ActionRFSQAE: Robust RFSQ AutoEncoder (Phase 1)
2. RFSQDraftModelWithProjection: Draft Model (Phase 2 - predicts L0-L2)
3. ConditionedRFSQHead: Conditional Main Model (Phase 2 - predicts all layers conditioned on L0-L2)

All models are extracted from Modal scripts and converted to standalone PyTorch modules.
"""

import torch
import torch.nn as nn


# ============================================================
# Phase 1: Robust RFSQ AutoEncoder Components
# ============================================================

class RobustSTEQuantizer(nn.Module):
    """
    Improved STE quantizer with LayerNorm strategy.

    Key features:
    - LayerNorm normalization of residual signals
    - Quantization in normalized space
    - Inverse LayerNorm to restore scale
    - Maintains effectiveness of deeper layers

    Parameters:
        num_levels: Number of quantization levels (default 7)
        use_layernorm: Whether to use LayerNorm strategy (default True)
    """

    def __init__(self, num_levels=7, use_layernorm=True):
        super().__init__()
        self.num_levels = num_levels
        self.use_layernorm = use_layernorm

        # Quantization boundaries [-1, 1]
        self.register_buffer('boundaries', torch.linspace(-1, 1, num_levels))

    def forward(self, z):
        """
        Forward pass with optional LayerNorm.

        Args:
            z: [Batch, Seq, Dim] - residual signal

        Returns:
            z_q: [Batch, Seq, Dim] - quantized values (original scale)
            indices: [Batch, Seq, Dim] - discrete indices [0, num_levels-1]
        """
        if self.use_layernorm:
            # LayerNorm strategy: normalize -> quantize -> denormalize

            # Step 1: Save original scale information
            original_mean = z.mean(dim=-1, keepdim=True)  # [B, S, 1]
            original_std = z.std(dim=-1, keepdim=True) + 1e-5  # [B, S, 1]

            # Step 2: Normalize
            z_norm = (z - original_mean) / original_std  # [B, S, D]

            # Step 3: Quantize (in normalized space)
            dist = torch.abs(z_norm.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)  # [B, S, D]
            z_q_norm = self.boundaries[indices]  # [B, S, D]

            # Step 4: Denormalize
            z_q = z_q_norm * original_std + original_mean  # [B, S, D]
        else:
            # Original strategy: direct quantization
            dist = torch.abs(z.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(dist, dim=-1)
            z_q = self.boundaries[indices]

        # Straight-Through Estimator (gradient bypass)
        z_q_out = z + (z_q - z).detach()

        return z_q_out, indices


class RobustRFSQBlock(nn.Module):
    """
    Improved RFSQ Block (multi-layer residual quantization).

    Key features:
    - Uses RobustSTEQuantizer instead of naive quantizer
    - LayerNorm strategy applied per layer
    - Deeper layers (L3-L7) become effective again

    Parameters:
        num_layers: Number of layers (default 8)
        num_levels: Quantization levels per layer (default 7)
        use_layernorm: Whether to use LayerNorm strategy (default True)
    """

    def __init__(self, num_layers=8, num_levels=7, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.use_layernorm = use_layernorm

        # Each layer is an independent quantizer
        self.layers = nn.ModuleList([
            RobustSTEQuantizer(num_levels=num_levels, use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])

    def forward(self, z):
        """
        Residual quantization with LayerNorm.

        Args:
            z: [Batch, Seq, Dim] - input latent

        Returns:
            quantized_sum: [Batch, Seq, Dim] - quantized reconstruction
            codes: [Batch, Seq, Dim, Num_Layers] - discrete codes
        """
        residual = z
        quantized_sum = 0
        all_indices = []

        for layer_idx, layer in enumerate(self.layers):
            # Quantize current residual
            z_q, indices = layer(residual)

            # Accumulate quantized values
            quantized_sum = quantized_sum + z_q

            # Update residual
            residual = residual - z_q

            # Record indices
            all_indices.append(indices)

        # Stack codes: [B, S, D, L]
        codes = torch.stack(all_indices, dim=-1)

        return quantized_sum, codes

    def decode_from_indices(self, indices):
        """
        Decode from discrete indices back to continuous latent.

        Args:
            indices: [Batch, Seq, Dim, Num_Layers] - discrete codes

        Returns:
            reconstruction: [Batch, Seq, Dim] - reconstructed latent
        """
        batch_size, seq_len, dim, num_layers = indices.shape
        assert num_layers == self.num_layers, f"Expected {self.num_layers} layers, got {num_layers}"

        # Initialize reconstruction
        reconstruction = torch.zeros(batch_size, seq_len, dim, device=indices.device)

        # Accumulate layer by layer
        for layer_idx in range(num_layers):
            layer_indices = indices[:, :, :, layer_idx]  # [B, S, D]
            layer_values = self.layers[layer_idx].boundaries[layer_indices]
            reconstruction = reconstruction + layer_values

        return reconstruction


class ActionRFSQAE(nn.Module):
    """
    Improved Action RFSQ AutoEncoder.

    Key features:
    - Uses RobustRFSQBlock instead of naive RFSQ
    - Automatic LayerNorm precision improvement
    - Unchanged encoder/decoder architecture

    Parameters:
        action_dim: Action dimension (default 7)
        hidden_dim: Latent dimension (default 16)
        num_layers: RFSQ layers (default 8)
        num_levels: Quantization levels (default 7)
        use_layernorm: Whether to use LayerNorm strategy (default True)
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

        # Encoder: action -> latent
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Mish(),
            nn.Linear(64, hidden_dim),
            nn.Tanh()
        )

        # RFSQ Block (improved version)
        self.rfsq = RobustRFSQBlock(
            num_layers=num_layers,
            num_levels=num_levels,
            use_layernorm=use_layernorm,
        )

        # Decoder: latent -> action
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Mish(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        """
        Forward pass: encode -> quantize -> decode.

        Args:
            x: [Batch, Seq, Action_Dim] - action sequence

        Returns:
            x_recon: [Batch, Seq, Action_Dim] - reconstructed actions
            codes: [Batch, Seq, Hidden_Dim, Num_Layers] - discrete codes
        """
        # Encode
        z = self.encoder(x)  # [B, S, Hidden]

        # Quantize (with LayerNorm if enabled)
        z_quantized, codes = self.rfsq(z)  # [B, S, Hidden], [B, S, H, L]

        # Decode
        x_recon = self.decoder(z_quantized)  # [B, S, Action]

        return x_recon, codes

    def encode(self, x):
        """Encode only, return codes."""
        z = self.encoder(x)
        _, codes = self.rfsq(z)
        return codes

    def decode_from_indices(self, indices):
        """
        Decode from discrete indices back to actions.

        Args:
            indices: [Batch, Chunk, Hidden_Dim, Num_Layers]

        Returns:
            actions: [Batch, Chunk, Action_Dim]
        """
        batch_size, chunk_len, hidden_dim, num_layers = indices.shape

        # RFSQ decode: indices -> latent
        z_reconstructed = self.rfsq.decode_from_indices(indices)

        # Decoder: latent -> actions
        # Reshape for decoder: [B, C, H] -> [B*C, H]
        z_flat = z_reconstructed.view(-1, self.hidden_dim)
        actions_flat = self.decoder(z_flat)

        # Reshape back: [B*C, A] -> [B, C, A]
        actions = actions_flat.view(batch_size, chunk_len, -1)

        return actions


# ============================================================
# Phase 2: Draft Model Components
# ============================================================

class DraftTransformerDecoder(nn.Module):
    """Simple transformer decoder for Draft Model."""

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

        self.position_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_dim) * 0.02
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        pos_enc = self.position_encoding[:, :seq_len, :]
        hidden_states = hidden_states + pos_enc
        output = self.decoder_layer(hidden_states, hidden_states)
        return self.output_norm(output)


class RFSQDraftModelWithProjection(nn.Module):
    """
    Draft Model: Predicts coarse RFSQ tokens (L0-L2) from OpenVLA hidden states.

    Architecture:
    - Input projection: 4096 -> 512
    - Transformer decoder: Single layer
    - 3 parallel heads: One per coarse layer (L0, L1, L2)

    Output: Logits for L0-L2 tokens
    """

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
# Phase 2: Conditional Main Model
# ============================================================

class ConditionedRFSQHead(nn.Module):
    """
    Conditional RFSQ Head with Mode Locking.

    Key Innovation:
    - Accepts L0-L2 tokens as conditioning input (ground truth during training)
    - Fuses image features with token embeddings
    - Predicts all 8 layers, enabling verification during inference
    - Solves mean-seeking problem by locking model into specific mode

    Architecture Flow:
    1. Image features [B, 4096] → feature_proj → [B, 1024]
    2. Condition tokens [B, 8, 16, 3] → embedding → [B, 8, 16, 3, 64]
    3. Flatten & project: [B, 24576] → token_proj → [B, 1024]
    4. Fusion: concat([img_feat, token_feat]) → [B, 2048] → [B, 1024]
    5. 8 parallel heads predict all layers from fused features
    """

    def __init__(
        self,
        input_dim=4096,           # OpenVLA hidden state dimension
        hidden_dim=1024,          # Internal processing dimension
        num_layers=8,             # Total RFSQ layers (L0-L7)
        chunk_len=8,              # Action chunk length
        action_hidden_dim=16,     # RFSQ latent dimension
        grid_size=7,              # Quantization levels (0-6)
        condition_layers=3,       # Layers to condition on (L0-L2)
        token_embed_dim=64,       # Embedding dimension per token
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size
        self.condition_layers = condition_layers
        self.token_embed_dim = token_embed_dim

        # A. Image Feature Projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # B. Token Embedding Layer
        # Maps discrete tokens [0-6] to continuous embeddings [64-dim]
        self.token_embedding = nn.Embedding(grid_size, token_embed_dim)

        # C. Token Projection Layer
        # Input: [Batch, Chunk=8, Hidden=16, Layers=3] with 64-dim embeddings
        # Total: 8 * 16 * 3 * 64 = 24,576 dimensions
        token_flat_dim = chunk_len * action_hidden_dim * condition_layers * token_embed_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_flat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # D. Fusion Layer
        # Combines image features [1024] + token features [1024] → [1024]
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # E. Output Heads
        # 8 parallel heads, one per RFSQ layer
        output_size = chunk_len * action_hidden_dim * grid_size  # 8*16*7 = 896
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_size),
            )
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, condition_tokens):
        """
        Forward pass with conditional input.

        Args:
            hidden_states: [Batch, 4096] - OpenVLA image features
            condition_tokens: [Batch, 8, 16, 3] - L0-L2 tokens (ground truth during training)

        Returns:
            logits: [Batch, 8, 128, 7] - Predictions for all 8 layers
                    Shape breakdown:
                    - Batch: batch size
                    - 8: number of RFSQ layers (L0-L7)
                    - 128: chunk_len(8) * action_hidden_dim(16)
                    - 7: grid_size (quantization levels)
        """
        batch_size = hidden_states.shape[0]

        # Step 1: Process image features
        img_feat = self.feature_proj(hidden_states)  # [B, 1024]

        # Step 2: Process condition tokens
        # condition_tokens: [B, 8, 16, 3] with integer values in [0, 6]
        token_embeds = self.token_embedding(condition_tokens)  # [B, 8, 16, 3, 64]

        # Step 3: Flatten all token embeddings
        token_flat = token_embeds.view(batch_size, -1)  # [B, 24576]
        token_feat = self.token_proj(token_flat)  # [B, 1024]

        # Step 4: Fuse image and token features
        # MODE LOCKING HAPPENS HERE: token features modulate the hidden state
        combined = torch.cat([img_feat, token_feat], dim=-1)  # [B, 2048]
        fused_feat = self.fusion(combined)  # [B, 1024]

        # Step 5: Predict all 8 layers using fused features
        layer_outputs = []
        for head in self.layer_heads:
            logits = head(fused_feat)  # [B, 896]
            logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
            # Reshape to [B, 128, 7]
            layer_outputs.append(logits)

        return torch.stack(layer_outputs, dim=1)  # [B, 8, 128, 7]


# ============================================================
# Helper Functions
# ============================================================

def create_rfsq_ae(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
    use_layernorm=True,
    device='cuda',
):
    """
    Create RFSQ AutoEncoder.

    Args:
        action_dim: Action dimension
        hidden_dim: Latent dimension
        num_layers: RFSQ layers
        num_levels: Quantization levels
        use_layernorm: Whether to use LayerNorm strategy
        device: Device

    Returns:
        model: ActionRFSQAE instance
    """
    model = ActionRFSQAE(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_levels=num_levels,
        use_layernorm=use_layernorm,
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ RFSQ AutoEncoder: {total_params:,} parameters")

    return model


def create_draft_model(
    input_dim=4096,
    hidden_dim=512,
    num_coarse_layers=3,
    chunk_len=8,
    action_hidden_dim=16,
    grid_size=7,
    device='cuda',
):
    """Create Draft Model."""
    model = RFSQDraftModelWithProjection(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_coarse_layers=num_coarse_layers,
        chunk_len=chunk_len,
        action_hidden_dim=action_hidden_dim,
        grid_size=grid_size,
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Draft Model: {total_params:,} parameters")

    return model


def create_conditional_model(
    input_dim=4096,
    hidden_dim=1024,
    num_layers=8,
    chunk_len=8,
    action_hidden_dim=16,
    grid_size=7,
    condition_layers=3,
    token_embed_dim=64,
    device='cuda',
):
    """Create Conditional Main Model."""
    model = ConditionedRFSQHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        chunk_len=chunk_len,
        action_hidden_dim=action_hidden_dim,
        grid_size=grid_size,
        condition_layers=condition_layers,
        token_embed_dim=token_embed_dim,
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Conditional Model: {total_params:,} parameters")

    return model
