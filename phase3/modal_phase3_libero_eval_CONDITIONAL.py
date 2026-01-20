"""
Phase 3: LIBERO Evaluation with CONDITIONAL RSD Inference Engine
Mode Locking Version - Uses Conditional RFSQ Head with Token Embedding

KEY DIFFERENCES from baseline:
1. ConditionedRFSQHead: Accepts L0-L2 tokens as conditioning input
2. Mode Locking: Fuses image features with token embeddings
3. Solves mean-seeking problem by locking model into specific mode
4. Training-inference consistency: Matches Phase 2 conditional training

Architecture Flow:
- Draft Model predicts L0-L2 from image features
- Main Model receives image + Draft's L0-L2 tokens → MODE LOCKING
- Main Model predicts all 8 layers (L0-L7) conditioned on Draft output
- Verification: Check if Main's L0-L2 matches Draft's L0-L2

Usage:
    # Test with few trials
    modal run phase3/modal_phase3_libero_eval_CONDITIONAL.py --num-trials 3

    # Full evaluation
    modal run phase3/modal_phase3_libero_eval_CONDITIONAL.py --task-suite libero_spatial --num-trials 50
"""

import os
import sys
import modal
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

# ============================================================
# Modal App Setup
# ============================================================
app = modal.App("rsd-phase3-libero-eval-conditional")

# Volumes
data_volume = modal.Volume.from_name("rsd-libero-data", create_if_missing=True)
models_volume = modal.Volume.from_name("rsd-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("rsd-results", create_if_missing=True)

# Get SDK path
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

# Build evaluation image
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev",
                 "libosmesa6-dev", "libglew-dev", "patchelf")
    .pip_install("uv")
    .run_commands(
        # Install PyTorch and deps
        "uv pip install --system 'numpy<2' torch==2.2.0 torchvision==0.17.0 "
        "transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "
        "accelerate peft bitsandbytes pillow einops sentencepiece protobuf "
        "huggingface_hub scipy tqdm matplotlib pandas requests json-numpy jsonlines",
    )
    # Clone and install LIBERO
    .run_commands(
        "cd /root && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "cd /root/LIBERO && sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' "
        "libero/libero/benchmark/__init__.py",
        "mkdir -p /root/.libero",
        "echo 'benchmark_root: /root/LIBERO/libero/libero' > /root/.libero/config.yaml",
        "echo 'bddl_files: /root/LIBERO/libero/libero/bddl_files' >> /root/.libero/config.yaml",
        "echo 'init_states: /root/LIBERO/libero/libero/init_files' >> /root/.libero/config.yaml",
        "cd /root/LIBERO && uv pip install --system -e .",
        "uv pip install --system mujoco dm-control robosuite==1.4.0 termcolor h5py bddl easydict cloudpickle gym gymnasium",
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
        "HF_HOME": "/hf_cache",
        "TRANSFORMERS_CACHE": "/hf_cache",
        "LIBERO_NO_PROMPT": "1",
        "LIBERO_FOLDER": "/data/libero",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)


# ============================================================
# Helper: OpenVLA Action Extraction
# ============================================================

def safe_extract_action(action_result):
    """Extract action from OpenVLA predict_action result."""
    import torch
    import numpy as np

    # Step 1: Handle tuple
    if isinstance(action_result, tuple):
        if len(action_result) > 0:
            action = action_result[0]
        else:
            return None
    else:
        action = action_result

    # Step 2: Convert to numpy
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    elif isinstance(action, list):
        action = np.array(action, dtype=np.float32)
    elif not isinstance(action, np.ndarray):
        try:
            action = np.array(action, dtype=np.float32)
        except:
            return None

    # Step 3: Handle action chunk [8, 7] -> [7]
    if action.ndim == 2:
        if action.shape[0] == 8 and action.shape[1] == 7:
            action = action[0]
        elif action.shape == (1, 7):
            action = action.squeeze(0)
        else:
            action = action.flatten()
    elif action.ndim == 3:
        action = action.squeeze(0)
        if action.shape[0] == 8 and action.shape[1] == 7:
            action = action[0]
        else:
            action = action.flatten()
    elif action.ndim > 3:
        action = action.flatten()

    # Step 4: Ensure 1D
    if action.ndim > 1:
        action = action.flatten()

    # Step 5: Adjust to shape (7,)
    if action.shape[0] == 0:
        return None
    elif action.shape[0] > 7:
        action = action[:7]
    elif action.shape[0] < 7:
        action = np.pad(action, (0, 7 - action.shape[0]), 'constant')

    return action.astype(np.float32)


# ============================================================
# Evaluation Function
# ============================================================

@app.function(
    image=eval_image,
    gpu="A100",
    timeout=28800,  # 8 hours
    volumes={
        "/data": data_volume,
        "/models": models_volume,
        "/hf_cache": hf_cache,
        "/results": results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("orchestra-supabase"),
    ],
)
def run_libero_evaluation(
    task_suite: str = "libero_spatial",
    num_trials: int = 50,
    use_speculative_decoding: bool = True,
):
    """Run LIBERO evaluation with CONDITIONAL RSD models."""
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import numpy as np
    import time
    import json
    from pathlib import Path
    from PIL import Image as PILImage

    # Add paths
    sys.path.insert(0, "/root/LIBERO")
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    # Try to import experiment tracking
    try:
        from src.orchestra_sdk.experiment import Experiment
        use_experiment_tracking = True
    except ImportError:
        print("⚠️ Orchestra SDK not available, skipping experiment tracking")
        use_experiment_tracking = False

    print("=" * 80)
    print(f"🚀 Phase 3: LIBERO Evaluation (CONDITIONAL) - {task_suite}")
    print(f"   Mode Locking: ENABLED")
    print(f"   Speculative Decoding: {'ENABLED' if use_speculative_decoding else 'DISABLED'}")
    print("=" * 80)

    # Initialize experiment tracking
    exp = None
    if use_experiment_tracking:
        try:
            exp = Experiment.init(
                name=f"Phase 3 CONDITIONAL - LIBERO {task_suite}",
                description=f"Conditional RSD with Mode Locking (HSD={'ON' if use_speculative_decoding else 'OFF'})",
                config={
                    "task_suite": task_suite,
                    "num_trials": num_trials,
                    "use_speculative_decoding": use_speculative_decoding,
                    "model_type": "conditional",
                    "gpu_type": "a100",
                    "gpu_count": 1,
                }
            )
            exp.add_tags(['phase3', 'evaluation', 'libero', 'conditional', 'mode-locking', task_suite])
        except Exception as e:
            print(f"⚠️ Failed to initialize experiment: {e}")
            exp = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n📦 Device: {device}")

        # ============================================================
        # Define RFSQ Components (from Phase 1)
        # ============================================================

        class RobustSTEQuantizer(nn.Module):
            """STE Quantizer with LayerNorm strategy"""
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
                    dist = torch.abs(z_norm.unsqueeze(-1) - self.boundaries.view(1, 1, 1, -1))
                    indices = torch.argmin(dist, dim=-1)
                    z_q_norm = self.boundaries[indices]
                    z_q = z_q_norm * original_std + original_mean
                else:
                    dist = torch.abs(z.unsqueeze(-1) - self.boundaries.view(1, 1, 1, -1))
                    indices = torch.argmin(dist, dim=-1)
                    z_q = self.boundaries[indices]
                z_q_out = z + (z_q - z).detach()
                return z_q_out, indices

        class RobustRFSQBlock(nn.Module):
            """Multi-layer residual quantization block"""
            def __init__(self, num_layers=8, num_levels=7, use_layernorm=True):
                super().__init__()
                self.num_layers = num_layers
                self.num_levels = num_levels
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
                """Decode from indices [B, S, D, L] -> [B, S, D]"""
                batch_size, seq_len, dim, num_layers = indices.shape
                reconstruction = torch.zeros(batch_size, seq_len, dim, device=indices.device)
                for layer_idx in range(num_layers):
                    layer_indices = indices[:, :, :, layer_idx]
                    layer_values = self.layers[layer_idx].boundaries[layer_indices]
                    reconstruction = reconstruction + layer_values
                return reconstruction

        class ActionRFSQAE(nn.Module):
            """RFSQ AutoEncoder for actions"""
            def __init__(self, action_dim=7, hidden_dim=16, num_layers=8, num_levels=7, use_layernorm=True):
                super().__init__()
                self.action_dim = action_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.num_levels = num_levels

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
                """Decode from indices [B, C, H, L] -> [B, C, A]"""
                z_reconstructed = self.rfsq.decode_from_indices(indices)
                batch_size, chunk_len, hidden_dim = z_reconstructed.shape
                z_flat = z_reconstructed.view(-1, self.hidden_dim)
                actions_flat = self.decoder(z_flat)
                actions = actions_flat.view(batch_size, chunk_len, -1)
                return actions

        # ============================================================
        # Define Draft Model (EXACT match with Phase 2 training)
        # ============================================================

        class DraftTransformerDecoder(nn.Module):
            """Transformer Decoder for Draft Model"""
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
            """
            Draft Model - EXACT match with Phase 2 training
            Predicts coarse layers L0-L2
            """
            def __init__(self, input_dim=4096, hidden_dim=512, num_coarse_layers=3,
                         chunk_len=8, action_hidden_dim=16, grid_size=7):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.num_coarse_layers = num_coarse_layers
                self.chunk_len = chunk_len
                self.action_hidden_dim = action_hidden_dim
                self.grid_size = grid_size

                # Projection Layer (4096 -> 512)
                self.input_projection = nn.Linear(input_dim, hidden_dim)

                # Transformer Decoder
                self.decoder = DraftTransformerDecoder(hidden_dim=hidden_dim)

                # Classification Heads (one per coarse layer)
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

                # Project 4096 -> 512
                projected = self.input_projection(openvla_hidden_states)

                # Add sequence dimension
                x = projected.unsqueeze(1)

                # Transformer Decoder
                decoder_output = self.decoder(x)
                decoder_output = decoder_output.squeeze(1)

                # Classification Heads
                layer_outputs = []
                for head in self.classification_heads:
                    logits = head(decoder_output)
                    logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
                    layer_outputs.append(logits)

                return torch.stack(layer_outputs, dim=1)  # [B, 3, 128, 7]

        # ============================================================
        # Define CONDITIONAL RFSQ Head (EXACT match with Phase 2 training)
        # ============================================================

        class ConditionedRFSQHead(nn.Module):
            """
            ✨ CONDITIONAL RFSQ Head with Mode Locking ✨

            Key Innovation:
            - Accepts L0-L2 tokens as conditioning input (from Draft Model)
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

                # B. Token Embedding Layer (NEW)
                # Maps discrete tokens [0-6] to continuous embeddings [64-dim]
                self.token_embedding = nn.Embedding(grid_size, token_embed_dim)

                # C. Token Projection Layer (NEW)
                # Input: [Batch, Chunk=8, Hidden=16, Layers=3] with 64-dim embeddings
                # Total: 8 * 16 * 3 * 64 = 24,576 dimensions
                token_flat_dim = chunk_len * action_hidden_dim * condition_layers * token_embed_dim
                self.token_proj = nn.Sequential(
                    nn.Linear(token_flat_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                )

                # D. Fusion Layer (NEW)
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
                    condition_tokens: [Batch, 8, 16, 3] - L0-L2 tokens (Draft output during inference)

                Returns:
                    logits: [Batch, 8, 128, 7] - Predictions for all 8 layers
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
                # 🔥 MODE LOCKING HAPPENS HERE: token features modulate the hidden state
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
        # CONDITIONAL RSD Inference Engine
        # ============================================================

        class ConditionalRSDInferenceEngine:
            """
            ✨ Conditional Residual Speculative Decoding Inference Engine ✨

            With Mode Locking:
            1. Draft Model predicts L0-L2
            2. Main Model receives image + Draft's L0-L2 → MODE LOCKING
            3. Main Model predicts all 8 layers conditioned on Draft
            4. Verification: Check if Main's L0-L2 matches Draft's L0-L2
            """

            def __init__(
                self,
                openvla_model,
                openvla_processor,
                conditional_rfsq_head,
                rfsq_decoder,
                draft_model=None,
                device='cuda',
                use_speculative=True,
                acceptance_threshold=0.7,
            ):
                self.openvla = openvla_model
                self.processor = openvla_processor
                self.conditional_rfsq_head = conditional_rfsq_head
                self.rfsq_decoder = rfsq_decoder
                self.draft_model = draft_model
                self.device = device
                self.use_speculative = use_speculative and (draft_model is not None)
                self.acceptance_threshold = acceptance_threshold

                # RFSQ parameters
                self.chunk_len = 8
                self.action_hidden_dim = 16
                self.num_rfsq_layers = 8

                self.stats = {
                    'total_predictions': 0,
                    'draft_acceptances': 0,
                    'partial_acceptances': 0,
                    'full_rejections': 0,
                    'fallback_to_openvla': 0,
                    'mode_locking_enabled': 0,
                }

            def predict(self, image, task_description):
                """Predict action using CONDITIONAL RSD with Mode Locking."""
                import torch
                import time

                start_time = time.time()

                with torch.no_grad():
                    # Step 1: Get OpenVLA hidden states
                    hidden_states = self._get_openvla_features(image, task_description)

                    if hidden_states is None:
                        # Fallback to direct OpenVLA action
                        self.stats['fallback_to_openvla'] += 1
                        action = self._get_openvla_action(image, task_description)
                        inference_time = time.time() - start_time
                        return action, inference_time

                    # Step 2: Conditional Speculative Decoding with Mode Locking
                    if self.use_speculative and self.draft_model is not None:
                        # 2a. Draft Model predicts L0-L2
                        draft_logits = self.draft_model(hidden_states)  # [1, 3, 128, 7]
                        draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]

                        # 2b. Reshape Draft tokens to condition format [B, Chunk, Hidden, Layers]
                        # draft_tokens: [1, 3, 128] -> [1, 128, 3] -> [1, 8, 16, 3]
                        draft_tokens_reshaped = draft_tokens.permute(0, 2, 1)  # [1, 128, 3]
                        draft_condition = draft_tokens_reshaped.view(1, self.chunk_len, self.action_hidden_dim, 3)  # [1, 8, 16, 3]

                        # 2c. 🔥 CONDITIONAL Main Model with Mode Locking
                        main_logits = self.conditional_rfsq_head(hidden_states, draft_condition)  # [1, 8, 128, 7]
                        self.stats['mode_locking_enabled'] += 1

                        # 2d. Verification: Check if Main's L0-L2 matches Draft's L0-L2
                        final_logits, acceptance_info = self._accept_reject(draft_logits, main_logits)
                        self._update_stats(acceptance_info)
                    else:
                        # Baseline: Use dummy condition (zeros)
                        dummy_condition = torch.zeros(1, self.chunk_len, self.action_hidden_dim, 3,
                                                     dtype=torch.long, device=hidden_states.device)
                        final_logits = self.conditional_rfsq_head(hidden_states, dummy_condition)  # [1, 8, 128, 7]

                    # Step 3: Decode RFSQ tokens to actions
                    actions = self._decode_actions(final_logits)

                inference_time = time.time() - start_time
                self.stats['total_predictions'] += 1

                # Return first action from chunk
                if actions is not None and len(actions) > 0:
                    return actions[0], inference_time
                else:
                    # Fallback
                    self.stats['fallback_to_openvla'] += 1
                    action = self._get_openvla_action(image, task_description)
                    return action, inference_time

            def _get_openvla_features(self, image, task_description):
                """Get hidden states from OpenVLA"""
                import torch
                from PIL import Image as PILImage

                try:
                    if not isinstance(image, PILImage.Image):
                        if isinstance(image, np.ndarray):
                            image = PILImage.fromarray(image.astype(np.uint8))

                    inputs = self.processor(task_description, image).to(self.device, dtype=torch.bfloat16)

                    try:
                        outputs = self.openvla(**inputs, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()
                            if hidden_4096.shape == (1, 4096):
                                return hidden_4096
                    except Exception as e:
                        print(f"   ⚠️ Hidden states extraction failed: {e}")

                    return None

                except Exception as e:
                    print(f"   ⚠️ OpenVLA feature extraction error: {e}")
                    return None

            def _get_openvla_action(self, image, task_description):
                """Get action directly from OpenVLA (fallback)"""
                from PIL import Image as PILImage

                try:
                    if not isinstance(image, PILImage.Image):
                        if isinstance(image, np.ndarray):
                            image = PILImage.fromarray(image.astype(np.uint8))

                    inputs = self.processor(task_description, image).to(self.device, dtype=torch.bfloat16)
                    action_result = self.openvla.predict_action(**inputs, do_sample=False)
                    action = safe_extract_action(action_result)
                    return action
                except Exception as e:
                    print(f"   ⚠️ OpenVLA action error: {e}")
                    return np.zeros(7, dtype=np.float32)

            def _accept_reject(self, draft_logits, main_logits):
                """Accept/reject mechanism with verification."""
                import torch

                # Get token predictions
                draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]
                main_tokens_coarse = torch.argmax(main_logits[:, :3], dim=-1)  # [1, 3, 128]

                # Calculate agreement
                matches = (draft_tokens == main_tokens_coarse).float()
                agreement_rate = matches.mean().item()

                acceptance_info = {
                    'agreement_rate': agreement_rate,
                    'type': 'full_acceptance' if agreement_rate >= self.acceptance_threshold else 'rejection'
                }

                return main_logits, acceptance_info

            def _decode_actions(self, logits):
                """
                Decode RFSQ logits to actions.

                Args:
                    logits: [batch, num_layers, chunk*hidden, grid_size]
                           e.g., [1, 8, 128, 7]

                Returns:
                    actions: [chunk_len, action_dim] e.g., [8, 7]
                """
                import torch

                try:
                    batch_size = logits.shape[0]
                    num_layers = logits.shape[1]
                    chunk_hidden = logits.shape[2]  # 128 = chunk_len * hidden_dim = 8 * 16
                    grid_size = logits.shape[3]

                    # Get token indices: [batch, num_layers, chunk*hidden]
                    token_indices = torch.argmax(logits, dim=-1)  # [1, 8, 128]

                    # Reshape to [batch, chunk_len, hidden_dim, num_layers]
                    # Transpose to [1, 128, 8]
                    token_indices = token_indices.permute(0, 2, 1)  # [1, 128, 8]

                    # Reshape to [1, 8, 16, 8]
                    token_indices = token_indices.view(batch_size, self.chunk_len, self.action_hidden_dim, num_layers)

                    # Use RFSQ decoder to get actions
                    actions = self.rfsq_decoder.decode_from_indices(token_indices)  # [1, 8, 7]

                    return actions[0].cpu().numpy()  # [8, 7]

                except Exception as e:
                    print(f"   ⚠️ Action decoding error: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

            def _update_stats(self, acceptance_info):
                """Update acceptance statistics"""
                if acceptance_info['type'] == 'full_acceptance':
                    self.stats['draft_acceptances'] += 1
                elif acceptance_info['type'] == 'partial_acceptance':
                    self.stats['partial_acceptances'] += 1
                else:
                    self.stats['full_rejections'] += 1

            def get_stats(self):
                """Get inference statistics"""
                total = self.stats['total_predictions']
                if total == 0:
                    return {}

                return {
                    'total_predictions': total,
                    'draft_acceptance_rate': self.stats['draft_acceptances'] / total,
                    'partial_acceptance_rate': self.stats['partial_acceptances'] / total,
                    'full_rejection_rate': self.stats['full_rejections'] / total,
                    'fallback_rate': self.stats['fallback_to_openvla'] / total,
                    'mode_locking_rate': self.stats['mode_locking_enabled'] / total,
                }

        # ============================================================
        # Load Models
        # ============================================================
        print("\n📦 Loading models...")

        # 1. Load RFSQ Decoder (Robust version from Phase 1)
        rfsq_decoder = ActionRFSQAE(
            action_dim=7,
            hidden_dim=16,
            num_layers=8,
            num_levels=7,
            use_layernorm=True,
        ).to(device)

        rfsq_decoder_path = "/models/rfsq_robust_best.pt"
        if Path(rfsq_decoder_path).exists():
            checkpoint = torch.load(rfsq_decoder_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            rfsq_decoder.load_state_dict(state_dict)
            print(f"   ✓ RFSQ Decoder loaded from {rfsq_decoder_path}")
        else:
            print(f"   ⚠️ RFSQ Decoder not found at {rfsq_decoder_path}, using random init")

        rfsq_decoder.eval()

        # 2. Load OpenVLA Base Model
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print("\n   Loading OpenVLA base model...")
        openvla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir="/hf_cache",
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True,
            cache_dir="/hf_cache",
        )

        openvla.eval()
        print(f"   ✓ OpenVLA base loaded")

        # 3. ✨ Load CONDITIONAL RFSQ Head (from Phase 2 conditional training)
        print("\n   ✨ Loading CONDITIONAL RFSQ Head...")
        conditional_rfsq_head = ConditionedRFSQHead(
            input_dim=4096,
            hidden_dim=1024,
            num_layers=8,
            chunk_len=8,
            action_hidden_dim=16,
            grid_size=7,
            condition_layers=3,      # NEW: L0-L2 conditioning
            token_embed_dim=64,      # NEW: Token embedding dimension
        ).to(device)

        # ✅ Load from CONDITIONAL checkpoint
        conditional_rfsq_head_path = "/models/openvla_rfsq_conditional/best_rfsq_head.pt"
        if Path(conditional_rfsq_head_path).exists():
            checkpoint = torch.load(conditional_rfsq_head_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            try:
                conditional_rfsq_head.load_state_dict(state_dict)
                acc = checkpoint.get('val_accuracy', checkpoint.get('best_accuracy', 'unknown'))
                print(f"   ✓ CONDITIONAL RFSQ Head loaded (accuracy: {acc})")
                print(f"   ✓ Mode Locking: ENABLED")
            except Exception as e:
                print(f"   ⚠️ CONDITIONAL RFSQ Head loading failed: {e}")
                print(f"   ⚠️ Using random init for CONDITIONAL RFSQ Head")
        else:
            print(f"   ⚠️ CONDITIONAL RFSQ Head not found at {conditional_rfsq_head_path}")
            print(f"   ⚠️ Using random init (Mode Locking will still work, but accuracy will be poor)")

        conditional_rfsq_head.eval()

        # 4. Load Draft Model (if speculative decoding enabled)
        draft_model = None
        if use_speculative_decoding:
            draft_model = RFSQDraftModelWithProjection(
                input_dim=4096,
                hidden_dim=512,
                num_coarse_layers=3,
                chunk_len=8,
                action_hidden_dim=16,
                grid_size=7,
            ).to(device)

            draft_model_path = "/models/best_draft_with_projection.pt"
            if Path(draft_model_path).exists():
                checkpoint = torch.load(draft_model_path, map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
                try:
                    draft_model.load_state_dict(state_dict)
                    acc = checkpoint.get('val_accuracy', checkpoint.get('best_accuracy', 'unknown'))
                    print(f"   ✓ Draft Model loaded (accuracy: {acc})")
                except Exception as e:
                    print(f"   ⚠️ Draft Model state dict mismatch: {e}")
                    print(f"   ⚠️ Using random init for Draft Model")
            else:
                print(f"   ⚠️ Draft Model not found at {draft_model_path}, using random init")

            draft_model.eval()

        # ============================================================
        # Initialize CONDITIONAL RSD Engine
        # ============================================================
        print("\n🤖 Initializing CONDITIONAL RSD Inference Engine...")

        rsd_engine = ConditionalRSDInferenceEngine(
            openvla_model=openvla,
            openvla_processor=processor,
            conditional_rfsq_head=conditional_rfsq_head,
            rfsq_decoder=rfsq_decoder,
            draft_model=draft_model,
            device=device,
            use_speculative=use_speculative_decoding,
        )

        print(f"   ✓ CONDITIONAL RSD Engine initialized")
        print(f"   ✓ Speculative Decoding: {use_speculative_decoding}")
        print(f"   ✓ Mode Locking: ENABLED")

        # ============================================================
        # Initialize LIBERO
        # ============================================================
        print(f"\n🏗️ Initializing LIBERO {task_suite}...")

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_obj = benchmark_dict[task_suite]()
        num_tasks = task_suite_obj.n_tasks

        print(f"   Number of tasks: {num_tasks}")

        # ============================================================
        # Run Evaluation
        # ============================================================
        print(f"\n🎯 Starting evaluation ({num_trials} trials per task)...\n")

        total_episodes = 0
        total_successes = 0
        total_inference_time = 0.0
        task_results = []

        for task_id in range(num_tasks):
            task = task_suite_obj.get_task(task_id)
            task_description = task.language

            print(f"\n{'='*80}")
            print(f"Task {task_id + 1}/{num_tasks}: {task_description}")
            print(f"{'='*80}")

            init_states = task_suite_obj.get_task_init_states(task_id)

            task_successes = 0
            task_episodes = 0

            for trial_idx in range(min(num_trials, len(init_states))):
                task_episodes += 1
                total_episodes += 1

                # Create environment
                bddl_file_path = os.path.join(
                    "/root/LIBERO/libero/libero/bddl_files",
                    task.problem_folder,
                    task.bddl_file
                )

                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_file_path,
                    camera_heights=256,
                    camera_widths=256,
                )
                env.reset()
                obs = env.set_init_state(init_states[trial_idx])

                # Run episode
                episode_success = False
                episode_inference_times = []

                for step in range(300):
                    try:
                        # Get image
                        image = PILImage.fromarray(obs['agentview_image'].astype(np.uint8))

                        # CONDITIONAL RSD prediction with Mode Locking
                        action, inference_time = rsd_engine.predict(image, task_description)

                        if action is None:
                            action = np.zeros(7, dtype=np.float32)

                        episode_inference_times.append(inference_time)

                        # Step environment
                        obs, reward, done, info = env.step(action)

                        if done:
                            episode_success = True
                            break

                    except Exception as step_error:
                        print(f"      ⚠️ Step error: {step_error}")
                        break

                env.close()

                # Record results
                if episode_success:
                    task_successes += 1
                    total_successes += 1

                avg_episode_time = np.mean(episode_inference_times) if episode_inference_times else 0.0
                total_inference_time += avg_episode_time

                print(f"   Trial {trial_idx + 1}: "
                      f"{'✓' if episode_success else '✗'} "
                      f"({avg_episode_time*1000:.1f}ms avg)")

            # Task summary
            task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
            task_results.append({
                'task_id': task_id,
                'task_description': task_description,
                'successes': task_successes,
                'episodes': task_episodes,
                'success_rate': task_success_rate,
            })

            print(f"\n   Task Success Rate: {task_success_rate:.1%} ({task_successes}/{task_episodes})")

            if exp:
                try:
                    exp.log({"task_success_rate": task_success_rate}, step=task_id)
                except:
                    pass

        # ============================================================
        # Final Results
        # ============================================================
        final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
        avg_inference_time = total_inference_time / total_episodes if total_episodes > 0 else 0.0

        rsd_stats = rsd_engine.get_stats()

        results = {
            'task_suite': task_suite,
            'use_speculative_decoding': use_speculative_decoding,
            'model_type': 'conditional_with_mode_locking',
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'final_success_rate': final_success_rate,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'task_results': task_results,
            'rsd_stats': rsd_stats,
        }

        print(f"\n{'='*80}")
        print(f"🎉 CONDITIONAL EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"   Success Rate: {final_success_rate:.1%}")
        print(f"   Avg Inference Time: {avg_inference_time*1000:.1f} ms")
        if use_speculative_decoding and rsd_stats:
            print(f"   Draft Acceptance Rate: {rsd_stats.get('draft_acceptance_rate', 0):.1%}")
            print(f"   Mode Locking Rate: {rsd_stats.get('mode_locking_rate', 0):.1%}")
            print(f"   Fallback Rate: {rsd_stats.get('fallback_rate', 0):.1%}")
        print(f"{'='*80}")

        # Log final metrics
        if exp:
            try:
                exp.log({"final_success_rate": final_success_rate})
                exp.log({"avg_inference_time_ms": avg_inference_time * 1000})
                if rsd_stats:
                    exp.log({"draft_acceptance_rate": rsd_stats.get('draft_acceptance_rate', 0)})
                    exp.log({"mode_locking_rate": rsd_stats.get('mode_locking_rate', 0)})
                exp.finish(status="completed")
            except:
                pass

        # Save results
        results_path = f"/results/{task_suite}_conditional_{'rsd' if use_speculative_decoding else 'baseline'}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")
        results_volume.commit()

        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        if exp:
            try:
                exp.finish(status="failed")
            except:
                pass
        raise


@app.local_entrypoint()
def main(
    task_suite: str = "libero_spatial",
    num_trials: int = 50,
    use_speculative_decoding: bool = True,
):
    """Main entry point for CONDITIONAL evaluation."""
    print(f"🚀 Starting Phase 3 CONDITIONAL LIBERO Evaluation...")
    print(f"   Task Suite: {task_suite}")
    print(f"   Num Trials: {num_trials}")
    print(f"   Speculative Decoding: {use_speculative_decoding}")
    print(f"   ✨ Mode Locking: ENABLED")

    results = run_libero_evaluation.remote(
        task_suite=task_suite,
        num_trials=num_trials,
        use_speculative_decoding=use_speculative_decoding,
    )

    print(f"\n✅ CONDITIONAL Evaluation complete!")
    print(f"   Success Rate: {results['final_success_rate']:.1%}")
    print(f"   Avg Inference Time: {results['avg_inference_time_ms']:.1f} ms")

    return results
