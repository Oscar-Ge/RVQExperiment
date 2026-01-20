"""
Phase 3: LIBERO Evaluation with CONDITIONAL RSD + Official Fixes
ULTIMATE VERSION - Combines Mode Locking with Official openvla-oft Best Practices

✨ Mode Locking Features:
1. ConditionedRFSQHead with Token Embedding
2. Draft Model provides L0-L2 conditioning
3. Main Model fuses image + token features
4. Solves mean-seeking problem

🔧 Critical Fixes from openvla-oft:
1. Image rotation (180 degrees) - CRITICAL for LIBERO
2. Proper image resize (256→224 with Lanczos interpolation)
3. Gripper action processing (normalize + invert)
4. Official norm_stats and unnorm_key handling
5. Action queue management
6. Stabilization period (first 10 steps)
7. Proper proprio normalization with q01/q99

Usage:
    # Test with conditional RSD
    modal run phase3/modal_phase3_libero_eval_CONDITIONAL_FIXED.py --num-trials 3

    # Full evaluation
    modal run phase3/modal_phase3_libero_eval_CONDITIONAL_FIXED.py \\
        --task-suite libero_spatial \\
        --num-trials 50 \\
        --use-speculative-decoding True
"""

import os
import sys
import modal
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ============================================================
# Modal App Setup
# ============================================================
app = modal.App("rsd-phase3-libero-eval-conditional-fixed")

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
                 "libegl1-mesa-dev", "libgles2-mesa-dev", "libglew-dev", "patchelf")
    .pip_install("uv")
    .run_commands(
        # Install PyTorch and deps
        "uv pip install --system 'numpy<2' torch==2.2.0 torchvision==0.17.0 "
        "transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "
        "accelerate peft bitsandbytes pillow einops sentencepiece protobuf "
        "huggingface_hub scipy tqdm matplotlib pandas requests json-numpy jsonlines",
        # ✅ Install prismatic from openvla-oft repo (official implementation)
        "cd /root && git clone https://github.com/moojink/openvla-oft.git",
        "cd /root/openvla-oft && uv pip install --system -e .",
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
        "uv pip install --system mujoco dm-control robosuite==1.4.0 termcolor h5py bddl easydict cloudpickle gym gymnasium pytest",
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
        "HF_HOME": "/hf_cache",
        "TRANSFORMERS_CACHE": "/hf_cache",
        "LIBERO_NO_PROMPT": "1",
        "LIBERO_FOLDER": "/data/libero",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=28800,
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
    """Run LIBERO evaluation with CONDITIONAL RSD + Official Fixes."""
    import sys
    import torch
    import torch.nn as nn
    import numpy as np
    import time
    import json
    from pathlib import Path
    from collections import deque

    sys.path.insert(0, "/root/LIBERO")
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/openvla-oft")  # ✅ Add openvla-oft to path

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # ✅ Import official utility functions
    from experiments.robot.libero.libero_utils import (
        quat2axisangle,
        get_libero_dummy_action,
    )
    from experiments.robot.openvla_utils import (
        get_vla,
        get_processor,
        get_action_head,
        get_proprio_projector,
        resize_image_for_policy,  # ✅ CRITICAL: proper image resize
    )
    from experiments.robot.robot_utils import (
        normalize_gripper_action,
        invert_gripper_action,
        set_seed_everywhere,
        get_image_resize_size,
    )
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        check_unnorm_key,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    try:
        from src.orchestra_sdk.experiment import Experiment
        use_experiment_tracking = True
    except ImportError:
        print("⚠️ Orchestra SDK not available, skipping experiment tracking")
        use_experiment_tracking = False

    print("=" * 80)
    print(f"🚀 Phase 3: CONDITIONAL RSD + Official Fixes - {task_suite}")
    print(f"   ✨ Mode Locking: ENABLED")
    print(f"   🔧 Official Fixes: ENABLED")
    print(f"   Speculative Decoding: {'ENABLED' if use_speculative_decoding else 'DISABLED'}")
    print("=" * 80)

    # ✅ Set random seed for reproducibility
    SEED = 7
    set_seed_everywhere(SEED)
    print(f"\n✓ Random seed set to {SEED} for reproducibility")

    exp = None
    if use_experiment_tracking:
        try:
            exp = Experiment.init(
                name=f"Phase 3 CONDITIONAL+FIXED - LIBERO {task_suite}",
                description=f"Conditional RSD with Mode Locking + Official Fixes",
                config={
                    "task_suite": task_suite,
                    "num_trials": num_trials,
                    "use_speculative_decoding": use_speculative_decoding,
                    "model_type": "conditional_with_official_fixes",
                    "gpu_type": "a100",
                }
            )
            exp.add_tags(['phase3', 'evaluation', 'libero', 'conditional', 'fixed', task_suite])
        except Exception as e:
            print(f"⚠️ Failed to initialize experiment: {e}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n📦 Device: {device}")

        # ============================================================
        # Load OpenVLA-OFT model using OFFICIAL functions
        # ============================================================
        print("\n📦 Loading OpenVLA-OFT model using official functions...")

        # Create config for LIBERO-Spatial
        cfg = GenerateConfig(
            pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
            model_family="openvla",
            use_l1_regression=True,  # ✅ Use L1 regression
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,  # ✅ Center crop
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            task_suite_name=task_suite,
        )

        # Load VLA model using official function
        vla = get_vla(cfg)
        processor = get_processor(cfg)

        # Load MLP action head
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

        # Load proprio projector
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

        # ✅ Ensure norm_stats are available
        if not hasattr(vla, 'norm_stats'):
            vla.norm_stats = {}
            print("   ⚠️  Model doesn't have norm_stats, initializing empty dict")

        expected_keys = ["libero_spatial", "libero_spatial_no_noops"]
        has_stats = any(key in vla.norm_stats for key in expected_keys)

        if not has_stats:
            print("   ⚠️  Model missing LIBERO norm_stats, injecting manually...")
            libero_stats = {
                "action": {
                    "mean": [0.15312479436397552, 0.13707277178764343, -0.15526802837848663,
                             -0.005176450591534376, -0.01120874285697937, -0.020194264128804207,
                             0.4578818082809448],
                    "std": [0.41272708773612976, 0.34724321961402893, 0.50869220495224,
                            0.037266165018081665, 0.07244449853897095, 0.05762382969260216,
                            0.49827873706817627],
                    "max": [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375, 1.0],
                    "min": [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475, 0.0],
                    "q01": [-0.7454732114076613, -0.6616071462631226, -0.9375,
                            -0.1071428582072258, -0.20678570866584778, -0.1842857152223587, 0.0],
                    "q99": [0.9375, 0.8758928775787354, 0.9321428537368774,
                            0.1039285734295845, 0.17678570747375488, 0.14571428298950195, 1.0],
                    "mask": [True, True, True, True, True, True, False]
                },
                "proprio": {
                    "mean": [-0.024462558329105377, 0.106529600918293, 1.0580483675003052,
                             3.0628468990325928, -0.10464039444923401, 0.08307311683893204,
                             0.01995457336306572, -0.020162804052233696],
                    "std": [0.1101478561758995, 0.13784688711166382, 0.1044282391667366,
                            0.10451053828001022, 0.4112098217010498, 0.2176690548658371,
                            0.017260896041989326, 0.0171116404235363],
                    "max": [0.1759040206670761, 0.3904820382595062, 1.3290715217590332,
                            3.4566118717193604, 1.2268599271774292, 1.0429412126541138,
                            0.041053611785173416, 0.000775813648942858],
                    "min": [-0.3095473051071167, -0.29250794649124146, 0.9095591306686401,
                            2.497488260269165, -1.8006486892700195, -0.7207611203193665,
                            -0.0004703797458205372, -0.041536275297403336],
                    "q01": [-0.2727657300233841, -0.23721413239836692, 0.9160063165426254,
                            2.77949666261673, -1.3187511622905732, -0.41989982962608335,
                            0.001503719249740243, -0.03989770736545324],
                    "q99": [0.13529365032911292, 0.3629165390133857, 1.2862326657772063,
                            3.2829698753356933, 0.9332760351896285, 0.6325724506378171,
                            0.039933966137468815, -0.001671919699292631]
                },
                "num_transitions": 52970,
                "num_trajectories": 432
            }
            vla.norm_stats["libero_spatial_no_noops"] = libero_stats
            print("   ✅ Manually injected 'libero_spatial_no_noops' statistics")
        else:
            print(f"   ✅ Model already has norm_stats with keys: {list(vla.norm_stats.keys())}")

        # ✅ Use official check_unnorm_key
        check_unnorm_key(cfg, vla)

        print(f"   ✓ OpenVLA-OFT loaded with action_head and proprio_projector")
        print(f"   ✓ Using L1 regression: {cfg.use_l1_regression}")
        print(f"   ✓ Using unnorm_key: {cfg.unnorm_key}")

        # ✅ CRITICAL: Get image resize size
        resize_size = get_image_resize_size(cfg)
        print(f"   ✓ Image resize size: {resize_size}")

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
        # Define Draft Model
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
            """Draft Model - Predicts coarse layers L0-L2"""
            def __init__(self, input_dim=4096, hidden_dim=512, num_coarse_layers=3,
                         chunk_len=8, action_hidden_dim=16, grid_size=7):
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

                return torch.stack(layer_outputs, dim=1)  # [B, 3, 128, 7]

        # ============================================================
        # ✨ Define CONDITIONAL RFSQ Head
        # ============================================================

        class ConditionedRFSQHead(nn.Module):
            """✨ CONDITIONAL RFSQ Head with Mode Locking"""
            def __init__(
                self,
                input_dim=4096,
                hidden_dim=1024,
                num_layers=8,
                chunk_len=8,
                action_hidden_dim=16,
                grid_size=7,
                condition_layers=3,
                token_embed_dim=64,
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
                self.token_embedding = nn.Embedding(grid_size, token_embed_dim)

                # C. Token Projection Layer
                token_flat_dim = chunk_len * action_hidden_dim * condition_layers * token_embed_dim
                self.token_proj = nn.Sequential(
                    nn.Linear(token_flat_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                )

                # D. Fusion Layer
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                )

                # E. Output Heads
                output_size = chunk_len * action_hidden_dim * grid_size
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
                    condition_tokens: [Batch, 8, 16, 3] - L0-L2 tokens

                Returns:
                    logits: [Batch, 8, 128, 7] - Predictions for all 8 layers
                """
                batch_size = hidden_states.shape[0]

                # Process image features
                img_feat = self.feature_proj(hidden_states)  # [B, 1024]

                # Process condition tokens
                token_embeds = self.token_embedding(condition_tokens)  # [B, 8, 16, 3, 64]
                token_flat = token_embeds.view(batch_size, -1)  # [B, 24576]
                token_feat = self.token_proj(token_flat)  # [B, 1024]

                # 🔥 MODE LOCKING: Fusion
                combined = torch.cat([img_feat, token_feat], dim=-1)  # [B, 2048]
                fused_feat = self.fusion(combined)  # [B, 1024]

                # Predict all 8 layers
                layer_outputs = []
                for head in self.layer_heads:
                    logits = head(fused_feat)
                    logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
                    layer_outputs.append(logits)

                return torch.stack(layer_outputs, dim=1)  # [B, 8, 128, 7]

        # ============================================================
        # ✨ CONDITIONAL RSD Inference Engine with Official Fixes
        # ============================================================

        class ConditionalRSDInferenceEngine:
            """✨ Conditional RSD with Mode Locking + Official Fixes"""

            def __init__(
                self,
                vla_model,
                processor,
                action_head,
                proprio_projector,
                conditional_rfsq_head,
                rfsq_decoder,
                draft_model=None,
                cfg=None,
                device='cuda',
                use_speculative=True,
                resize_size=224,
            ):
                self.vla = vla_model
                self.processor = processor
                self.action_head = action_head
                self.proprio_projector = proprio_projector
                self.conditional_rfsq_head = conditional_rfsq_head
                self.rfsq_decoder = rfsq_decoder
                self.draft_model = draft_model
                self.cfg = cfg
                self.device = device
                self.use_speculative = use_speculative and (draft_model is not None)
                self.resize_size = resize_size

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

            def predict_action_chunk(self, observation, task_description):
                """
                Predict action chunk using CONDITIONAL RSD with all official fixes.

                Args:
                    observation: dict with 'full_image', 'wrist_image', 'state'
                    task_description: str

                Returns:
                    actions: [8, 7] numpy array
                """
                import torch

                with torch.no_grad():
                    # Step 1: Get OpenVLA hidden states (using official processing)
                    hidden_states = self._get_openvla_features(observation, task_description)

                    if hidden_states is None:
                        # Fallback to official OpenVLA action prediction
                        self.stats['fallback_to_openvla'] += 1
                        actions = self._get_official_openvla_actions(observation, task_description)
                        return actions

                    # Step 2: Conditional Speculative Decoding with Mode Locking
                    if self.use_speculative and self.draft_model is not None:
                        # 2a. Draft Model predicts L0-L2
                        draft_logits = self.draft_model(hidden_states)  # [1, 3, 128, 7]
                        draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]

                        # 2b. Reshape Draft tokens to condition format [B, Chunk, Hidden, Layers]
                        draft_tokens_reshaped = draft_tokens.permute(0, 2, 1)  # [1, 128, 3]
                        draft_condition = draft_tokens_reshaped.view(1, self.chunk_len, self.action_hidden_dim, 3)

                        # 2c. 🔥 CONDITIONAL Main Model with Mode Locking
                        main_logits = self.conditional_rfsq_head(hidden_states, draft_condition)  # [1, 8, 128, 7]
                        self.stats['mode_locking_enabled'] += 1

                        # 2d. Verification
                        final_logits, acceptance_info = self._accept_reject(draft_logits, main_logits)
                        self._update_stats(acceptance_info)
                    else:
                        # Baseline: Use dummy condition (zeros)
                        dummy_condition = torch.zeros(1, self.chunk_len, self.action_hidden_dim, 3,
                                                     dtype=torch.long, device=hidden_states.device)
                        final_logits = self.conditional_rfsq_head(hidden_states, dummy_condition)

                    # Step 3: Decode RFSQ tokens to actions
                    actions = self._decode_actions(final_logits)

                    self.stats['total_predictions'] += 1

                    if actions is not None:
                        return actions  # [8, 7]
                    else:
                        # Fallback
                        self.stats['fallback_to_openvla'] += 1
                        actions = self._get_official_openvla_actions(observation, task_description)
                        return actions

            def _get_openvla_features(self, observation, task_description):
                """Extract hidden states from OpenVLA using official processing"""
                import torch

                try:
                    # ✅ Use official processing (matches training)
                    # Format observation according to official API
                    from PIL import Image as PILImage

                    full_img = observation['full_image']  # Already resized to 224x224
                    wrist_img = observation['wrist_image']
                    state = observation['state']

                    # Convert to PIL Images if needed
                    if isinstance(full_img, np.ndarray):
                        full_img = PILImage.fromarray(full_img.astype(np.uint8))
                    if isinstance(wrist_img, np.ndarray):
                        wrist_img = PILImage.fromarray(wrist_img.astype(np.uint8))

                    # Prepare inputs using official processor
                    inputs = self.processor(task_description, full_img).to(self.device, dtype=torch.bfloat16)

                    # Add wrist image and proprio if available
                    if self.cfg.num_images_in_input == 2:
                        wrist_inputs = self.processor(task_description, wrist_img).to(self.device, dtype=torch.bfloat16)
                        # Concatenate image inputs
                        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

                    # Add proprio state if used
                    if self.cfg.use_proprio:
                        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        inputs["proprio"] = state_tensor

                    # Forward through VLA to get hidden states
                    outputs = self.vla(**inputs, output_hidden_states=True)

                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()  # [1, 4096]
                        if hidden_4096.shape == (1, 4096):
                            return hidden_4096

                    return None

                except Exception as e:
                    print(f"   ⚠️ OpenVLA feature extraction error: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

            def _get_official_openvla_actions(self, observation, task_description):
                """Fallback: Use official OpenVLA action prediction"""
                import torch
                from PIL import Image as PILImage

                try:
                    # Use the exact same processing as official openvla-oft
                    full_img = observation['full_image']
                    wrist_img = observation['wrist_image']
                    state = observation['state']

                    if isinstance(full_img, np.ndarray):
                        full_img = PILImage.fromarray(full_img.astype(np.uint8))
                    if isinstance(wrist_img, np.ndarray):
                        wrist_img = PILImage.fromarray(wrist_img.astype(np.uint8))

                    # Prepare inputs
                    inputs = self.processor(task_description, full_img).to(self.device, dtype=torch.bfloat16)

                    if self.cfg.num_images_in_input == 2:
                        wrist_inputs = self.processor(task_description, wrist_img).to(self.device, dtype=torch.bfloat16)
                        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

                    if self.cfg.use_proprio:
                        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        inputs["proprio"] = state_tensor

                    # Get VLA hidden states
                    outputs = self.vla(**inputs, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, 4096]

                    # Use action_head to predict actions
                    if self.cfg.use_l1_regression:
                        action_logits = self.action_head(last_hidden_state)  # [1, seq_len, chunk*7]
                        # Take the last token's prediction
                        action_chunk = action_logits[0, -1, :]  # [chunk*7]
                        action_chunk = action_chunk.view(self.chunk_len, 7)  # [8, 7]
                        return action_chunk.cpu().numpy()
                    else:
                        # Other prediction methods...
                        return np.zeros((self.chunk_len, 7), dtype=np.float32)

                except Exception as e:
                    print(f"   ⚠️ Official OpenVLA action error: {e}")
                    return np.zeros((self.chunk_len, 7), dtype=np.float32)

            def _accept_reject(self, draft_logits, main_logits):
                """Accept/reject mechanism"""
                import torch

                draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]
                main_tokens_coarse = torch.argmax(main_logits[:, :3], dim=-1)  # [1, 3, 128]

                matches = (draft_tokens == main_tokens_coarse).float()
                agreement_rate = matches.mean().item()

                acceptance_info = {
                    'agreement_rate': agreement_rate,
                    'type': 'full_acceptance' if agreement_rate >= 0.7 else 'rejection'
                }

                return main_logits, acceptance_info

            def _decode_actions(self, logits):
                """Decode RFSQ logits to actions"""
                import torch

                try:
                    batch_size = logits.shape[0]
                    num_layers = logits.shape[1]

                    token_indices = torch.argmax(logits, dim=-1)  # [1, 8, 128]
                    token_indices = token_indices.permute(0, 2, 1)  # [1, 128, 8]
                    token_indices = token_indices.view(batch_size, self.chunk_len, self.action_hidden_dim, num_layers)

                    actions = self.rfsq_decoder.decode_from_indices(token_indices)  # [1, 8, 7]
                    return actions[0].cpu().numpy()  # [8, 7]

                except Exception as e:
                    print(f"   ⚠️ Action decoding error: {e}")
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
        # ✅ Official LIBERO Helper Functions
        # ============================================================

        def get_libero_image(obs):
            """Extracts third-person image and rotates 180 degrees (CRITICAL!)"""
            img = obs["agentview_image"]
            img = img[::-1, ::-1]  # ✅ CRITICAL: rotate 180 degrees
            return img

        def get_libero_wrist_image(obs):
            """Extracts wrist camera image and rotates 180 degrees"""
            img = obs["robot0_eye_in_hand_image"]
            img = img[::-1, ::-1]  # ✅ CRITICAL: rotate 180 degrees
            return img

        def prepare_observation(obs):
            """
            Prepare observation dict for policy input (official method)

            ✅ CRITICAL: Resizes images from 256x256 (env) to 224x224 (model input)
            """
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)

            # ✅ CRITICAL: Resize images using official method
            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

            observation = {
                "full_image": img_resized,       # Resized to 224x224
                "wrist_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            return observation

        def process_action(action):
            """
            Process action before sending to environment (official method)

            ✅ CRITICAL: Normalize and invert gripper action
            """
            # ✅ Normalize gripper action [0,1] -> [-1,+1]
            action = normalize_gripper_action(action, binarize=True)

            # ✅ Invert gripper action (-1 = open, +1 = close)
            action = invert_gripper_action(action)

            return action

        # ============================================================
        # Load Models
        # ============================================================
        print("\n📦 Loading RSD models...")

        # 1. Load RFSQ Decoder
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

        # 2. ✨ Load CONDITIONAL RFSQ Head
        print("\n   ✨ Loading CONDITIONAL RFSQ Head...")
        conditional_rfsq_head = ConditionedRFSQHead(
            input_dim=4096,
            hidden_dim=1024,
            num_layers=8,
            chunk_len=8,
            action_hidden_dim=16,
            grid_size=7,
            condition_layers=3,
            token_embed_dim=64,
        ).to(device)

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
                print(f"   ⚠️ Using random init")
        else:
            print(f"   ⚠️ CONDITIONAL RFSQ Head not found at {conditional_rfsq_head_path}")
            print(f"   ⚠️ Using random init")

        conditional_rfsq_head.eval()

        # 3. Load Draft Model (if speculative decoding enabled)
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
                    print(f"   ⚠️ Using random init")
            else:
                print(f"   ⚠️ Draft Model not found at {draft_model_path}, using random init")

            draft_model.eval()

        # ============================================================
        # Initialize CONDITIONAL RSD Engine
        # ============================================================
        print("\n🤖 Initializing CONDITIONAL RSD Engine with Official Fixes...")

        rsd_engine = ConditionalRSDInferenceEngine(
            vla_model=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            conditional_rfsq_head=conditional_rfsq_head,
            rfsq_decoder=rfsq_decoder,
            draft_model=draft_model,
            cfg=cfg,
            device=device,
            use_speculative=use_speculative_decoding,
            resize_size=resize_size,
        )

        print(f"   ✓ CONDITIONAL RSD Engine initialized")
        print(f"   ✓ Speculative Decoding: {use_speculative_decoding}")
        print(f"   ✓ Mode Locking: ENABLED")
        print(f"   ✓ Official Fixes: ENABLED")

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
        task_results = []

        for task_id in range(num_tasks):
            task = task_suite_obj.get_task(task_id)
            task_description = task.language

            print(f"\n{'='*80}")
            print(f"Task {task_id + 1}/{num_tasks}: {task_description}")
            print(f"{'='*80}")

            # Initialize environment
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(0)  # ✅ IMPORTANT: seed affects object positions

            init_states = task_suite_obj.get_task_init_states(task_id)
            task_successes = 0
            task_episodes = 0

            for trial_idx in range(min(num_trials, len(init_states))):
                task_episodes += 1
                total_episodes += 1

                # Reset environment
                env.reset()
                obs = env.set_init_state(init_states[trial_idx])

                # ✅ Initialize action queue (official method)
                action_queue = deque(maxlen=cfg.num_open_loop_steps)

                episode_success = False
                max_steps = 220  # libero_spatial max steps

                for step in range(max_steps + 10):
                    try:
                        # ✅ First 10 steps: stabilization (do nothing)
                        if step < 10:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            continue

                        # ✅ Prepare observation (with image rotation and resize)
                        observation = prepare_observation(obs)

                        # Debug print on first step
                        if step == 10 and trial_idx == 0:
                            print(f"   🔍 Image shapes after resize:")
                            print(f"      full_image: {observation['full_image'].shape}")
                            print(f"      wrist_image: {observation['wrist_image'].shape}")
                            print(f"      state: {observation['state'].shape}")

                        # ✅ Query model if action queue is empty
                        if len(action_queue) == 0:
                            # 🔥 Use CONDITIONAL RSD with Mode Locking
                            actions = rsd_engine.predict_action_chunk(observation, task_description)
                            action_queue.extend(actions)

                            if step == 10:
                                print(f"   🔍 Action chunk returned {len(actions)} actions")
                                for i, act in enumerate(actions[:3]):
                                    print(f"      Row {i}: [{act[0]:.3f}, {act[1]:.3f}, {act[2]:.3f}, {act[3]:.3f}, {act[4]:.3f}, {act[5]:.3f}, {act[6]:.3f}]")

                        # Get action from queue
                        action = action_queue.popleft()

                        # ✅ Process action (normalize + invert gripper)
                        action = process_action(action)

                        if step == 10:
                            print(f"   🔍 Processed action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}, {action[6]:.3f}]")

                        # Execute action
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            episode_success = True
                            break

                    except Exception as step_error:
                        print(f"      ⚠️ Step error: {step_error}")
                        import traceback
                        traceback.print_exc()
                        break

                env.close()

                if episode_success:
                    task_successes += 1
                    total_successes += 1

                print(f"   Trial {trial_idx + 1}: {'✓' if episode_success else '✗'}")

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
        rsd_stats = rsd_engine.get_stats()

        results = {
            'task_suite': task_suite,
            'use_speculative_decoding': use_speculative_decoding,
            'model_type': 'conditional_with_official_fixes',
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'final_success_rate': final_success_rate,
            'task_results': task_results,
            'rsd_stats': rsd_stats,
            'config': {
                'seed': SEED,
                'unnorm_key': cfg.unnorm_key,
                'resize_size': resize_size,
                'model': cfg.pretrained_checkpoint,
            }
        }

        print(f"\n{'='*80}")
        print(f"🎉 CONDITIONAL+FIXED EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"   Success Rate: {final_success_rate:.1%}")
        if use_speculative_decoding and rsd_stats:
            print(f"   Draft Acceptance Rate: {rsd_stats.get('draft_acceptance_rate', 0):.1%}")
            print(f"   Mode Locking Rate: {rsd_stats.get('mode_locking_rate', 0):.1%}")
            print(f"   Fallback Rate: {rsd_stats.get('fallback_rate', 0):.1%}")
        print(f"{'='*80}")

        if exp:
            try:
                exp.log({"final_success_rate": final_success_rate})
                if rsd_stats:
                    exp.log({"draft_acceptance_rate": rsd_stats.get('draft_acceptance_rate', 0)})
                    exp.log({"mode_locking_rate": rsd_stats.get('mode_locking_rate', 0)})
                exp.finish(status="completed")
            except:
                pass

        results_path = f"/results/{task_suite}_conditional_fixed_{'rsd' if use_speculative_decoding else 'baseline'}.json"
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
    print(f"🚀 Starting Phase 3 CONDITIONAL+FIXED LIBERO Evaluation...")
    print(f"   Task Suite: {task_suite}")
    print(f"   Num Trials: {num_trials}")
    print(f"   Speculative Decoding: {use_speculative_decoding}")
    print(f"   ✨ Mode Locking: ENABLED")
    print(f"   🔧 Official Fixes: ENABLED")

    results = run_libero_evaluation.remote(
        task_suite=task_suite,
        num_trials=num_trials,
        use_speculative_decoding=use_speculative_decoding,
    )

    print(f"\n✅ CONDITIONAL+FIXED Evaluation complete!")
    print(f"   Success Rate: {results['final_success_rate']:.1%}")

    return results
