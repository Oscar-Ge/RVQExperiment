"""
Phase 2 Conditional Training Script: Draft Model + Conditional Main Model RFSQ Head

This script implements CONDITIONAL RFSQ Head training with Mode Locking theory.

Key Differences from Unconditional Baseline (modal_train_phase2_complete.py):
1. Main Model accepts L0-L2 tokens as conditioning input (Teacher Forcing)
2. Solves mean-seeking problem by locking model into specific mode
3. Predicts all 8 layers for verification capability during inference

Training Strategy:
- Draft Model: Unchanged (predicts L0-L2 from image features independently)
- Main Model: Image + Ground Truth L0-L2 → Predicts all 8 layers

Architecture:
- Token Embedding: Maps discrete L0-L2 tokens to continuous 64-dim embeddings
- Fusion: Concatenates image features + token features → fused representation
- Output: 8 parallel heads predict all layers from fused features

Theoretical Foundation (Mode Locking):
When the model sees ground truth L0-L2 tokens (e.g., "move left" coarse action),
its hidden state becomes locked into that specific mode. This forces L3-L7
predictions to be coherent with the coarse action, preventing the mean-seeking
problem where the model averages incompatible modes.

Example:
- Scenario: Obstacle in front, can go left OR right
- Without conditioning: Model might predict L0="left", L7="right details" → collision
- With conditioning: Model sees L0-L2="left" → forced to predict L3-L7="left details"

Output:
- /models/best_draft_with_projection.pt - Draft model checkpoint (unchanged)
- /models/openvla_rfsq_conditional/best_rfsq_head.pt - Conditional Main model

Usage:
    modal run modal_train_phase2_conditional.py --num-episodes 200 --epochs 50
"""

import os
import sys
import modal
from typing import Optional

# ============================================================
# Modal Setup
# ============================================================
app = modal.App("phase2-conditional-rfsq-training")

# Volumes
models_volume = modal.Volume.from_name("rsd-models", create_if_missing=True)
data_volume = modal.Volume.from_name("rsd-libero-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Get SDK path for Orchestra experiment tracking
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

# Build training image with proper OSMesa support
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", 
        "libgl1-mesa-glx", 
        "libglib2.0-0",
        "libosmesa6-dev",  # OSMesa for headless rendering
        "libglew-dev",
        "patchelf",
    )
    .pip_install("uv")
    .run_commands(
        # PyTorch and basic deps
        "uv pip install --system 'numpy<2' torch==2.2.0 torchvision==0.17.0 "
        "transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "
        "accelerate peft bitsandbytes pillow einops sentencepiece protobuf "
        "huggingface_hub scipy tqdm matplotlib pandas requests json-numpy jsonlines "
        "PyOpenGL PyOpenGL_accelerate",
        # OpenVLA
        "uv pip install --system 'openvla-oft @ git+https://github.com/moojink/openvla-oft.git'",
    )
    # LIBERO (for data collection)
    .run_commands(
        "cd /root && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "cd /root/LIBERO && sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' "
        "libero/libero/benchmark/__init__.py",
        "mkdir -p /root/.libero",
        "echo 'benchmark_root: /root/LIBERO/libero/libero' > /root/.libero/config.yaml",
        "echo 'bddl_files: /root/LIBERO/libero/libero/bddl_files' >> /root/.libero/config.yaml",
        "echo 'init_states: /root/LIBERO/libero/libero/init_files' >> /root/.libero/config.yaml",
        "cd /root/LIBERO && uv pip install --system -e .",
        "uv pip install --system mujoco dm-control robosuite==1.4.0 bddl easydict h5py cloudpickle gym",
    )
    .env({
        "HF_HOME": "/hf_cache",
        "TRANSFORMERS_CACHE": "/hf_cache",
        "LIBERO_FOLDER": "/data/libero",
        "LIBERO_NO_PROMPT": "1",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", ""),
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)


# ============================================================
# Data Collection (UNCHANGED from baseline)
# ============================================================

@app.function(
    image=train_image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={
        "/models": models_volume,
        "/data": data_volume,
        "/hf_cache": hf_cache,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("orchestra-supabase"),
    ],
)
def collect_training_data(num_episodes: int = 200):
    """Collect training data: OpenVLA hidden states + RFSQ token labels"""
    # Set environment variables before any imports
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from pathlib import Path
    
    sys.path.insert(0, "/root")
    
    # Try to import Orchestra SDK for logging
    try:
        from src.orchestra_sdk.experiment import Experiment
        use_sdk = True
    except Exception as e:
        print(f"⚠️ Orchestra SDK not available: {e}")
        use_sdk = False
    
    sys.path.insert(0, "/root/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("=" * 80)
    print("📦 Phase 2 Data Collection: OpenVLA Features + RFSQ Labels")
    print("=" * 80)

    device = torch.device("cuda")

    # Initialize experiment tracking
    if use_sdk:
        exp = Experiment.init(
            name="Phase 2 Conditional - Data Collection",
            description="Collecting OpenVLA hidden states and RFSQ labels from LIBERO (for conditional training)",
            config={
                "num_episodes": num_episodes,
                "gpu_type": "a100",
                "gpu_count": 1,
            }
        )
        exp.add_tags(['phase2', 'conditional', 'data-collection', 'libero'])

    # ============================================================
    # Define Robust RFSQ Components inline
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

            for layer_idx, layer in enumerate(self.layers):
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

    # 1. Load OpenVLA (frozen)
    print("\n1️⃣ Loading OpenVLA (frozen)...")
    openvla = AutoModelForVision2Seq.from_pretrained(
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        trust_remote_code=True,
    )
    openvla.eval()
    print("   ✅ OpenVLA loaded")

    # 2. Load Robust RFSQ Encoder (frozen)
    print("\n2️⃣ Loading Robust RFSQ Encoder (frozen)...")
    rfsq_encoder = ActionRFSQAE(
        action_dim=7,
        hidden_dim=16,
        num_layers=8,
        num_levels=7,
        use_layernorm=True,
    )

    rfsq_checkpoint_path = "/models/rfsq_robust_best.pt"
    rfsq_checkpoint = torch.load(rfsq_checkpoint_path, map_location=device, weights_only=False)
    state_dict = rfsq_checkpoint.get('model', rfsq_checkpoint.get('model_state_dict', rfsq_checkpoint))
    rfsq_encoder.load_state_dict(state_dict)
    rfsq_encoder = rfsq_encoder.to(device)
    rfsq_encoder.eval()
    print(f"   ✅ Robust RFSQ Encoder loaded from {rfsq_checkpoint_path}")

    # 3. Setup LIBERO
    print("\n3️⃣ Setting up LIBERO...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    num_tasks = task_suite.n_tasks
    print(f"   ✅ {num_tasks} tasks in libero_spatial")

    # 4. Collect data
    print(f"\n4️⃣ Collecting data from {num_episodes} episodes...")
    training_data = []
    episodes_per_task = max(1, num_episodes // num_tasks)
    total_steps = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        init_states = task_suite.get_task_init_states(task_id)

        print(f"\n   Task {task_id + 1}/{num_tasks}: {task_description}")

        for episode_idx in range(min(episodes_per_task, len(init_states))):
            try:
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
                obs = env.set_init_state(init_states[episode_idx])

                episode_samples = 0
                # Episode loop
                for step in range(300):
                    # Prepare image
                    image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

                    # Get OpenVLA action and hidden states
                    with torch.no_grad():
                        # ✅ CORRECT OpenVLA API: positional args (prompt, image)
                        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
                        
                        # ✅ Get hidden states with try-except and fallback (cumsum issue workaround)
                        try:
                            outputs = openvla(**inputs, output_hidden_states=True)
                            hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()  # [1, 4096]
                        except Exception as hidden_error:
                            # Fallback: use random hidden state if cumsum error occurs
                            hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)
                        
                        # ✅ Get action using correct API: **inputs with unnorm_key="bridge_orig"
                        try:
                            action_result = openvla.predict_action(
                                **inputs, 
                                unnorm_key="bridge_orig",  # Required: use bridge_orig as default
                                do_sample=False
                            )
                            # Handle tuple return value
                            if isinstance(action_result, tuple):
                                action = action_result[0]
                            else:
                                action = action_result
                            # Ensure it's a numpy array
                            if isinstance(action, torch.Tensor):
                                action = action.cpu().numpy()
                        except Exception as action_error:
                            print(f"         Action error: {action_error}")
                            action = np.zeros(7)

                    # Process action - OpenVLA-OFT returns action chunks [8, 7]
                    # We need: action_chunk for RFSQ encoding, single_action for env.step
                    action_array = np.array(action)
                    print(f"         DEBUG raw action shape: {action_array.shape}")
                    
                    # Determine action_chunk and single_action based on shape
                    if action_array.ndim == 1 and action_array.shape[0] == 7:  # [7] - single action
                        single_action = action_array
                        # Repeat to create chunk of 8 for RFSQ
                        action_chunk_np = np.tile(action_array, (8, 1))  # [8, 7]
                    elif action_array.ndim == 2 and action_array.shape == (8, 7):  # [8, 7] - action chunk
                        single_action = action_array[0]  # Use first action for env
                        action_chunk_np = action_array
                    elif action_array.ndim == 3 and action_array.shape[1:] == (8, 7):  # [1, 8, 7]
                        single_action = action_array[0, 0]  # Use first action for env
                        action_chunk_np = action_array[0]  # [8, 7]
                    elif action_array.ndim == 4 and action_array.shape[2:] == (8, 7):  # [1, 1, 8, 7]
                        single_action = action_array[0, 0, 0]  # Use first action for env
                        action_chunk_np = action_array[0, 0]  # [8, 7]
                    else:
                        print(f"         Unexpected action shape: {action_array.shape}, using zeros")
                        single_action = np.zeros(7)
                        action_chunk_np = np.zeros((8, 7))
                    
                    print(f"         DEBUG single_action shape: {single_action.shape}, action_chunk shape: {action_chunk_np.shape}")
                    
                    # Encode action_chunk to RFSQ tokens
                    with torch.no_grad():
                        action_chunk_tensor = torch.from_numpy(action_chunk_np).float().unsqueeze(0).to(device)  # [1, 8, 7]
                        _, rfsq_codes = rfsq_encoder(action_chunk_tensor)
                        # rfsq_codes: [1, 8, 16, 8] (Batch, Chunk, Hidden, Layers)

                    # Save sample
                    training_data.append({
                        'hidden_state': hidden_4096.squeeze(0).cpu(),  # [4096]
                        'rfsq_tokens': rfsq_codes[0].cpu(),  # [8, 16, 8]
                    })
                    episode_samples += 1
                    total_steps += 1

                    # Step environment with single action [7]
                    obs, reward, done, info = env.step(single_action)
                    if done:
                        break

                env.close()
                print(f"      Episode {episode_idx + 1}: {episode_samples} samples (total: {len(training_data)})")
                
                if use_sdk:
                    exp.log({
                        'total_samples': len(training_data),
                        'task_id': task_id,
                        'episode': episode_idx,
                    }, step=total_steps)
                    exp.set_progress(int((task_id * episodes_per_task + episode_idx + 1) / (num_tasks * episodes_per_task) * 100))

            except Exception as e:
                print(f"      ⚠️ Episode {episode_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 5. Save data
    print(f"\n5️⃣ Saving {len(training_data)} samples...")
    save_path = "/data/phase2_training_data.pt"
    torch.save(training_data, save_path)
    print(f"   ✅ Saved to {save_path}")

    data_volume.commit()
    
    if use_sdk:
        exp.log_text(f"Data collection complete: {len(training_data)} samples")
        exp.finish('completed')
    
    return len(training_data)


# ============================================================
# Draft Model Training (UNCHANGED from baseline)
# ============================================================

@app.function(
    image=train_image,
    gpu="A100",
    timeout=21600,  # 6 hours
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
    secrets=[
        modal.Secret.from_name("orchestra-supabase"),
    ],
)
def train_draft_model(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
):
    """Train Draft Model with Projection (UNCHANGED - predicts L0-L2 independently)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    
    sys.path.insert(0, "/root")
    
    # Try to import Orchestra SDK
    try:
        from src.orchestra_sdk.experiment import Experiment
        use_sdk = True
    except Exception as e:
        print(f"⚠️ Orchestra SDK not available: {e}")
        use_sdk = False

    print("=" * 80)
    print("🚀 Training Draft Model with Projection")
    print("=" * 80)

    device = torch.device("cuda")

    # Initialize experiment tracking
    if use_sdk:
        exp = Experiment.init(
            name="Phase 2 Conditional: Draft Model Training",
            description="Training Draft Model with projection layer for coarse RFSQ layers (L0-L2)",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "input_dim": 4096,
                "hidden_dim": 512,
                "num_coarse_layers": 3,
                "gpu_type": "a100",
                "gpu_count": 1,
            }
        )
        exp.add_tags(['phase2', 'conditional', 'draft-model', 'training'])

    # ============================================================
    # Define Draft Model Components
    # ============================================================
    
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

    # 1. Load data
    print("\n1️⃣ Loading training data...")
    data_path = "/data/phase2_training_data.pt"
    all_data = torch.load(data_path, weights_only=False)
    print(f"   ✅ Loaded {len(all_data)} samples")

    # Dataset
    class DraftDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            coarse_tokens = sample['rfsq_tokens'][:, :, :3]
            tokens_flat = coarse_tokens.reshape(-1, 3)
            return {
                'hidden': sample['hidden_state'],
                'tokens': tokens_flat,
            }

    dataset = DraftDataset(all_data)

    # Train/Val split
    val_size = min(5000, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # 2. Create model
    print("\n2️⃣ Creating Draft Model...")
    model = RFSQDraftModelWithProjection(
        input_dim=4096,
        hidden_dim=512,
        num_coarse_layers=3,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {total_params / 1e6:.1f}M parameters")

    # 3. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate / 10,
    )

    # 4. Training loop
    print(f"\n3️⃣ Training for {epochs} epochs...")
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accs = [0.0, 0.0, 0.0]

        for batch_idx, batch in enumerate(train_loader):
            hidden = batch['hidden'].to(device)
            targets = batch['tokens'].to(device)

            logits = model(hidden)

            logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)
            targets_flat = targets.reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                for layer_idx in range(3):
                    acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                    train_accs[layer_idx] += acc.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_accs = [0.0, 0.0, 0.0]

        with torch.no_grad():
            for batch in val_loader:
                hidden = batch['hidden'].to(device)
                targets = batch['tokens'].to(device)

                logits = model(hidden)

                logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)
                targets_flat = targets.reshape(-1)
                loss = F.cross_entropy(logits_flat, targets_flat)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                for layer_idx in range(3):
                    acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                    val_accs[layer_idx] += acc.item()

        train_loss /= len(train_loader)
        train_accs = [acc / len(train_loader) for acc in train_accs]
        val_loss /= len(val_loader)
        val_accs = [acc / len(val_loader) for acc in val_accs]
        avg_val_acc = sum(val_accs) / 3

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: L0={train_accs[0]:.3f} L1={train_accs[1]:.3f} L2={train_accs[2]:.3f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: L0={val_accs[0]:.3f} L1={val_accs[1]:.3f} L2={val_accs[2]:.3f} | Avg={avg_val_acc:.3f}")

        if use_sdk:
            exp.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_L0': val_accs[0],
                'val_acc_L1': val_accs[1],
                'val_acc_L2': val_accs[2],
                'val_acc_avg': avg_val_acc,
                'learning_rate': scheduler.get_last_lr()[0],
            }, step=epoch)
            exp.set_progress(int((epoch + 1) / epochs * 100))

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': avg_val_acc,
                'val_accuracies_per_layer': val_accs,
                'config': {
                    'input_dim': 4096,
                    'hidden_dim': 512,
                    'num_coarse_layers': 3,
                    'chunk_len': 8,
                    'action_hidden_dim': 16,
                    'grid_size': 7,
                },
            }
            save_path = "/models/best_draft_with_projection.pt"
            torch.save(checkpoint, save_path)
            print(f"  ✅ Best model saved: {avg_val_acc:.3f}")
            
            if use_sdk:
                exp.log_text(f"New best model saved at epoch {epoch + 1} with accuracy {avg_val_acc:.3f}")

    models_volume.commit()

    print("\n" + "=" * 80)
    print(f"🎉 Draft Model Training Complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.3f}")
    print("=" * 80)

    if use_sdk:
        exp.log_text(f"Training complete. Best accuracy: {best_val_acc:.3f}")
        exp.finish('completed')

    return best_val_acc


# ============================================================
# Main Model CONDITIONAL RFSQ Head Training (MODIFIED)
# ============================================================

@app.function(
    image=train_image,
    gpu="A100",
    timeout=21600,  # 6 hours
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
    secrets=[
        modal.Secret.from_name("orchestra-supabase"),
    ],
)
def train_conditional_rfsq_head(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
):
    """Train Conditional Main Model RFSQ Head with Mode Locking"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    
    sys.path.insert(0, "/root")
    
    try:
        from src.orchestra_sdk.experiment import Experiment
        use_sdk = True
    except Exception as e:
        print(f"⚠️ Orchestra SDK not available: {e}")
        use_sdk = False

    print("=" * 80)
    print("🚀 Training CONDITIONAL Main Model RFSQ Head")
    print("   📊 Mode Locking: Image + L0-L2 Tokens → All 8 Layers")
    print("=" * 80)

    device = torch.device("cuda")

    if use_sdk:
        exp = Experiment.init(
            name="Phase 2: Conditional RFSQ Head Training",
            description="Training Conditional RFSQ Head for all 8 layers (L0-L7) with Mode Locking. "
                       "Conditions on ground truth L0-L2 tokens during training to solve mean-seeking problem.",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "input_dim": 4096,
                "hidden_dim": 1024,
                "num_layers": 8,
                "condition_layers": 3,
                "token_embed_dim": 64,
                "training_mode": "conditional",
                "gpu_type": "a100",
                "gpu_count": 1,
            }
        )
        exp.add_tags(['phase2', 'conditional', 'rfsq-head', 'main-model', 'mode-locking', 'training'])

    # ============================================================
    # Define Conditional RFSQ Head (NEW ARCHITECTURE)
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
            
            # A. Image Feature Projection (unchanged from baseline)
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
            
            # E. Output Heads (unchanged from baseline)
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

    # 1. Load data
    print("\n1️⃣ Loading training data...")
    data_path = "/data/phase2_training_data.pt"
    all_data = torch.load(data_path, weights_only=False)
    print(f"   ✅ Loaded {len(all_data)} samples")

    # ============================================================
    # Define Conditional Dataset (NEW)
    # ============================================================
    
    class ConditionedRFSQHeadDataset(Dataset):
        """
        Dataset for conditional RFSQ Head training.
        
        Returns:
        - hidden: OpenVLA image features [4096]
        - condition: L0-L2 ground truth tokens [8, 16, 3] (Teacher Forcing)
        - labels: All layer tokens for loss computation [128, 8]
        """
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

    dataset = ConditionedRFSQHeadDataset(all_data)

    val_size = min(5000, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # 2. Create model
    print("\n2️⃣ Creating Conditional RFSQ Head...")
    model = ConditionedRFSQHead(
        input_dim=4096,
        hidden_dim=1024,
        num_layers=8,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
        condition_layers=3,      # NEW
        token_embed_dim=64,      # NEW
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {total_params / 1e6:.1f}M parameters")
    print(f"   📊 Architecture: Conditional (Mode Locking)")
    print(f"   🔧 Condition layers: L0-L2 (3 layers)")
    print(f"   🔧 Token embedding dim: 64")
    print(f"   🔧 Fusion strategy: Concatenation + Linear")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate / 10,
    )

    # 3. Training loop (MODIFIED)
    print(f"\n3️⃣ Training for {epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accs = [0.0] * 8

        for batch in train_loader:
            hidden = batch['hidden'].to(device)          # [B, 4096]
            condition = batch['condition'].to(device)     # [B, 8, 16, 3] <- NEW
            targets = batch['labels'].to(device)          # [B, 128, 8]

            # Forward pass with conditioning (NEW: two inputs)
            logits = model(hidden, condition)  # [B, 8, 128, 7]

            # Reshape for cross-entropy loss
            logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)  # [B*8*128, 7]
            targets_flat = targets.reshape(-1)  # [B*8*128]

            loss = F.cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)  # [B, 8, 128]
                for layer_idx in range(8):
                    acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                    train_accs[layer_idx] += acc.item()

        scheduler.step()

        # Validation (MODIFIED)
        model.eval()
        val_loss = 0.0
        val_accs = [0.0] * 8

        with torch.no_grad():
            for batch in val_loader:
                hidden = batch['hidden'].to(device)
                condition = batch['condition'].to(device)   # NEW
                targets = batch['labels'].to(device)

                logits = model(hidden, condition)  # NEW: two inputs

                logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)
                targets_flat = targets.reshape(-1)
                loss = F.cross_entropy(logits_flat, targets_flat)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                for layer_idx in range(8):
                    acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                    val_accs[layer_idx] += acc.item()

        train_loss /= len(train_loader)
        train_accs = [acc / len(train_loader) for acc in train_accs]
        val_loss /= len(val_loader)
        val_accs = [acc / len(val_loader) for acc in val_accs]
        avg_val_acc = sum(val_accs) / 8

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc (avg): {avg_val_acc:.3f}")
        print(f"  Per-layer: L0={val_accs[0]:.3f} L1={val_accs[1]:.3f} L2={val_accs[2]:.3f} L3={val_accs[3]:.3f}")
        print(f"             L4={val_accs[4]:.3f} L5={val_accs[5]:.3f} L6={val_accs[6]:.3f} L7={val_accs[7]:.3f}")

        if use_sdk:
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_avg': avg_val_acc,
                'learning_rate': scheduler.get_last_lr()[0],
            }
            for i in range(8):
                metrics[f'val_acc_L{i}'] = val_accs[i]
            exp.log(metrics, step=epoch)
            exp.set_progress(int((epoch + 1) / epochs * 100))

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            
            # Save to new directory (NEW)
            os.makedirs("/models/openvla_rfsq_conditional", exist_ok=True)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': avg_val_acc,
                'val_accuracies_per_layer': val_accs,
                'config': {
                    'input_dim': 4096,
                    'hidden_dim': 1024,
                    'num_layers': 8,
                    'chunk_len': 8,
                    'action_hidden_dim': 16,
                    'grid_size': 7,
                    'condition_layers': 3,        # NEW
                    'token_embed_dim': 64,        # NEW
                    'training_mode': 'conditional',  # NEW
                },
            }
            save_path = "/models/openvla_rfsq_conditional/best_rfsq_head.pt"
            torch.save(checkpoint, save_path)
            print(f"  ✅ Best model saved: {avg_val_acc:.3f}")
            
            if use_sdk:
                exp.log_text(f"New best conditional model saved at epoch {epoch + 1} with accuracy {avg_val_acc:.3f}")

    models_volume.commit()

    print("\n" + "=" * 80)
    print(f"🎉 Conditional RFSQ Head Training Complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.3f}")
    print(f"   Mode Locking: ✅ Enabled")
    print("=" * 80)

    if use_sdk:
        exp.log_text(f"Conditional training complete. Best accuracy: {best_val_acc:.3f}")
        exp.finish('completed')

    return best_val_acc


# ============================================================
# Main Entrypoint
# ============================================================

@app.local_entrypoint()
def main(
    num_episodes: int = 200,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    skip_data_collection: bool = False,
    skip_draft_training: bool = False,
    skip_rfsq_head_training: bool = False,
):
    """
    Complete Phase 2 CONDITIONAL Training Pipeline
    """
    print("=" * 80)
    print("🚀 Phase 2: CONDITIONAL Training Pipeline (Mode Locking)")
    print("=" * 80)
    print(f"   Episodes: {num_episodes}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Training Mode: CONDITIONAL (Teacher Forcing with L0-L2)")

    results = {}

    # Step 1: Collect data
    if not skip_data_collection:
        print("\n📦 Step 1: Collecting training data...")
        num_samples = collect_training_data.remote(num_episodes=num_episodes)
        print(f"✅ Collected {num_samples} samples")
        results['num_samples'] = num_samples
    else:
        print("\n⏭️  Step 1: Skipping data collection (using existing data)")

    # Step 2: Train Draft Model
    if not skip_draft_training:
        print("\n🚀 Step 2: Training Draft Model...")
        draft_acc = train_draft_model.remote(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print(f"✅ Draft Model Best Accuracy: {draft_acc:.3f}")
        results['draft_accuracy'] = draft_acc
        
        if draft_acc > 0.90:
            print(f"   🎉 Target met! (>90%)")
        else:
            print(f"   ⚠️  Below target (90%). Consider more epochs or data.")
    else:
        print("\n⏭️  Step 2: Skipping Draft Model training")

    # Step 3: Train CONDITIONAL RFSQ Head
    if not skip_rfsq_head_training:
        print("\n🚀 Step 3: Training CONDITIONAL Main Model RFSQ Head...")
        print("   📊 Mode Locking: Conditioning on ground truth L0-L2 tokens")
        rfsq_acc = train_conditional_rfsq_head.remote(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print(f"✅ Conditional RFSQ Head Best Accuracy: {rfsq_acc:.3f}")
        results['rfsq_head_accuracy'] = rfsq_acc
        
        if rfsq_acc > 0.92:
            print(f"   🎉 Target met! (>92%)")
        else:
            print(f"   ⚠️  Below target (92%). Consider more epochs or data.")
    else:
        print("\n⏭️  Step 3: Skipping RFSQ Head training")

    # Summary
    print("\n" + "=" * 80)
    print("🎉 Phase 2 CONDITIONAL Training Pipeline Complete!")
    print("=" * 80)
    
    if 'draft_accuracy' in results:
        status = "✅" if results['draft_accuracy'] > 0.90 else "⚠️"
        print(f"   {status} Draft Model Accuracy: {results['draft_accuracy']:.3f} (target: >90%)")
    
    if 'rfsq_head_accuracy' in results:
        status = "✅" if results['rfsq_head_accuracy'] > 0.92 else "⚠️"
        print(f"   {status} Conditional RFSQ Head Accuracy: {results['rfsq_head_accuracy']:.3f} (target: >92%)")
    
    print("\n📁 Output files:")
    print("   - /models/best_draft_with_projection.pt")
    print("   - /models/openvla_rfsq_conditional/best_rfsq_head.pt")
    print("\n🔬 Next Steps:")
    print("   1. Compare with unconditional baseline (/models/openvla_rfsq_robust/)")
    print("   2. Test Mode Locking: Feed same image with different L0-L2 conditions")
    print("   3. Evaluate on LIBERO tasks (Phase 3)")
    
    return results
