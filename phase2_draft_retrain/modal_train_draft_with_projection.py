"""
Draft Model Training with Projection Layer

这个脚本训练一个包含projection layer的Draft Model，能够：
1. 接受OpenVLA的4096维hidden states
2. 预测RFSQ的前3个coarse layers
3. 达到>85%的准确率

训练完成后，可以直接集成到Phase 3使用。

Usage:
    modal run modal_train_draft_with_projection.py \
        --num-episodes 200 \
        --epochs 50 \
        --batch-size 32

注意：这是模板代码，agent需要根据实际环境调整。
"""

import os
import sys
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

# ============================================================
# Modal Setup
# ============================================================
app = modal.App("draft-model-training-with-projection")

# Volumes
models_volume = modal.Volume.from_name("rsd-models", create_if_missing=True)
data_volume = modal.Volume.from_name("rsd-libero-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Get SDK path
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

# Build training image
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        # PyTorch and basic deps
        "uv pip install --system 'numpy<2' torch==2.2.0 torchvision==0.17.0 "
        "transformers==4.40.1 timm==0.9.10 tokenizers==0.19.1 "
        "accelerate peft bitsandbytes pillow einops sentencepiece protobuf "
        "huggingface_hub scipy tqdm matplotlib pandas requests json-numpy jsonlines",
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
        "uv pip install --system mujoco dm-control robosuite==1.4.0",
    )
    .env({
        "HF_HOME": "/hf_cache",
        "TRANSFORMERS_CACHE": "/hf_cache",
        "LIBERO_FOLDER": "/data/libero",
        "LIBERO_NO_PROMPT": "1",
        "MUJOCO_GL": "osmesa",
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)


# ============================================================
# Model Definitions
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
    Draft Model with Projection Layer

    关键改进：
    1. 添加了input_projection (4096 -> 512)
    2. 可以直接处理OpenVLA的hidden states
    3. 预测RFSQ的前3个coarse layers
    """

    def __init__(
        self,
        input_dim=4096,         # OpenVLA hidden size
        hidden_dim=512,         # Draft model internal size
        num_coarse_layers=3,    # L0, L1, L2
        chunk_len=8,
        action_hidden_dim=16,   # RFSQ hidden dim
        grid_size=7,            # RFSQ vocab size
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_coarse_layers = num_coarse_layers
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size

        # 🔑 Projection Layer (4096 -> 512)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer Decoder
        self.decoder = DraftTransformerDecoder(hidden_dim=hidden_dim)

        # Classification Heads (一个head预测一个layer)
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
        """
        Args:
            openvla_hidden_states: [Batch, 4096]

        Returns:
            logits: [Batch, Num_Coarse_Layers=3, Chunk*Hidden=128, Grid=7]
        """
        batch_size = openvla_hidden_states.shape[0]

        # Step 1: Project 4096 -> 512
        projected = self.input_projection(openvla_hidden_states)  # [B, 512]

        # Step 2: Add sequence dimension
        x = projected.unsqueeze(1)  # [B, 1, 512]

        # Step 3: Transformer Decoder
        decoder_output = self.decoder(x)  # [B, 1, 512]
        decoder_output = decoder_output.squeeze(1)  # [B, 512]

        # Step 4: Classification Heads
        layer_outputs = []
        for head in self.classification_heads:
            logits = head(decoder_output)  # [B, 8*16*7=896]
            # Reshape to [B, Chunk*Hidden=128, Grid=7]
            logits = logits.view(batch_size, self.chunk_len * self.action_hidden_dim, self.grid_size)
            layer_outputs.append(logits)

        # Stack: [B, 3, 128, 7]
        return torch.stack(layer_outputs, dim=1)


# ============================================================
# RFSQ Components (for encoding actions)
# ============================================================

class STEQuantizer(nn.Module):
    """Straight-Through Estimator Quantizer"""

    def __init__(self, num_levels=7):
        super().__init__()
        self.num_levels = num_levels
        self.register_buffer('boundaries', torch.linspace(-1, 1, num_levels))

    def forward(self, z):
        dist = torch.abs(z.unsqueeze(-1) - self.boundaries.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(dist, dim=-1)
        z_q = self.boundaries[indices]
        z_q_out = z + (z_q - z).detach()
        return z_q_out, indices


class RFSQBlock(nn.Module):
    """Residual Finite Scalar Quantization Block"""

    def __init__(self, num_layers=8, num_levels=7):
        super().__init__()
        self.num_levels = num_levels
        self.layers = nn.ModuleList([
            STEQuantizer(num_levels=num_levels) for _ in range(num_layers)
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


class ActionRFSQAE(nn.Module):
    """Action RFSQ AutoEncoder (Phase 1 trained)"""

    def __init__(self, action_dim=7, hidden_dim=16, num_layers=8, num_levels=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.Mish(),
            nn.Linear(64, hidden_dim),
            nn.Tanh()
        )
        self.rfsq = RFSQBlock(num_layers=num_layers, num_levels=num_levels)
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


# ============================================================
# Data Collection
# ============================================================

@app.function(
    image=train_image,
    gpu="A100",
    timeout=7200,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
        "/hf_cache": hf_cache,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def collect_training_data(num_episodes: int = 200):
    """
    收集训练数据：OpenVLA hidden states + RFSQ token labels

    返回：训练样本数量
    """
    import numpy as np
    from PIL import Image
    from pathlib import Path
    sys.path.insert(0, "/root/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("=" * 80)
    print("📦 Data Collection: OpenVLA Features + RFSQ Labels")
    print("=" * 80)

    device = torch.device("cuda")

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

    # 2. Load RFSQ Encoder (frozen)
    print("\n2️⃣ Loading RFSQ Encoder (frozen)...")
    rfsq_encoder = ActionRFSQAE(action_dim=7, hidden_dim=16, num_layers=8, num_levels=7)
    rfsq_checkpoint = torch.load("/models/rfsq_best.pt", map_location=device, weights_only=False)
    state_dict = rfsq_checkpoint.get('model', rfsq_checkpoint.get('model_state_dict', rfsq_checkpoint))
    rfsq_encoder.load_state_dict(state_dict)
    rfsq_encoder = rfsq_encoder.to(device)
    rfsq_encoder.eval()
    print("   ✅ RFSQ Encoder loaded")

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

                # Episode loop
                for step in range(300):
                    # Prepare image
                    image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

                    # Get OpenVLA hidden states
                    with torch.no_grad():
                        inputs = processor(
                            text=task_description,
                            images=image,
                            return_tensors="pt"
                        ).to(device)

                        outputs = openvla(**inputs, output_hidden_states=True)
                        hidden_4096 = outputs.hidden_states[-1][:, -1, :]  # [1, 4096]

                        # Get action
                        action = openvla.predict_action(
                            image,
                            task_description,
                            unnorm_key="libero_spatial",
                        )

                    # Encode action to RFSQ tokens
                    with torch.no_grad():
                        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)  # [1, 7]
                        # Expand to chunk
                        action_chunk = action_tensor.unsqueeze(1).expand(1, 8, 7)  # [1, 8, 7]
                        _, rfsq_codes = rfsq_encoder(action_chunk)
                        # rfsq_codes: [1, 8, 16, 8] (Batch, Chunk, Hidden, Layers)

                        # Extract coarse layers (L0, L1, L2)
                        coarse_tokens = rfsq_codes[0, :, :, :3]  # [8, 16, 3]

                    # Save sample
                    training_data.append({
                        'hidden_state': hidden_4096.squeeze(0).cpu(),  # [4096]
                        'coarse_tokens': coarse_tokens.cpu(),  # [8, 16, 3]
                    })

                    # Step environment
                    obs, reward, done, info = env.step(action)
                    if done:
                        break

                env.close()
                print(f"      Episode {episode_idx + 1}: {len(training_data)} samples")

            except Exception as e:
                print(f"      ⚠️ Episode {episode_idx + 1} failed: {e}")
                continue

    # 5. Save data
    print(f"\n5️⃣ Saving {len(training_data)} samples...")
    save_path = "/data/draft_training_data.pt"
    torch.save(training_data, save_path)
    print(f"   ✅ Saved to {save_path}")

    data_volume.commit()
    return len(training_data)


# ============================================================
# Training
# ============================================================

@app.function(
    image=train_image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
)
def train_draft_model(
    num_episodes: int = 200,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
):
    """训练Draft Model with Projection"""
    from torch.utils.data import Dataset, DataLoader, random_split

    print("=" * 80)
    print("🚀 Training Draft Model with Projection")
    print("=" * 80)

    device = torch.device("cuda")

    # 1. Load data
    print("\n1️⃣ Loading training data...")
    data_path = "/data/draft_training_data.pt"
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
            # hidden_state: [4096]
            # coarse_tokens: [8, 16, 3]
            # Flatten tokens to [8*16, 3] = [128, 3]
            tokens_flat = sample['coarse_tokens'].view(-1, 3)
            return {
                'hidden': sample['hidden_state'],
                'tokens': tokens_flat,  # [128, 3]
            }

    dataset = DraftDataset(all_data)

    # Train/Val split
    val_size = min(5000, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accs = [0.0, 0.0, 0.0]

        for batch in train_loader:
            hidden = batch['hidden'].to(device)  # [B, 4096]
            targets = batch['tokens'].to(device)  # [B, 128, 3]

            # Forward
            logits = model(hidden)  # [B, 3, 128, 7]

            # Compute loss
            # Reshape: [B, 3, 128, 7] -> [B*3*128, 7]
            logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, 7)
            targets_flat = targets.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # Accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)  # [B, 3, 128]
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
                targets_flat = targets.view(-1)
                loss = F.cross_entropy(logits_flat, targets_flat)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                for layer_idx in range(3):
                    acc = (preds[:, layer_idx] == targets[:, :, layer_idx]).float().mean()
                    val_accs[layer_idx] += acc.item()

        # Average
        train_loss /= len(train_loader)
        train_accs = [acc / len(train_loader) for acc in train_accs]
        val_loss /= len(val_loader)
        val_accs = [acc / len(val_loader) for acc in val_accs]
        avg_val_acc = sum(val_accs) / 3

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: L0={train_accs[0]:.3f} L1={train_accs[1]:.3f} L2={train_accs[2]:.3f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: L0={val_accs[0]:.3f} L1={val_accs[1]:.3f} L2={val_accs[2]:.3f} | Avg={avg_val_acc:.3f}")

        # Save best
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
                },
            }
            save_path = "/models/best_draft_with_projection.pt"
            torch.save(checkpoint, save_path)
            print(f"  ✅ Best model saved: {avg_val_acc:.3f}")

    models_volume.commit()

    print("\n" + "=" * 80)
    print(f"🎉 Training Complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.3f}")
    print("=" * 80)

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
):
    """
    训练Pipeline

    Args:
        num_episodes: 收集多少episodes的数据
        epochs: 训练多少epochs
        batch_size: batch size
        learning_rate: 学习率
        skip_data_collection: 是否跳过数据收集（如果已有数据）
    """
    print("🚀 Starting Draft Model Training Pipeline")
    print(f"   Episodes: {num_episodes}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")

    # Step 1: Collect data
    if not skip_data_collection:
        print("\n📦 Step 1: Collecting training data...")
        num_samples = collect_training_data.remote(num_episodes=num_episodes)
        print(f"✅ Collected {num_samples} samples")
    else:
        print("\n⏭️  Step 1: Skipping data collection (using existing data)")

    # Step 2: Train model
    print("\n🚀 Step 2: Training Draft Model...")
    best_acc = train_draft_model.remote(
        num_episodes=num_episodes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print(f"\n✅ Pipeline Complete!")
    print(f"   Best Validation Accuracy: {best_acc:.3f}")

    if best_acc > 0.85:
        print(f"   🎉 Success! Accuracy target met (>85%)")
    else:
        print(f"   ⚠️  Accuracy below target. Consider:")
        print(f"      - Increase num_episodes")
        print(f"      - Increase epochs")
        print(f"      - Adjust learning_rate")

    return best_acc
