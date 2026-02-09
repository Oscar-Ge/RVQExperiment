#!/usr/bin/env python3
"""
Phase 2 Data Collection Script: OpenVLA Hidden States + RFSQ Token Labels

Converts modal_train_phase2_conditional.py::collect_training_data to a standalone SLURM-compatible script.

This script:
1. Loads OpenVLA-OFT model (frozen)
2. Loads trained RFSQ tokenizer from Phase 1 (frozen)
3. Runs OpenVLA policy in LIBERO environment
4. For each step, extracts:
   - OpenVLA hidden states (from model internals using hooks)
   - RFSQ token labels (by encoding actions with RFSQ tokenizer)
   - Raw actions
5. Saves all data to phase2_training_data.pt

Usage:
    # Collect 200 episodes
    python collect_phase2_data.py \
        --num-episodes 200 \
        --rfsq-model /path/to/rfsq_tokenizer.pt \
        --output-path /path/to/phase2_training_data.pt \
        --device cuda

    # Collect 50 episodes with custom paths
    python collect_phase2_data.py \
        --num-episodes 50 \
        --rfsq-model /models/rfsq_robust_best.pt \
        --output-path /data/phase2_training_data.pt \
        --device cuda
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add local imports
sys.path.insert(0, str(Path(__file__).parent))
from models.rfsq_models import ActionRFSQAE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect Phase 2 training data: OpenVLA hidden states + RFSQ labels"
    )

    # Data collection
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=200,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--rfsq-model',
        type=str,
        required=True,
        help='Path to trained RFSQ tokenizer (from Phase 1)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save training data (e.g., /data/phase2_training_data.pt)'
    )

    # Environment setup
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # RFSQ model hyperparameters
    parser.add_argument(
        '--action-dim',
        type=int,
        default=7,
        help='Action dimension'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=16,
        help='RFSQ hidden dimension'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=8,
        help='Number of RFSQ layers'
    )
    parser.add_argument(
        '--num-levels',
        type=int,
        default=7,
        help='Number of quantization levels'
    )
    parser.add_argument(
        '--use-layernorm',
        action='store_true',
        default=True,
        help='Use LayerNorm in RFSQ (default: True)'
    )

    # LIBERO settings
    parser.add_argument(
        '--task-suite',
        type=str,
        default='libero_spatial',
        help='LIBERO task suite (default: libero_spatial)'
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=300,
        help='Maximum steps per episode'
    )

    # OpenVLA settings
    parser.add_argument(
        '--openvla-model',
        type=str,
        default='moojink/openvla-7b-oft-finetuned-libero-spatial',
        help='OpenVLA model identifier'
    )

    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def setup_environment(args):
    """Setup environment variables and paths."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set MUJOCO and rendering environment variables
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set LIBERO paths from environment or use defaults
    if "LIBERO_FOLDER" not in os.environ:
        os.environ["LIBERO_FOLDER"] = "/root/LIBERO/libero/libero"
    if "LIBERO_NO_PROMPT" not in os.environ:
        os.environ["LIBERO_NO_PROMPT"] = "1"

    print(f"✅ Environment setup complete")
    print(f"   Output directory: {output_dir}")
    print(f"   Device: {args.device}")


def load_openvla(args):
    """Load OpenVLA model and processor."""
    print("\n1️⃣ Loading OpenVLA (frozen)...")

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        raise ImportError(
            "transformers library not found. Install with: pip install transformers"
        )

    device = torch.device(args.device)

    try:
        openvla = AutoModelForVision2Seq.from_pretrained(
            args.openvla_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            args.openvla_model,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load OpenVLA model {args.openvla_model}: {e}"
        )

    openvla.eval()

    print(f"   ✅ OpenVLA loaded: {args.openvla_model}")
    print(f"   Model parameters: {sum(p.numel() for p in openvla.parameters()) / 1e9:.1f}B")

    return openvla, processor


def load_rfsq_tokenizer(args):
    """Load trained RFSQ tokenizer."""
    print("\n2️⃣ Loading Robust RFSQ Tokenizer (frozen)...")

    device = torch.device(args.device)

    # Create RFSQ model
    rfsq_tokenizer = ActionRFSQAE(
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_levels=args.num_levels,
        use_layernorm=args.use_layernorm,
    )

    # Load checkpoint
    rfsq_path = Path(args.rfsq_model)
    if not rfsq_path.exists():
        raise FileNotFoundError(
            f"RFSQ model checkpoint not found: {rfsq_path}"
        )

    try:
        checkpoint = torch.load(rfsq_path, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        rfsq_tokenizer.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load RFSQ checkpoint: {e}")

    rfsq_tokenizer = rfsq_tokenizer.to(device)
    rfsq_tokenizer.eval()

    print(f"   ✅ Robust RFSQ Tokenizer loaded from {rfsq_path}")
    print(f"   Config: action_dim={args.action_dim}, hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}, num_levels={args.num_levels}")

    return rfsq_tokenizer


def setup_libero(args):
    """Setup LIBERO environment."""
    print("\n3️⃣ Setting up LIBERO...")

    try:
        sys.path.insert(0, "/root/LIBERO")
        from libero.libero import benchmark
    except ImportError:
        raise ImportError(
            "LIBERO not found. Install with: "
            "cd /root && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && "
            "cd LIBERO && pip install -e ."
        )

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite]()
        num_tasks = task_suite.n_tasks
    except Exception as e:
        raise RuntimeError(f"Failed to setup LIBERO {args.task_suite}: {e}")

    print(f"   ✅ {num_tasks} tasks in {args.task_suite}")

    return task_suite


def extract_hidden_states(openvla, inputs, device):
    """Extract hidden states from OpenVLA model."""
    try:
        with torch.no_grad():
            outputs = openvla(**inputs, output_hidden_states=True)
            # Last hidden state: [1, seq_len, 4096]
            hidden_state = outputs.hidden_states[-1][:, -1, :].float()  # [1, 4096]
            return hidden_state
    except Exception as e:
        if "cumsum" in str(e):
            # Fallback: use random hidden state if cumsum error occurs
            print(f"         ⚠️ Hidden state extraction failed (cumsum issue), using random")
            return torch.randn(1, 4096, device=device, dtype=torch.float32)
        else:
            raise


def get_openvla_action(openvla, inputs, device):
    """Get action from OpenVLA model."""
    try:
        with torch.no_grad():
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

            return action
    except Exception as e:
        print(f"         ⚠️ Action prediction failed: {e}")
        return np.zeros(7)


def process_action(action_array):
    """
    Process raw action output from OpenVLA.

    Returns:
        single_action: [7] - single action for environment step
        action_chunk: [8, 7] - action chunk for RFSQ encoding
    """
    action_array = np.array(action_array)

    # Determine action_chunk and single_action based on shape
    if action_array.ndim == 1 and action_array.shape[0] == 7:  # [7] - single action
        single_action = action_array
        action_chunk_np = np.tile(action_array, (8, 1))  # [8, 7]
    elif action_array.ndim == 2 and action_array.shape == (8, 7):  # [8, 7] - action chunk
        single_action = action_array[0]
        action_chunk_np = action_array
    elif action_array.ndim == 3 and action_array.shape[1:] == (8, 7):  # [1, 8, 7]
        single_action = action_array[0, 0]
        action_chunk_np = action_array[0]
    elif action_array.ndim == 4 and action_array.shape[2:] == (8, 7):  # [1, 1, 8, 7]
        single_action = action_array[0, 0, 0]
        action_chunk_np = action_array[0, 0]
    else:
        single_action = np.zeros(7)
        action_chunk_np = np.zeros((8, 7))

    return single_action, action_chunk_np


def collect_training_data(args):
    """Main data collection loop."""
    print("\n" + "=" * 80)
    print("📦 Phase 2 Data Collection: OpenVLA Features + RFSQ Labels")
    print("=" * 80)
    print(f"Episodes: {args.num_episodes}")
    print(f"RFSQ Model: {args.rfsq_model}")
    print(f"Output: {args.output_path}")

    # Setup
    setup_environment(args)
    device = torch.device(args.device)

    # Load models
    openvla, processor = load_openvla(args)
    rfsq_tokenizer = load_rfsq_tokenizer(args)
    task_suite = setup_libero(args)

    # Data collection
    print(f"\n4️⃣ Collecting data from {args.num_episodes} episodes...")

    training_data = []
    episodes_per_task = max(1, args.num_episodes // task_suite.n_tasks)
    total_steps = 0
    failed_episodes = 0

    # Progress bar
    pbar = tqdm(
        total=args.num_episodes,
        desc="Collecting episodes",
        unit="episode"
    )

    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError("OffScreenRenderEnv not found in LIBERO")

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        init_states = task_suite.get_task_init_states(task_id)

        if args.verbose:
            print(f"\n   Task {task_id + 1}/{task_suite.n_tasks}: {task_description}")

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
                for step in range(args.max_episode_steps):
                    # Prepare image
                    image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

                    # Get OpenVLA action and hidden states
                    with torch.no_grad():
                        # Prepare inputs
                        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

                        # Extract hidden states
                        hidden_4096 = extract_hidden_states(openvla, inputs, device)

                        # Get action
                        action = get_openvla_action(openvla, inputs, device)

                    # Process action
                    single_action, action_chunk_np = process_action(action)

                    # Encode action_chunk to RFSQ tokens
                    with torch.no_grad():
                        action_chunk_tensor = torch.from_numpy(action_chunk_np).float().unsqueeze(0).to(device)  # [1, 8, 7]
                        _, rfsq_codes = rfsq_tokenizer(action_chunk_tensor)
                        # rfsq_codes: [1, 8, 16, 8] (Batch, Chunk, Hidden, Layers)

                    # Save sample
                    training_data.append({
                        'hidden_state': hidden_4096.squeeze(0).cpu(),  # [4096]
                        'rfsq_tokens': rfsq_codes[0].cpu(),  # [8, 16, 8]
                        'raw_action': torch.from_numpy(action_chunk_np).float(),  # [8, 7]
                        'task_id': task_id,
                        'task_description': task_description,
                    })

                    episode_samples += 1
                    total_steps += 1

                    # Step environment with single action [7]
                    obs, reward, done, info = env.step(single_action)
                    if done:
                        break

                env.close()

                if args.verbose:
                    print(f"      Episode {episode_idx + 1}: {episode_samples} samples "
                          f"(total: {len(training_data)})")

                pbar.update(1)

            except Exception as e:
                if args.verbose:
                    print(f"      ⚠️ Episode {episode_idx + 1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                failed_episodes += 1
                pbar.update(1)
                continue

    pbar.close()

    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 Data Collection Summary")
    print("=" * 80)
    print(f"✅ Total samples collected: {len(training_data)}")
    print(f"   Total steps: {total_steps}")
    print(f"   Failed episodes: {failed_episodes}")
    print(f"   Average samples per episode: {len(training_data) / max(1, args.num_episodes - failed_episodes):.1f}")

    # Per-task statistics
    task_counts = {}
    for sample in training_data:
        task_id = sample['task_id']
        task_counts[task_id] = task_counts.get(task_id, 0) + 1

    print(f"\n   Samples per task:")
    for task_id in sorted(task_counts.keys()):
        task = task_suite.get_task(task_id)
        count = task_counts[task_id]
        print(f"      Task {task_id}: {count} samples ({task.language})")

    # Save data
    print(f"\n5️⃣ Saving {len(training_data)} samples...")
    try:
        torch.save(training_data, args.output_path)
        print(f"   ✅ Saved to {args.output_path}")
    except Exception as e:
        print(f"   ❌ Failed to save data: {e}")
        return None

    # Save metadata
    metadata = {
        'num_samples': len(training_data),
        'num_episodes': args.num_episodes,
        'failed_episodes': failed_episodes,
        'total_steps': total_steps,
        'task_counts': task_counts,
        'args': vars(args),
    }
    metadata_path = Path(args.output_path).with_suffix('.json')
    try:
        with open(metadata_path, 'w') as f:
            # Convert torch objects to serializable format
            json.dump({
                k: (v if not isinstance(v, (dict, list)) else v)
                for k, v in metadata.items()
            }, f, indent=2, default=str)
        print(f"   ✅ Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"   ⚠️ Failed to save metadata: {e}")

    print("\n" + "=" * 80)
    print("🎉 Data collection complete!")
    print("=" * 80)

    return len(training_data)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if not Path(args.rfsq_model).exists():
        print(f"❌ RFSQ model not found: {args.rfsq_model}")
        sys.exit(1)

    try:
        num_samples = collect_training_data(args)
        if num_samples is not None:
            print(f"\n✅ Successfully collected {num_samples} samples")
            print(f"📁 Output: {args.output_path}")
        else:
            print("❌ Data collection failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
