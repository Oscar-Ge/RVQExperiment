"""
Phase 3: LIBERO Evaluation with RSD Inference Engine
FIXED v2 - Uses official openvla-oft code for action prediction

Key fixes:
1. Image rotation (180 degrees) - CRITICAL for LIBERO
2. Uses official get_vla_action from openvla_utils.py
3. Proper proprio normalization with q01/q99
4. Correct action processing (normalize + invert gripper)

Usage:
    modal run finalCode/modal_phase3_libero_eval_FIXED_v2.py --num-trials 3
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
app = modal.App("rsd-phase3-libero-eval-fixed-v2")

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
        # Install prismatic from openvla-oft repo (required for fine-tuned model)
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
    use_speculative_decoding: bool = False,  # 先关闭 RSD，测试 baseline
):
    """Run LIBERO evaluation using official openvla-oft code."""
    import sys
    import torch
    import torch.nn as nn
    import numpy as np
    import time
    import json
    from pathlib import Path

    sys.path.insert(0, "/root/LIBERO")
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/openvla-oft")  # 添加 openvla-oft 到 path

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # 导入官方的工具函数！
    from experiments.robot.libero.libero_utils import (
        quat2axisangle,
        get_libero_dummy_action,  # Added: for stabilization period
    )
    from experiments.robot.openvla_utils import (
        get_vla,
        get_processor,
        get_action_head,
        get_proprio_projector,
        resize_image_for_policy,  # Added: CRITICAL for image preprocessing
    )
    from experiments.robot.robot_utils import (
        normalize_gripper_action,
        invert_gripper_action,
        set_seed_everywhere,
        get_action,               # Added: official wrapper for action prediction
        get_image_resize_size,    # Added: CRITICAL for getting correct image size
    )
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        check_unnorm_key,         # Added: official unnorm_key verification
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    try:
        from src.orchestra_sdk.experiment import Experiment
        use_experiment_tracking = True
    except ImportError:
        print("⚠️ Orchestra SDK not available, skipping experiment tracking")
        use_experiment_tracking = False

    print("=" * 80)
    print(f"🚀 Phase 3: LIBERO Evaluation (FIXED v2) - {task_suite}")
    print(f"   Speculative Decoding: {'ENABLED' if use_speculative_decoding else 'DISABLED'}")
    print("=" * 80)

    # 🆕 Set random seed for reproducibility (官方代码 Line 468)
    SEED = 7
    set_seed_everywhere(SEED)
    print(f"\n✓ Random seed set to {SEED} for reproducibility")

    exp = None
    if use_experiment_tracking:
        try:
            exp = Experiment.init(
                name=f"Phase 3 - LIBERO {task_suite} (FIXED v2)",
                description=f"Official openvla-oft baseline",
                config={
                    "task_suite": task_suite,
                    "num_trials": num_trials,
                    "use_speculative_decoding": use_speculative_decoding,
                    "gpu_type": "a100",
                }
            )
            exp.add_tags(['phase3', 'evaluation', 'libero', 'fixed_v2'])
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
            model_family="openvla",          # Explicitly specify model family
            use_l1_regression=True,          # 👈 关键！使用 L1 regression
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,           # LIBERO uses 2 images
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,                # 👈 重要！center crop
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            task_suite_name=task_suite,      # Set task_suite_name for unnorm_key verification
        )

        # Load VLA model using official function
        vla = get_vla(cfg)
        processor = get_processor(cfg)

        # Load MLP action head (for L1 regression)
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

        # Load proprio projector
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

        # ==============================================================================
        # 🛠️ Ensure norm_stats are available (with fallback to manual injection)
        # ==============================================================================
        # Strategy: Try using model's norm_stats, fallback to manual injection if needed

        # First, check if model already has norm_stats
        if not hasattr(vla, 'norm_stats'):
            vla.norm_stats = {}
            print("   ⚠️  Model doesn't have norm_stats, initializing empty dict")

        # Check if the required key exists
        expected_keys = ["libero_spatial", "libero_spatial_no_noops"]
        has_stats = any(key in vla.norm_stats for key in expected_keys)

        if not has_stats:
            print("   ⚠️  Model missing LIBERO norm_stats, injecting manually...")
            # Manual injection as fallback (from official HuggingFace model)
            # Source: https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial/resolve/main/dataset_statistics.json
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

        # Now use official check_unnorm_key to verify and set the correct key
        check_unnorm_key(cfg, vla)

        print(f"   ✓ OpenVLA-OFT loaded with action_head and proprio_projector")
        print(f"   ✓ Using L1 regression: {cfg.use_l1_regression}")
        print(f"   ✓ Using unnorm_key: {cfg.unnorm_key}")

        # CRITICAL: Get image resize size for preprocessing
        resize_size = get_image_resize_size(cfg)  # Returns 224 for openvla
        print(f"   ✓ Image resize size: {resize_size}")

        # ============================================================
        # 官方的 LIBERO 辅助函数
        # ============================================================

        def get_libero_image(obs):
            """Extracts third-person image and rotates 180 degrees (CRITICAL!)"""
            img = obs["agentview_image"]
            img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
            return img

        def get_libero_wrist_image(obs):
            """Extracts wrist camera image and rotates 180 degrees"""
            img = obs["robot0_eye_in_hand_image"]
            img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
            return img

        def prepare_observation(obs):
            """
            Prepare observation dict for policy input (官方方式)

            CRITICAL: Resizes images from 256x256 (env) to 224x224 (model input)
            """
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)

            # CRITICAL: Resize images to match training distribution
            # This uses JPEG encode/decode + lanczos3 interpolation to match training preprocessing
            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

            # 构建 observation dict
            observation = {
                "full_image": img_resized,       # Resized to 224x224 for model
                "wrist_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            return observation, img  # 返回 processed obs 和 原始图像用于 replay

        def process_action(action):
            """Process action before sending to environment (官方方式)"""
            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)

            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            action = invert_gripper_action(action)

            return action

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

            # 初始化环境
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(0)  # IMPORTANT: seed affects object positions

            init_states = task_suite_obj.get_task_init_states(task_id)
            task_successes = 0
            task_episodes = 0

            for trial_idx in range(min(num_trials, len(init_states))):
                task_episodes += 1
                total_episodes += 1

                # Reset environment
                env.reset()
                obs = env.set_init_state(init_states[trial_idx])

                # 初始化 action queue (官方方式)
                from collections import deque
                action_queue = deque(maxlen=cfg.num_open_loop_steps)

                episode_success = False
                max_steps = 220  # libero_spatial max steps

                for step in range(max_steps + 10):  # 10 steps for stabilization
                    try:
                        # 前10步什么都不做，让物体稳定
                        if step < 10:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            continue

                        # Prepare observation (with image resize)
                        observation, img = prepare_observation(obs)

                        # Verify image shapes on first step of first trial
                        if step == 10 and trial_idx == 0:
                            print(f"   🔍 Image shapes after resize:")
                            print(f"      full_image: {observation['full_image'].shape}")
                            print(f"      wrist_image: {observation['wrist_image'].shape}")
                            print(f"      state: {observation['state'].shape}")

                        # 如果 action queue 是空的，requery model
                        if len(action_queue) == 0:
                            # 🔥 使用官方的 get_action 包装器（添加 torch.no_grad）
                            actions = get_action(
                                cfg, vla, observation, task_description,
                                processor=processor,
                                action_head=action_head,
                                proprio_projector=proprio_projector,
                                use_film=cfg.use_film,
                            )
                            action_queue.extend(actions)

                            # 第一步打印 action chunk
                            if step == 10:
                                print(f"   🔍 Action chunk returned {len(actions)} actions")
                                for i, act in enumerate(actions[:3]):
                                    print(f"      Row {i}: [{act[0]:.3f}, {act[1]:.3f}, {act[2]:.3f}, {act[3]:.3f}, {act[4]:.3f}, {act[5]:.3f}, {act[6]:.3f}]")

                        # Get action from queue
                        action = action_queue.popleft()

                        # Process action (官方方式)
                        action = process_action(action)

                        # 第一步打印 processed action
                        if step == 10:
                            print(f"   🔍 Processed action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}, {action[6]:.3f}]")

                        # Execute action in environment
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

        results = {
            'task_suite': task_suite,
            'use_speculative_decoding': use_speculative_decoding,
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'final_success_rate': final_success_rate,
            'task_results': task_results,
            'config': {
                'seed': SEED,
                'unnorm_key': cfg.unnorm_key,     # Record which unnorm_key was used
                'resize_size': resize_size,        # Record image resize size
                'model': cfg.pretrained_checkpoint,
            }
        }

        print(f"\n{'='*80}")
        print(f"🎉 EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"   Success Rate: {final_success_rate:.1%}")
        print(f"{'='*80}")

        if exp:
            try:
                exp.log({"final_success_rate": final_success_rate})
                exp.finish(status="completed")
            except:
                pass

        results_path = f"/results/{task_suite}_baseline_fixed_v2.json"
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
    use_speculative_decoding: bool = False,
):
    print(f"🚀 Starting Phase 3 LIBERO Evaluation (FIXED v2)...")
    print(f"   Task Suite: {task_suite}")
    print(f"   Num Trials: {num_trials}")
    print(f"   Mode: BASELINE (official openvla-oft code)")

    results = run_libero_evaluation.remote(
        task_suite=task_suite,
        num_trials=num_trials,
        use_speculative_decoding=use_speculative_decoding,
    )

    print(f"\n✅ Evaluation complete!")
    print(f"   Success Rate: {results['final_success_rate']:.1%}")

    return results
