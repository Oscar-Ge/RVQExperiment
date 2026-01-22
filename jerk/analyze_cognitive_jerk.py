"""
Cognitive Jerk Analysis Script for LIBERO-Spatial Task 0

This script tracks and visualizes cognitive jerk (hidden state changes) during episode execution.

Key Innovation: Dense Prediction Mode
- Runs 7B backbone at EVERY step to extract hidden states (heavy compute)
- Runs action head at EVERY step to get "hypothetical" action plans (cheap compute)
- This decouples monitoring frequency (every step) from control frequency (every 8 steps)
- Enables detection of when the model "changes its mind" between action chunks

Metrics tracked (all DENSE - computed every step):
1. Cognitive Jerk: Cosine distance between consecutive 4096-dim hidden states
2. Action Plan Instability: Temporal consistency of overlapping plan regions
3. Gripper Events: Track gripper open/close moments (task-critical events)

Output:
- jerk_analysis_task0.png: 4-panel time-series visualization
- jerk_metrics_task0.json: Raw data for later analysis

Usage:
    modal run analyze_cognitive_jerk.py
"""

import os
import sys
import modal
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from collections import deque
import numpy as np

# ============================================================
# Modal App Setup
# ============================================================
app = modal.App("cognitive-jerk-analysis")

# Volumes
data_volume = modal.Volume.from_name("rsd-libero-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("rsd-results", create_if_missing=True)

# Get SDK path
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

# Build evaluation image (same as phase3)
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
        # Install prismatic from openvla-oft repo
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


# ============================================================
# Metric Computation Functions
# ============================================================

def compute_cosine_distance(h1, h2):
    """
    Compute cosine distance between consecutive hidden states.

    Args:
        h1, h2: torch.Tensor [1, 4096]

    Returns:
        float: cosine distance in [0, 2]
    """
    import torch.nn.functional as F
    cos_sim = F.cosine_similarity(h1, h2, dim=-1)
    return (1.0 - cos_sim).item()


def compute_action_change(prev_chunk, curr_chunk):
    """
    Compute L2 norm of action chunk difference.

    This function is kept for backward compatibility but is not used in the
    main loop. Instead, we compute temporal consistency directly in the loop.

    Args:
        prev_chunk, curr_chunk: np.ndarray [N, 7]

    Returns:
        float: Frobenius norm
    """
    diff = curr_chunk - prev_chunk
    return np.linalg.norm(diff)


def detect_gripper_event(current_action, previous_action, threshold=0.0):
    """
    Detect gripper state transitions.

    Args:
        current_action: np.ndarray [7]
        previous_action: np.ndarray [7] or None
        threshold: float (0.0 for normalized gripper in [-1, 1])
            - Typically: -1 = fully open, +1 = fully closed
            - Threshold 0.0 separates open (<0) from closed (>0)

    Returns:
        str or None: 'closing', 'opening', or None
    """
    if previous_action is None:
        return None

    # Gripper is index 6, normalized to [-1, 1]
    # OpenVLA/LIBERO convention: -1 = open, +1 = closed
    curr_gripper = current_action[6]
    prev_gripper = previous_action[6]

    # Detect transitions (>threshold = closed, <threshold = open)
    curr_closed = curr_gripper > threshold
    prev_closed = prev_gripper > threshold

    if curr_closed and not prev_closed:
        return 'closing'
    elif not curr_closed and prev_closed:
        return 'opening'
    return None


# ============================================================
# Visualization Function
# ============================================================

def create_jerk_analysis_plot(episode_metrics, task_description, success, output_path='jerk_analysis_task0.png'):
    """
    Create 4-panel visualization:
    1. Cognitive Jerk time series (dense)
    2. Action Plan Instability (dense - temporal consistency metric)
    3. Gripper state with events (dense)
    4. Episode timeline with prediction markers
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    timesteps = episode_metrics['timesteps']
    prediction_steps = episode_metrics['prediction_steps']

    # Panel 1: Cognitive Jerk
    ax1 = axes[0]
    ax1.plot(timesteps, episode_metrics['cognitive_jerk'],
             linewidth=2, color='#2E86AB', alpha=0.8)

    # Highlight high jerk (>90th percentile)
    if len(episode_metrics['cognitive_jerk']) > 0:
        jerk_threshold = np.percentile(episode_metrics['cognitive_jerk'], 90)
        high_jerk_mask = np.array(episode_metrics['cognitive_jerk']) > jerk_threshold
        high_jerk_steps = np.array(timesteps)[high_jerk_mask]
        high_jerk_values = np.array(episode_metrics['cognitive_jerk'])[high_jerk_mask]
        ax1.scatter(high_jerk_steps, high_jerk_values,
                    color='red', s=50, zorder=5, alpha=0.6, label='High Jerk')

    # Mark prediction timesteps
    for pred_step in prediction_steps:
        ax1.axvline(pred_step, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    ax1.set_ylabel('Cognitive Jerk\n(Cosine Distance)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Cognitive Jerk Analysis: {task_description}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    if len(episode_metrics['cognitive_jerk']) > 0:
        ax1.legend(loc='upper right')

    # Panel 2: Action Plan Instability (Dense - every step)
    ax2 = axes[1]
    if len(episode_metrics['action_change']) > 0:
        # Dense metric: aligned with timesteps (offset by 1 since we need previous plan)
        change_steps = timesteps[1:len(episode_metrics['action_change'])+1]
        ax2.plot(change_steps, episode_metrics['action_change'],
                 linewidth=2, color='#A23B72', alpha=0.8)
        ax2.fill_between(change_steps, 0, episode_metrics['action_change'],
                         alpha=0.2, color='#A23B72')

        # Highlight high instability (>90th percentile)
        instability_threshold = np.percentile(episode_metrics['action_change'], 90)
        high_instability_mask = np.array(episode_metrics['action_change']) > instability_threshold
        high_instability_steps = np.array(change_steps)[high_instability_mask]
        high_instability_values = np.array(episode_metrics['action_change'])[high_instability_mask]
        ax2.scatter(high_instability_steps, high_instability_values,
                    color='red', s=50, zorder=5, alpha=0.6, label='High Instability')

        # Mark prediction timesteps
        for pred_step in prediction_steps:
            ax2.axvline(pred_step, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        if len(high_instability_steps) > 0:
            ax2.legend(loc='upper right')

    ax2.set_ylabel('Action Instability\n(Temporal Consistency)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Gripper State
    ax3 = axes[2]
    ax3.plot(timesteps, episode_metrics['gripper_state'],
             linewidth=2, color='#F18F01', alpha=0.8)
    ax3.axhline(0.0, color='black', linestyle=':', alpha=0.5, linewidth=1)

    # Annotate gripper events
    closing_plotted = False
    opening_plotted = False
    for i, (step, event) in enumerate(zip(timesteps, episode_metrics['gripper_events'])):
        if event == 'closing':
            label = 'Closing' if not closing_plotted else None
            ax3.scatter(step, episode_metrics['gripper_state'][i],
                       marker='v', s=100, color='red', zorder=5, label=label)
            closing_plotted = True
        elif event == 'opening':
            label = 'Opening' if not opening_plotted else None
            ax3.scatter(step, episode_metrics['gripper_state'][i],
                       marker='^', s=100, color='green', zorder=5, label=label)
            opening_plotted = True

    if closing_plotted or opening_plotted:
        ax3.legend(loc='upper right')

    ax3.set_ylabel('Gripper State\n(Normalized)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Timeline
    ax4 = axes[3]
    if len(timesteps) > 0:
        ax4.barh(0, len(timesteps), left=timesteps[0], height=0.5,
                 color='lightblue', alpha=0.3, edgecolor='black')

        for pred_step in prediction_steps:
            ax4.scatter(pred_step, 0, marker='|', s=500, linewidths=3,
                       color='purple', zorder=5)

        outcome_color = 'green' if success else 'red'
        outcome_text = 'SUCCESS' if success else 'FAILURE'
        ax4.scatter(timesteps[-1], 0, marker='*', s=500,
                   color=outcome_color, zorder=10, edgecolors='black', linewidths=2)
        ax4.text(timesteps[-1], 0.3, outcome_text,
                fontsize=12, fontweight='bold', color=outcome_color, ha='center')

    ax4.set_yticks([])
    ax4.set_ylim(-1, 1)
    ax4.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Timeline', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to {output_path}")


# ============================================================
# Main Analysis Function
# ============================================================

@app.function(
    image=eval_image,
    gpu="A100",
    timeout=3600,
    volumes={
        "/data": data_volume,
        "/hf_cache": hf_cache,
        "/results": results_volume,
    },
)
def run_cognitive_jerk_analysis():
    """Run Task 0, 1 trial with cognitive jerk tracking"""
    import torch
    import sys
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/openvla-oft")

    from experiments.robot.libero.libero_utils import (
        quat2axisangle,
        get_libero_dummy_action,
    )
    from experiments.robot.openvla_utils import (
        get_vla,
        get_processor,
        resize_image_for_policy,
    )
    from experiments.robot.robot_utils import (
        normalize_gripper_action,
        invert_gripper_action,
        set_seed_everywhere,
        get_image_resize_size,
    )
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
    )

    # Import LIBERO after path setup
    import libero.libero.benchmark as benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image as PILImage

    print("\n" + "="*80)
    print("COGNITIVE JERK ANALYSIS - LIBERO-Spatial Task 0")
    print("="*80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    set_seed_everywhere(42)

    # ============================================================
    # 1. Load OpenVLA-OFT
    # ============================================================
    print("\n📦 Loading OpenVLA-OFT model...")
    cfg = GenerateConfig(
        pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
        model_family="openvla",
        use_l1_regression=True,
        num_images_in_input=2,
        use_proprio=True,
        task_suite_name="libero_spatial",
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    vla.eval()
    print("✅ Model loaded successfully")

    # ============================================================
    # 2. Initialize LIBERO Task 0
    # ============================================================
    print("\n🎮 Initializing LIBERO environment...")
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(0)  # Only Task 0
    task_description = task.language
    init_states = task_suite.get_task_init_states(0)
    task_bddl_file = os.path.join(
        benchmark.get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    print(f"Task: {task_description}")
    print(f"BDDL file: {task_bddl_file}")

    # ============================================================
    # 3. Helper Functions
    # ============================================================

    def get_libero_image(obs):
        """Extracts third-person image and rotates 180 degrees (CRITICAL!)"""
        img = obs["agentview_image"]
        img = img[::-1, ::-1]  # CRITICAL: rotate 180 degrees
        return np.ascontiguousarray(img)

    def get_libero_wrist_image(obs):
        """Extracts wrist camera image and rotates 180 degrees"""
        img = obs["robot0_eye_in_hand_image"]
        img = img[::-1, ::-1]  # CRITICAL: rotate 180 degrees
        return np.ascontiguousarray(img)

    def prepare_observation(obs):
        """Prepare observation dict for policy input (official method)"""
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)

        # CRITICAL: Resize images using official method
        resize_size = get_image_resize_size(cfg)
        img_resized = resize_image_for_policy(img, resize_size)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        return observation

    def process_action(action):
        """Process action before sending to environment (official method)"""
        # Normalize gripper action [0,1] -> [-1,+1]
        action = normalize_gripper_action(action, binarize=True)
        # Invert gripper action (-1 = open, +1 = close)
        action = invert_gripper_action(action)
        return action

    def extract_hidden_state(observation, task_description):
        """
        Extract 4096-dim hidden state from OpenVLA at every timestep.

        Pattern from phase2_conditional/modal_train_phase2_complete.py:334-342
        and phase3/modal_phase3_libero_eval_CONDITIONAL_FIXED.py:671-718
        """
        try:
            # Convert to PIL images
            full_img = observation['full_image']
            wrist_img = observation['wrist_image']
            state = observation['state']

            if isinstance(full_img, np.ndarray):
                full_img = PILImage.fromarray(full_img.astype(np.uint8))
            if isinstance(wrist_img, np.ndarray):
                wrist_img = PILImage.fromarray(wrist_img.astype(np.uint8))

            # Prepare inputs with official processor
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            inputs = processor(prompt, full_img).to(device, dtype=torch.bfloat16)

            # Add wrist image if using 2 cameras
            if cfg.num_images_in_input == 2:
                wrist_inputs = processor(prompt, wrist_img).to(device, dtype=torch.bfloat16)
                inputs["pixel_values"] = torch.cat([
                    inputs["pixel_values"],
                    wrist_inputs["pixel_values"]
                ], dim=1)

            # Add proprio state
            if cfg.use_proprio:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                inputs["proprio"] = state_tensor

            # Forward pass with hidden state extraction
            with torch.no_grad():
                outputs = vla(**inputs, output_hidden_states=True)
                hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()  # [1, 4096]

            return hidden_4096

        except Exception as e:
            print(f"   ⚠️ Hidden state extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_actions(observation, task_description):
        """Predict action chunk from observation"""
        try:
            full_img = observation['full_image']
            wrist_img = observation['wrist_image']
            state = observation['state']

            if isinstance(full_img, np.ndarray):
                full_img = PILImage.fromarray(full_img.astype(np.uint8))
            if isinstance(wrist_img, np.ndarray):
                wrist_img = PILImage.fromarray(wrist_img.astype(np.uint8))

            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            inputs = processor(prompt, full_img).to(device, dtype=torch.bfloat16)

            if cfg.num_images_in_input == 2:
                wrist_inputs = processor(prompt, wrist_img).to(device, dtype=torch.bfloat16)
                inputs["pixel_values"] = torch.cat([
                    inputs["pixel_values"],
                    wrist_inputs["pixel_values"]
                ], dim=1)

            if cfg.use_proprio:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                inputs["proprio"] = state_tensor

            with torch.no_grad():
                action_result = vla.predict_action(
                    **inputs,
                    unnorm_key="bridge_orig",
                    do_sample=False
                )

                if isinstance(action_result, tuple):
                    actions = action_result[0]
                else:
                    actions = action_result

            return actions

        except Exception as e:
            print(f"   ⚠️ Action prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ============================================================
    # 4. Run Episode with Metric Tracking
    # ============================================================
    print("\n🚀 Starting episode execution...")

    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.reset()
    obs = env.set_init_state(init_states[0])  # Only 1 trial

    # Initialize tracking
    episode_metrics = {
        'timesteps': [],
        'cognitive_jerk': [],
        'action_change': [],  # Dense: computed every step
        'gripper_state': [],
        'gripper_events': [],
        'prediction_steps': [],  # Sparse: when queue refilled
    }

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    previous_hidden_state = None
    previous_action_plan = None  # Full plan from previous step
    previous_action = None  # Actual action executed
    episode_success = False
    max_steps = 220

    print(f"Max steps: {max_steps}")
    print(f"Action chunk size: {cfg.num_open_loop_steps}")
    print(f"✨ Dense prediction mode: Predicting actions at EVERY step for plan stability analysis")

    for step in range(max_steps + 10):
        try:
            # First 10 steps: stabilization (reset state to avoid measuring jerk from dummy actions)
            if step < 10:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                # Reset tracking variables at the end of stabilization
                if step == 9:
                    previous_hidden_state = None
                    previous_action_plan = None
                    previous_action = None
                    print("   🔄 Stabilization complete, resetting tracking variables")
                continue

            # Prepare observation
            observation = prepare_observation(obs)

            # ✅ ALWAYS extract hidden state (Heavy compute - 7B backbone)
            hidden_state = extract_hidden_state(observation, task_description)

            if hidden_state is None:
                print(f"   ⚠️ Hidden state extraction failed at step {step}")
                break

            # ✅ ALWAYS predict actions (Cheap - lightweight action head)
            # This gives us "hypothetical" actions to see what the model is "thinking"
            current_action_plan = predict_actions(observation, task_description)

            if current_action_plan is None:
                print(f"   ⚠️ Action prediction failed at step {step}")
                break

            # Convert to numpy for consistent handling
            if not isinstance(current_action_plan, np.ndarray):
                current_action_plan = np.array(current_action_plan)

            # === Compute Metrics (Dense - every step) ===

            # 1. Cognitive Jerk
            if previous_hidden_state is not None:
                jerk = compute_cosine_distance(previous_hidden_state, hidden_state)
                episode_metrics['cognitive_jerk'].append(jerk)
                episode_metrics['timesteps'].append(step)

                if step == 11:
                    print(f"   🔍 First cognitive jerk: {jerk:.6f}")

            # 2. Action Plan Instability (Temporal Consistency)
            # Compare current_plan[0] with previous_plan[1] (shifted comparison)
            if previous_action_plan is not None:
                # Temporal consistency: did the plan change from last step?
                # Compare overlapping region: current_plan[:-1] vs previous_plan[1:]
                overlap_len = min(len(current_action_plan) - 1, len(previous_action_plan) - 1)
                if overlap_len > 0:
                    current_overlap = current_action_plan[:overlap_len]
                    prev_overlap = previous_action_plan[1:overlap_len+1]
                    instability = np.linalg.norm(current_overlap - prev_overlap)
                    episode_metrics['action_change'].append(instability)

                    if len(episode_metrics['action_change']) == 1:
                        print(f"   🔍 First action instability: {instability:.6f}")

            # Update previous state
            previous_hidden_state = hidden_state
            previous_action_plan = current_action_plan.copy()

            # === Control Logic (Sparse - only when queue empty) ===
            # Only refill queue when empty to respect the control frequency
            if len(action_queue) == 0:
                action_queue.extend(current_action_plan)
                episode_metrics['prediction_steps'].append(step)

                if step == 10:
                    print(f"   🔍 Action chunk size: {len(current_action_plan)}")
                    print(f"   🔍 Refilling action queue")

            # Get action from queue
            action = action_queue.popleft()

            # Process action (normalize + invert gripper)
            action_processed = process_action(action)

            # Track gripper state and events
            if len(episode_metrics['timesteps']) > 0:  # Only if we have hidden state data
                episode_metrics['gripper_state'].append(action_processed[6])
                gripper_event = detect_gripper_event(action_processed, previous_action)
                episode_metrics['gripper_events'].append(gripper_event)

                # Diagnostic: verify gripper normalization at first step
                if step == 10:
                    print(f"   🔍 First gripper value: {action_processed[6]:.3f} (expecting range [-1, 1])")

                if gripper_event is not None:
                    print(f"   🤏 Gripper event at step {step}: {gripper_event}")

            # Step environment
            obs, reward, done, info = env.step(action_processed.tolist())

            if done:
                episode_success = True
                print(f"   ✅ Task completed at step {step}")
                break

            previous_action = action_processed

        except Exception as step_error:
            print(f"   ⚠️ Step error at step {step}: {step_error}")
            import traceback
            traceback.print_exc()
            break

    env.close()

    # ============================================================
    # 5. Save Raw Data
    # ============================================================
    print("\n💾 Saving raw metrics data...")
    import json

    # Save as JSON for easy inspection
    metrics_serializable = {
        'timesteps': [int(t) for t in episode_metrics['timesteps']],
        'cognitive_jerk': [float(j) for j in episode_metrics['cognitive_jerk']],
        'action_change': [float(a) for a in episode_metrics['action_change']],
        'gripper_state': [float(g) for g in episode_metrics['gripper_state']],
        'gripper_events': [e if e is None else str(e) for e in episode_metrics['gripper_events']],
        'prediction_steps': [int(p) for p in episode_metrics['prediction_steps']],
        'task_description': task_description,
        'success': episode_success,
    }

    with open('/results/jerk_metrics_task0.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    # Also save to /tmp
    with open('/tmp/jerk_metrics_task0.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    print("   ✅ Raw data saved to jerk_metrics_task0.json")

    # ============================================================
    # 6. Generate Visualization
    # ============================================================
    print("\n📊 Generating visualization...")
    create_jerk_analysis_plot(episode_metrics, task_description, episode_success, '/results/jerk_analysis_task0.png')

    # Also save a copy in /tmp for easier retrieval
    create_jerk_analysis_plot(episode_metrics, task_description, episode_success, '/tmp/jerk_analysis_task0.png')

    # ============================================================
    # 7. Print Summary Statistics
    # ============================================================
    print("\n" + "="*80)
    print("COGNITIVE JERK ANALYSIS SUMMARY")
    print("="*80)
    print(f"Task: {task_description}")
    print(f"Success: {'✅ YES' if episode_success else '❌ NO'}")
    print(f"Total steps: {len(episode_metrics['timesteps'])}")
    print(f"Prediction steps (queue refills): {len(episode_metrics['prediction_steps'])}")
    print(f"Dense metrics: {len(episode_metrics['cognitive_jerk'])} cognitive jerk + {len(episode_metrics['action_change'])} action change samples")

    if len(episode_metrics['cognitive_jerk']) > 0:
        print(f"\nCognitive Jerk Statistics:")
        print(f"  Mean: {np.mean(episode_metrics['cognitive_jerk']):.6f}")
        print(f"  Std:  {np.std(episode_metrics['cognitive_jerk']):.6f}")
        print(f"  Min:  {np.min(episode_metrics['cognitive_jerk']):.6f}")
        print(f"  Max:  {np.max(episode_metrics['cognitive_jerk']):.6f}")
        print(f"  90th percentile: {np.percentile(episode_metrics['cognitive_jerk'], 90):.6f}")

    if len(episode_metrics['action_change']) > 0:
        print(f"\nAction Plan Instability (Temporal Consistency):")
        print(f"  Mean: {np.mean(episode_metrics['action_change']):.6f}")
        print(f"  Std:  {np.std(episode_metrics['action_change']):.6f}")
        print(f"  Min:  {np.min(episode_metrics['action_change']):.6f}")
        print(f"  Max:  {np.max(episode_metrics['action_change']):.6f}")
        print(f"  90th percentile: {np.percentile(episode_metrics['action_change'], 90):.6f}")
        print(f"  Note: Low values = stable plan, high values = model 'changing mind'")

    closing_events = sum(1 for e in episode_metrics['gripper_events'] if e == 'closing')
    opening_events = sum(1 for e in episode_metrics['gripper_events'] if e == 'opening')
    print(f"\nGripper Events:")
    print(f"  Closing: {closing_events}")
    print(f"  Opening: {opening_events}")
    print(f"  Total transitions: {closing_events + opening_events}")

    # Correlation analysis (if both metrics have data)
    if len(episode_metrics['cognitive_jerk']) > 0 and len(episode_metrics['action_change']) > 0:
        # Align metrics (action_change is offset by 1)
        min_len = min(len(episode_metrics['cognitive_jerk']) - 1, len(episode_metrics['action_change']))
        if min_len > 1:
            jerk_aligned = episode_metrics['cognitive_jerk'][1:min_len+1]
            instability_aligned = episode_metrics['action_change'][:min_len]

            # Compute correlation
            correlation = np.corrcoef(jerk_aligned, instability_aligned)[0, 1]

            print(f"\n📈 Correlation Analysis:")
            print(f"  Cognitive Jerk ↔ Action Instability: {correlation:.4f}")
            if correlation > 0.5:
                print(f"  ✅ Strong positive correlation! High jerk → plan instability")
            elif correlation > 0.3:
                print(f"  ✓ Moderate correlation - jerk and instability tend to co-occur")
            elif correlation > 0:
                print(f"  Weak positive correlation")
            else:
                print(f"  ⚠️ No significant correlation detected")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("📊 Outputs:")
    print("  - jerk_analysis_task0.png (visualization)")
    print("  - jerk_metrics_task0.json (raw data)")
    print("="*80)

    return episode_metrics


# ============================================================
# Entry Point
# ============================================================

@app.local_entrypoint()
def main():
    """Entry point for modal run"""
    print("Starting cognitive jerk analysis...")
    results = run_cognitive_jerk_analysis.remote()
    print("\n✅ Analysis complete!")
    print("📊 Check /results/jerk_analysis_task0.png for the visualization")
