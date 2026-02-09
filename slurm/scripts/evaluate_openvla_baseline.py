#!/usr/bin/env python3
"""
OpenVLA Baseline Evaluation on LIBERO

Evaluates vanilla OpenVLA-OFT model on LIBERO benchmark WITHOUT RSD.
Records ALL episodes as videos and saves ONLY failure cases for analysis.

Key features:
- Load OpenVLA-OFT model only (no RSD components)
- Run on LIBERO benchmark tasks
- Record all episodes using VideoRecorder
- Filter and save only failure case videos to results/baseline_failures/
- Compute success rate and save detailed episode logs
- Output: baseline_results.json + failure videos

Usage:
    python evaluate_openvla_baseline.py --task-suite libero_spatial --num-trials 50
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional, List
import numpy as np
import torch

# Add LIBERO to path
sys.path.insert(0, os.getenv('LIBERO_PATH', '/root/LIBERO'))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.experiment_logger import ExperimentLogger
from utils.video_recorder import VideoRecorder


class OpenVLABaselinePolicy:
    """Wrapper for OpenVLA-OFT baseline policy"""

    def __init__(
        self,
        vla_model,
        processor,
        action_head,
        proprio_projector,
        cfg,
        device='cuda',
        resize_size=224,
    ):
        self.vla = vla_model
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.cfg = cfg
        self.device = device
        self.resize_size = resize_size
        self.chunk_len = 8

        self.stats = {
            'total_predictions': 0,
            'inference_time_ms': 0.0,
            'fallback_count': 0,
        }

    def predict_action_chunk(self, observation, task_description):
        """
        Predict action chunk using vanilla OpenVLA.

        Args:
            observation: dict with 'full_image', 'wrist_image', 'state'
            task_description: str

        Returns:
            actions: [8, 7] numpy array
        """
        from PIL import Image as PILImage

        with torch.no_grad():
            try:
                t0 = time.time()

                full_img = observation['full_image']
                wrist_img = observation['wrist_image']
                state = observation['state']

                # Convert to PIL Images if needed
                if isinstance(full_img, np.ndarray):
                    full_img = PILImage.fromarray(full_img.astype(np.uint8))
                if isinstance(wrist_img, np.ndarray):
                    wrist_img = PILImage.fromarray(wrist_img.astype(np.uint8))

                # Prepare inputs using official processor
                inputs = self.processor(task_description, full_img).to(self.device, dtype=torch.bfloat16)

                # Add wrist image if dual-camera setup
                if self.cfg.num_images_in_input == 2:
                    wrist_inputs = self.processor(task_description, wrist_img).to(self.device, dtype=torch.bfloat16)
                    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

                # Add proprio state if used
                if self.cfg.use_proprio:
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    inputs["proprio"] = state_tensor

                # Get VLA hidden states
                outputs = self.vla(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, 4096]

                # Use action_head to predict actions
                if self.cfg.use_l1_regression:
                    action_logits = self.action_head(last_hidden_state)  # [1, seq_len, chunk*7]
                    action_chunk = action_logits[0, -1, :]  # [chunk*7]
                    action_chunk = action_chunk.view(self.chunk_len, 7)  # [8, 7]
                    actions = action_chunk.cpu().numpy()
                else:
                    actions = np.zeros((self.chunk_len, 7), dtype=np.float32)

                # Update stats
                self.stats['inference_time_ms'] += (time.time() - t0) * 1000
                self.stats['total_predictions'] += 1

                return actions

            except Exception as e:
                print(f"   WARNING: OpenVLA prediction error: {e}")
                self.stats['fallback_count'] += 1
                self.stats['total_predictions'] += 1
                return np.zeros((self.chunk_len, 7), dtype=np.float32)

    def get_stats(self):
        """Get inference statistics"""
        total = self.stats['total_predictions']
        if total == 0:
            return {}

        return {
            'total_predictions': total,
            'avg_inference_time_ms': self.stats['inference_time_ms'] / total,
            'fallback_count': self.stats['fallback_count'],
            'fallback_rate': self.stats['fallback_count'] / total,
        }


def load_openvla_models(cfg, device):
    """Load OpenVLA-OFT model using official functions"""
    try:
        from experiments.robot.openvla_utils import (
            get_vla, get_processor, get_action_head, get_proprio_projector, resize_image_for_policy
        )
        from experiments.robot.robot_utils import get_image_resize_size
        from experiments.robot.libero.run_libero_eval import check_unnorm_key
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
    except ImportError:
        print("ERROR: OpenVLA utilities not found. Make sure openvla-oft is installed.")
        raise

    # Load VLA model
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
    resize_size = get_image_resize_size(cfg)

    # CRITICAL: Ensure norm_stats are available
    inject_libero_norm_stats_if_missing(vla, cfg.task_suite_name)

    # CRITICAL: Check and set unnorm_key
    check_unnorm_key(cfg, vla)

    return vla, processor, action_head, proprio_projector, resize_size


def inject_libero_norm_stats_if_missing(vla, task_suite):
    """Inject LIBERO normalization statistics if missing"""
    if not hasattr(vla, 'norm_stats'):
        vla.norm_stats = {}

    expected_keys = ["libero_spatial", "libero_spatial_no_noops"]
    has_stats = any(key in vla.norm_stats for key in expected_keys)

    if not has_stats:
        print("   WARNING: Model missing LIBERO norm_stats, injecting manually...")
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
        print("   INFO: Manually injected 'libero_spatial_no_noops' statistics")


def create_generate_config(task_suite: str, checkpoint: str):
    """Create GenerateConfig for OpenVLA"""
    try:
        from experiments.robot.libero.run_libero_eval import GenerateConfig
    except ImportError:
        # Fallback: create a simple config object
        class GenerateConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        model_family="openvla",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=8,  # NUM_ACTIONS_CHUNK
        task_suite_name=task_suite,
    )

    return cfg


def prepare_observation(obs, resize_fn, resize_size):
    """Prepare observation dict for policy input (official method)"""
    # Extract and rotate images (CRITICAL!)
    img = obs["agentview_image"][::-1, ::-1]
    wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]

    # Resize images using official method
    img_resized = resize_fn(img, resize_size)
    wrist_img_resized = resize_fn(wrist_img, resize_size)

    # Extract state
    try:
        from experiments.robot.libero.libero_utils import quat2axisangle
    except ImportError:
        # Fallback quat2axisangle
        def quat2axisangle(quat):
            return quat[1:4]

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
    try:
        from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
    except ImportError:
        # Fallback: simple processing
        action = action.copy()
        action[6] = 2.0 * action[6] - 1.0  # Normalize [0,1] -> [-1,1]
        action[6] = -action[6]  # Invert

    return action


def get_libero_dummy_action(model_family):
    """Get dummy action for stabilization period"""
    try:
        from experiments.robot.libero.libero_utils import get_libero_dummy_action as official_dummy
        return official_dummy(model_family)
    except ImportError:
        return np.zeros(7)


def run_evaluation(args):
    """Run LIBERO evaluation with OpenVLA baseline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"OpenVLA Baseline Evaluation - {args.task_suite}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Num Trials: {args.num_trials}")
    print(f"Record Videos: {args.record_videos}")
    print(f"Save All Videos: {args.save_all_videos}")
    print(f"{'='*80}\n")

    # CRITICAL: Set random seed for reproducibility
    try:
        from experiments.robot.robot_utils import set_seed_everywhere
        SEED = 7
        set_seed_everywhere(SEED)
        print(f"Random seed set to {SEED} for reproducibility\n")
    except ImportError:
        print("WARNING: set_seed_everywhere not available, using manual seed setting")
        import random
        SEED = 7
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        print(f"Random seed set to {SEED} for reproducibility\n")

    # Initialize experiment logger
    logger = ExperimentLogger(
        name=f"openvla_baseline_{args.task_suite}",
        log_dir=args.log_dir,
        config={
            'task_suite': args.task_suite,
            'num_trials': args.num_trials,
            'checkpoint': args.openvla_checkpoint,
            'record_videos': args.record_videos,
            'save_all_videos': args.save_all_videos,
        }
    )

    # Initialize video recorder
    video_recorder = None
    if args.record_videos:
        failure_video_dir = Path(args.results_dir) / "baseline_failures"
        all_video_dir = Path(args.results_dir) / "baseline_all"
        video_dir = all_video_dir if args.save_all_videos else failure_video_dir
        video_recorder = VideoRecorder(output_dir=str(video_dir), fps=20)
        logger.log_text(f"Video recorder initialized: {video_dir}")

    try:
        # 1. Load OpenVLA-OFT model
        logger.log_text("Loading OpenVLA-OFT model...")
        cfg = create_generate_config(args.task_suite, args.openvla_checkpoint)
        vla, processor, action_head, proprio_projector, resize_size = load_openvla_models(cfg, device)
        logger.log_text(f"OpenVLA loaded: {args.openvla_checkpoint}")
        logger.log_text(f"Image resize size: {resize_size}")

        # Import resize function
        try:
            from experiments.robot.openvla_utils import resize_image_for_policy
        except ImportError:
            def resize_image_for_policy(img, size):
                from PIL import Image
                return np.array(Image.fromarray(img).resize((size, size)))

        # 2. Initialize baseline policy
        logger.log_text("Initializing OpenVLA Baseline Policy...")
        policy = OpenVLABaselinePolicy(
            vla_model=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            cfg=cfg,
            device=device,
            resize_size=resize_size,
        )

        # 3. Initialize LIBERO
        logger.log_text(f"Initializing LIBERO {args.task_suite}...")
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_obj = benchmark_dict[args.task_suite]()
        num_tasks = task_suite_obj.n_tasks
        logger.log_text(f"Number of tasks: {num_tasks}")

        # 4. Run Evaluation
        total_episodes = 0
        total_successes = 0
        task_results = []
        episode_logs = []

        for task_id in range(num_tasks):
            task = task_suite_obj.get_task(task_id)
            task_description = task.language

            logger.log_text(f"\n{'='*80}")
            logger.log_text(f"Task {task_id + 1}/{num_tasks}: {task_description}")
            logger.log_text(f"{'='*80}")

            # Initialize environment
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(0)

            init_states = task_suite_obj.get_task_init_states(task_id)
            task_successes = 0
            task_episodes = 0

            for trial_idx in range(min(args.num_trials, len(init_states))):
                task_episodes += 1
                total_episodes += 1

                # Start video recording if enabled
                recording_name = f"task{task_id}_trial{trial_idx}"
                if args.record_videos:
                    video_recorder.start_recording(recording_name)

                # Reset environment
                env.reset()
                obs = env.set_init_state(init_states[trial_idx])

                # Initialize action queue
                action_queue = deque(maxlen=cfg.num_open_loop_steps)

                episode_success = False
                episode_steps = 0
                max_steps = 220

                for step in range(max_steps + 10):
                    try:
                        # First 10 steps: stabilization
                        if step < 10:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))

                            # Record frame
                            if args.record_videos:
                                frame = obs["agentview_image"]
                                video_recorder.add_frame(frame)
                            continue

                        # Prepare observation
                        observation = prepare_observation(obs, resize_image_for_policy, resize_size)

                        # Query model if action queue is empty
                        if len(action_queue) == 0:
                            actions = policy.predict_action_chunk(observation, task_description)
                            action_queue.extend(actions)

                        # Get action from queue
                        action = action_queue.popleft()

                        # Process action
                        action = process_action(action)

                        # Execute action
                        obs, reward, done, info = env.step(action.tolist())
                        episode_steps = step

                        # Record frame
                        if args.record_videos:
                            frame = obs["agentview_image"]
                            video_recorder.add_frame(frame)

                        if done:
                            episode_success = True
                            break

                    except Exception as step_error:
                        logger.log_text(f"Step error: {step_error}")
                        break

                env.close()

                # Save video based on success/failure
                if args.record_videos:
                    if args.save_all_videos:
                        # Save all videos
                        video_filename = f"{recording_name}_{'success' if episode_success else 'failure'}.mp4"
                        video_recorder.save_video(video_filename)
                    else:
                        # Save only failures
                        if not episode_success:
                            video_filename = f"{recording_name}_failure.mp4"
                            video_recorder.save_video(video_filename)
                            logger.log_text(f"Saved failure video: {video_filename}")
                        else:
                            video_recorder.discard()

                # Log episode
                episode_log = {
                    'task_id': task_id,
                    'task_description': task_description,
                    'trial_idx': trial_idx,
                    'success': episode_success,
                    'steps': episode_steps,
                }
                episode_logs.append(episode_log)

                if episode_success:
                    task_successes += 1
                    total_successes += 1

                logger.log_text(f"Trial {trial_idx + 1}: {'SUCCESS' if episode_success else 'FAILED'} ({episode_steps} steps)")

            task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
            task_results.append({
                'task_id': task_id,
                'task_description': task_description,
                'successes': task_successes,
                'episodes': task_episodes,
                'success_rate': task_success_rate,
            })

            logger.log_text(f"Task Success Rate: {task_success_rate:.1%} ({task_successes}/{task_episodes})")
            logger.log({'task_success_rate': task_success_rate}, step=task_id)

        # 5. Final Results
        final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
        policy_stats = policy.get_stats()

        results = {
            'task_suite': args.task_suite,
            'model_type': 'openvla_baseline',
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'final_success_rate': final_success_rate,
            'task_results': task_results,
            'episode_logs': episode_logs,
            'policy_stats': policy_stats,
            'config': {
                'checkpoint': args.openvla_checkpoint,
                'resize_size': resize_size,
                'unnorm_key': cfg.unnorm_key,
                'num_trials': args.num_trials,
            }
        }

        # Print summary table
        print(f"\n{'='*80}")
        print(f"OPENVLA BASELINE EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Task Suite: {args.task_suite}")
        print(f"Overall Success Rate: {final_success_rate:.1%} ({total_successes}/{total_episodes})")
        print(f"\nPer-Task Results:")
        print(f"{'Task':<50} {'Success Rate':<15} {'Episodes'}")
        print(f"{'-'*80}")
        for result in task_results:
            print(f"{result['task_description'][:47]:<50} {result['success_rate']:>6.1%} ({result['successes']}/{result['episodes']})")

        if policy_stats:
            print(f"\n{'='*80}")
            print(f"Policy Statistics:")
            print(f"  Total Predictions: {policy_stats.get('total_predictions', 0)}")
            print(f"  Avg Inference Time: {policy_stats.get('avg_inference_time_ms', 0):.2f}ms")
            print(f"  Fallback Rate: {policy_stats.get('fallback_rate', 0):.1%}")
        print(f"{'='*80}\n")

        # Compute failure statistics
        num_failures = total_episodes - total_successes
        print(f"Failure Analysis:")
        print(f"  Total Failures: {num_failures}/{total_episodes}")
        if args.record_videos and not args.save_all_videos:
            print(f"  Failure videos saved to: {failure_video_dir}")
        elif args.record_videos and args.save_all_videos:
            print(f"  All videos saved to: {all_video_dir}")
        print(f"{'='*80}\n")

        # Save results
        results_path = Path(args.results_dir) / "baseline_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.log_text(f"Results saved to {results_path}")

        # Save episode logs separately
        logs_path = Path(args.results_dir) / "baseline_episode_logs.json"
        with open(logs_path, 'w') as f:
            json.dump(episode_logs, f, indent=2)
        logger.log_text(f"Episode logs saved to {logs_path}")

        logger.log({'final_success_rate': final_success_rate})
        logger.finish(status="completed")

        return results

    except Exception as e:
        logger.log_text(f"ERROR: Evaluation failed: {e}")
        import traceback
        logger.log_text(traceback.format_exc())
        logger.finish(status="failed")
        raise


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Baseline Evaluation on LIBERO")

    # Task settings
    parser.add_argument('--task-suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                        help='LIBERO task suite to evaluate')
    parser.add_argument('--num-trials', type=int, default=50,
                        help='Number of trials per task')

    # Model paths (use env vars with defaults)
    parser.add_argument('--openvla-checkpoint', type=str,
                        default=os.getenv('OPENVLA_CHECKPOINT', 'moojink/openvla-7b-oft-finetuned-libero-spatial'),
                        help='OpenVLA-OFT checkpoint path or HF model ID')

    # Video recording settings
    parser.add_argument('--record-videos', action='store_true', default=True,
                        help='Record episode videos')
    parser.add_argument('--save-all-videos', action='store_true', default=False,
                        help='Save all videos (default: only failures)')
    parser.add_argument('--no-videos', dest='record_videos', action='store_false',
                        help='Disable video recording')

    # Output settings
    parser.add_argument('--results-dir', type=str,
                        default=os.getenv('RESULTS_DIR', './results'),
                        help='Directory to save results')
    parser.add_argument('--log-dir', type=str,
                        default=os.getenv('LOG_DIR', './logs'),
                        help='Directory to save logs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Run evaluation
    run_evaluation(args)


if __name__ == '__main__':
    main()
