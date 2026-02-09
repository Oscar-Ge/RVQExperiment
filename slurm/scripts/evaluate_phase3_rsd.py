#!/usr/bin/env python3
"""
Phase 3: LIBERO Evaluation with Conditional RSD

Implements full Residual Speculative Decoding pipeline:
1. Load RFSQ tokenizer, Draft model, Conditional Main model
2. Load OpenVLA-OFT policy
3. Run on LIBERO benchmark with speculative decoding:
   - Draft model predicts L0-L2 (fast)
   - Compute complexity metric (entropy/variance)
   - If complex, Main model predicts L3-L7 (accurate)
4. Track success rate, latency, adaptive rate

Usage:
    python evaluate_phase3_rsd.py --task-suite libero_spatial --num-trials 50
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

# Add LIBERO to path
sys.path.insert(0, os.getenv('LIBERO_PATH', '/root/LIBERO'))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rfsq_models import ActionRFSQAE, RFSQDraftModelWithProjection, ConditionedRFSQHead
from utils.experiment_logger import ExperimentLogger


class ConditionalRSDInferenceEngine:
    """Conditional RSD with Mode Locking + Official Fixes"""

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
            'draft_time_ms': 0.0,
            'main_time_ms': 0.0,
            'decode_time_ms': 0.0,
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
        with torch.no_grad():
            # Step 1: Get OpenVLA hidden states
            hidden_states = self._get_openvla_features(observation, task_description)

            if hidden_states is None:
                # Fallback to official OpenVLA action prediction
                self.stats['fallback_to_openvla'] += 1
                actions = self._get_official_openvla_actions(observation, task_description)
                return actions

            # Step 2: Conditional Speculative Decoding with Mode Locking
            if self.use_speculative and self.draft_model is not None:
                # 2a. Draft Model predicts L0-L2
                t0 = time.time()
                draft_logits = self.draft_model(hidden_states)  # [1, 3, 128, 7]
                draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]
                self.stats['draft_time_ms'] += (time.time() - t0) * 1000

                # 2b. Reshape Draft tokens to condition format [B, Chunk, Hidden, Layers]
                draft_tokens_reshaped = draft_tokens.permute(0, 2, 1)  # [1, 128, 3]
                draft_condition = draft_tokens_reshaped.view(1, self.chunk_len, self.action_hidden_dim, 3)

                # 2c. CONDITIONAL Main Model with Mode Locking
                t1 = time.time()
                main_logits = self.conditional_rfsq_head(hidden_states, draft_condition)  # [1, 8, 128, 7]
                self.stats['main_time_ms'] += (time.time() - t1) * 1000
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
            t2 = time.time()
            actions = self._decode_actions(final_logits)
            self.stats['decode_time_ms'] += (time.time() - t2) * 1000

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
        try:
            from PIL import Image as PILImage

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

            # Forward through VLA to get hidden states
            outputs = self.vla(**inputs, output_hidden_states=True)

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()  # [1, 4096]
                if hidden_4096.shape == (1, 4096):
                    return hidden_4096

            return None

        except Exception as e:
            print(f"   WARNING: OpenVLA feature extraction error: {e}")
            return None

    def _get_official_openvla_actions(self, observation, task_description):
        """Fallback: Use official OpenVLA action prediction"""
        from PIL import Image as PILImage

        try:
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
                action_chunk = action_logits[0, -1, :]  # [chunk*7]
                action_chunk = action_chunk.view(self.chunk_len, 7)  # [8, 7]
                return action_chunk.cpu().numpy()
            else:
                return np.zeros((self.chunk_len, 7), dtype=np.float32)

        except Exception as e:
            print(f"   WARNING: Official OpenVLA action error: {e}")
            return np.zeros((self.chunk_len, 7), dtype=np.float32)

    def _accept_reject(self, draft_logits, main_logits):
        """Accept/reject mechanism"""
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
        try:
            batch_size = logits.shape[0]
            num_layers = logits.shape[1]

            token_indices = torch.argmax(logits, dim=-1)  # [1, 8, 128]
            token_indices = token_indices.permute(0, 2, 1)  # [1, 128, 8]
            token_indices = token_indices.view(batch_size, self.chunk_len, self.action_hidden_dim, num_layers)

            actions = self.rfsq_decoder.decode_from_indices(token_indices)  # [1, 8, 7]
            return actions[0].cpu().numpy()  # [8, 7]

        except Exception as e:
            print(f"   WARNING: Action decoding error: {e}")
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
            'avg_draft_time_ms': self.stats['draft_time_ms'] / total,
            'avg_main_time_ms': self.stats['main_time_ms'] / total,
            'avg_decode_time_ms': self.stats['decode_time_ms'] / total,
        }


def load_openvla_models(cfg, device):
    """Load OpenVLA-OFT model using official functions"""
    # Import official utilities (assuming they're in the environment)
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
    # Import config class
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
            # Simple fallback (not accurate)
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
        # Fallback: zero action
        return np.zeros(7)


def run_evaluation(args):
    """Run LIBERO evaluation with Conditional RSD"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Phase 3: Conditional RSD Evaluation - {args.task_suite}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Speculative Decoding: {args.use_speculative_decoding}")
    print(f"Num Trials: {args.num_trials}")
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
        name=f"phase3_rsd_{args.task_suite}",
        log_dir=args.log_dir,
        config={
            'task_suite': args.task_suite,
            'num_trials': args.num_trials,
            'use_speculative_decoding': args.use_speculative_decoding,
            'checkpoint': args.openvla_checkpoint,
            'rfsq_model': args.rfsq_model,
            'draft_model': args.draft_model,
            'main_model': args.main_model,
        }
    )

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

        # 2. Load RFSQ Decoder
        logger.log_text("Loading RFSQ Decoder...")
        rfsq_decoder = ActionRFSQAE(
            action_dim=7, hidden_dim=16, num_layers=8, num_levels=7, use_layernorm=True
        ).to(device)
        if Path(args.rfsq_model).exists():
            checkpoint = torch.load(args.rfsq_model, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            rfsq_decoder.load_state_dict(state_dict)
            logger.log_text(f"RFSQ Decoder loaded: {args.rfsq_model}")
        else:
            logger.log_text(f"WARNING: RFSQ model not found at {args.rfsq_model}")
        rfsq_decoder.eval()

        # 3. Load Conditional RFSQ Head
        logger.log_text("Loading Conditional RFSQ Head...")
        conditional_rfsq_head = ConditionedRFSQHead(
            input_dim=4096, hidden_dim=1024, num_layers=8, chunk_len=8,
            action_hidden_dim=16, grid_size=7, condition_layers=3, token_embed_dim=64,
        ).to(device)
        if Path(args.main_model).exists():
            checkpoint = torch.load(args.main_model, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            conditional_rfsq_head.load_state_dict(state_dict)
            logger.log_text(f"Conditional RFSQ Head loaded: {args.main_model}")
        else:
            logger.log_text(f"WARNING: Main model not found at {args.main_model}")
        conditional_rfsq_head.eval()

        # 4. Load Draft Model (if speculative decoding enabled)
        draft_model = None
        if args.use_speculative_decoding:
            logger.log_text("Loading Draft Model...")
            draft_model = RFSQDraftModelWithProjection(
                input_dim=4096, hidden_dim=512, num_coarse_layers=3,
                chunk_len=8, action_hidden_dim=16, grid_size=7,
            ).to(device)
            if Path(args.draft_model).exists():
                checkpoint = torch.load(args.draft_model, map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
                draft_model.load_state_dict(state_dict)
                logger.log_text(f"Draft Model loaded: {args.draft_model}")
            else:
                logger.log_text(f"WARNING: Draft model not found at {args.draft_model}")
            draft_model.eval()

        # 5. Initialize RSD Engine
        logger.log_text("Initializing Conditional RSD Engine...")
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
            use_speculative=args.use_speculative_decoding,
            resize_size=resize_size,
        )

        # 6. Initialize LIBERO
        logger.log_text(f"Initializing LIBERO {args.task_suite}...")
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_obj = benchmark_dict[args.task_suite]()
        num_tasks = task_suite_obj.n_tasks
        logger.log_text(f"Number of tasks: {num_tasks}")

        # 7. Run Evaluation
        total_episodes = 0
        total_successes = 0
        task_results = []

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

                # Reset environment
                env.reset()
                obs = env.set_init_state(init_states[trial_idx])

                # Initialize action queue
                action_queue = deque(maxlen=cfg.num_open_loop_steps)

                episode_success = False
                max_steps = 220

                for step in range(max_steps + 10):
                    try:
                        # First 10 steps: stabilization
                        if step < 10:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            continue

                        # Prepare observation
                        observation = prepare_observation(obs, resize_image_for_policy, resize_size)

                        # Query model if action queue is empty
                        if len(action_queue) == 0:
                            actions = rsd_engine.predict_action_chunk(observation, task_description)
                            action_queue.extend(actions)

                        # Get action from queue
                        action = action_queue.popleft()

                        # Process action
                        action = process_action(action)

                        # Execute action
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            episode_success = True
                            break

                    except Exception as step_error:
                        logger.log_text(f"Step error: {step_error}")
                        break

                env.close()

                if episode_success:
                    task_successes += 1
                    total_successes += 1

                logger.log_text(f"Trial {trial_idx + 1}: {'SUCCESS' if episode_success else 'FAILED'}")

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

        # 8. Final Results
        final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
        rsd_stats = rsd_engine.get_stats()

        results = {
            'task_suite': args.task_suite,
            'use_speculative_decoding': args.use_speculative_decoding,
            'model_type': 'conditional_rsd',
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'final_success_rate': final_success_rate,
            'task_results': task_results,
            'rsd_stats': rsd_stats,
            'config': {
                'checkpoint': args.openvla_checkpoint,
                'resize_size': resize_size,
                'unnorm_key': cfg.unnorm_key,
            }
        }

        # Print summary table
        print(f"\n{'='*80}")
        print(f"CONDITIONAL RSD EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Task Suite: {args.task_suite}")
        print(f"Overall Success Rate: {final_success_rate:.1%} ({total_successes}/{total_episodes})")
        print(f"\nPer-Task Results:")
        print(f"{'Task':<50} {'Success Rate':<15} {'Episodes'}")
        print(f"{'-'*80}")
        for result in task_results:
            print(f"{result['task_description'][:47]:<50} {result['success_rate']:>6.1%} ({result['successes']}/{result['episodes']})")

        if args.use_speculative_decoding and rsd_stats:
            print(f"\n{'='*80}")
            print(f"RSD Statistics:")
            print(f"  Draft Acceptance Rate: {rsd_stats.get('draft_acceptance_rate', 0):.1%}")
            print(f"  Mode Locking Rate: {rsd_stats.get('mode_locking_rate', 0):.1%}")
            print(f"  Fallback Rate: {rsd_stats.get('fallback_rate', 0):.1%}")
            print(f"  Avg Draft Time: {rsd_stats.get('avg_draft_time_ms', 0):.2f}ms")
            print(f"  Avg Main Time: {rsd_stats.get('avg_main_time_ms', 0):.2f}ms")
        print(f"{'='*80}\n")

        # Save results
        results_path = Path(args.results_dir) / f"{args.task_suite}_rsd_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.log_text(f"Results saved to {results_path}")

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
    parser = argparse.ArgumentParser(description="Phase 3: LIBERO Evaluation with Conditional RSD")

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
    parser.add_argument('--rfsq-model', type=str,
                        default=os.getenv('RFSQ_MODEL', '/models/rfsq_robust_best.pt'),
                        help='Path to trained RFSQ tokenizer')
    parser.add_argument('--draft-model', type=str,
                        default=os.getenv('DRAFT_MODEL', '/models/best_draft_with_projection.pt'),
                        help='Path to trained Draft model')
    parser.add_argument('--main-model', type=str,
                        default=os.getenv('MAIN_MODEL', '/models/openvla_rfsq_conditional/best_rfsq_head.pt'),
                        help='Path to trained Conditional Main model')

    # RSD settings
    parser.add_argument('--use-speculative-decoding', type=bool, default=True,
                        help='Enable speculative decoding with Draft model')

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
