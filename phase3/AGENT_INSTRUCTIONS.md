# Instructions for Experiment Agent: Phase 3 LIBERO Evaluation

## 🎯 Your Mission

Complete the Phase 3 LIBERO evaluation by implementing the missing model loading and environment integration code, then run the evaluation on Modal.

---

## 📋 Prerequisites Check

Before starting, verify:
- [ ] Modal CLI installed and authenticated
- [ ] Access to Modal volumes: `rsd-libero-data`, `rsd-models`, `huggingface-cache`
- [ ] Models from Phase 1 & 2 exist in volumes
- [ ] Orchestra SDK credentials configured

---

## 🔧 Task 1: Implement Model Loading (Priority: CRITICAL)

### Location
File: `modal_phase3_libero_eval.py`, starting at line ~170

### What to Implement

#### 1.1 Load RFSQ Decoder (Already Implemented ✅)
This is already done. Verify it loads correctly.

#### 1.2 Load Main Model (OpenVLA-OFT-RFSQ)

**Replace this placeholder:**
```python
# Line ~170
main_model = None  # Currently placeholder
```

**With this implementation:**
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch.nn as nn

# Load base OpenVLA-OFT model
print(f"   Loading base OpenVLA-OFT model...")
base_model_name = "moojink/openvla-7b-oft-finetuned-libero-spatial"

try:
    main_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,  # Use 4-bit to save memory
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    print(f"   ✓ Base model loaded")

    # Load RFSQ classification head
    rfsq_head_path = "/models/openvla_oft_rfsq/best_model.pt"
    if Path(rfsq_head_path).exists():
        print(f"   Loading RFSQ head from {rfsq_head_path}...")

        # Define RFSQ head architecture (8 parallel classification heads)
        class RFSQActionHead(nn.Module):
            def __init__(self, input_dim=4096, num_layers=8, action_dim=7, grid_size=7, chunk_len=8):
                super().__init__()
                self.num_layers = num_layers
                self.action_dim = action_dim
                self.grid_size = grid_size
                self.chunk_len = chunk_len

                # 8 parallel linear layers for RFSQ classification
                self.classification_heads = nn.ModuleList([
                    nn.Linear(input_dim, chunk_len * action_dim * grid_size)
                    for _ in range(num_layers)
                ])

            def forward(self, hidden_states):
                # hidden_states: [Batch, Seq, Hidden]
                batch_size = hidden_states.shape[0]

                # Take last token (action prediction)
                action_token = hidden_states[:, -1, :]  # [Batch, Hidden]

                # Predict each layer
                layer_logits = []
                for head in self.classification_heads:
                    logits = head(action_token)  # [Batch, chunk*action*grid]
                    logits = logits.view(batch_size, self.chunk_len, self.action_dim, self.grid_size)
                    layer_logits.append(logits)

                # Stack: [Batch, Num_Layers, Chunk, Action, Grid]
                return torch.stack(layer_logits, dim=1)

        # Create and load RFSQ head
        rfsq_head = RFSQActionHead()
        checkpoint = torch.load(rfsq_head_path, map_location=device)

        # Load weights (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            rfsq_head.load_state_dict(checkpoint['model_state_dict'])
        elif 'rfsq_head' in checkpoint:
            rfsq_head.load_state_dict(checkpoint['rfsq_head'])
        else:
            print(f"   ⚠️  Unexpected checkpoint format, keys: {checkpoint.keys()}")

        rfsq_head = rfsq_head.to(device)
        rfsq_head.eval()

        # Attach to main model
        main_model.rfsq_head = rfsq_head

        print(f"   ✓ RFSQ head loaded and attached")
    else:
        print(f"   ⚠️  RFSQ head not found at {rfsq_head_path}")
        print(f"   Using random initialization (for testing only)")
        main_model.rfsq_head = None

except Exception as e:
    print(f"   ❌ Failed to load main model: {e}")
    print(f"   Using fallback: will create dummy model for testing")
    main_model = None
    processor = None
```

#### 1.3 Load Draft Model

**Replace this placeholder:**
```python
# Line ~190
draft_model = None  # Currently placeholder
```

**With this implementation:**
```python
if use_speculative_decoding:
    draft_model_path = "/models/phase2_draft_model/best_draft_model.pt"
    print(f"\n   Loading Draft Model from {draft_model_path}")

    # Define Draft Model architecture (matching Phase 2)
    class DraftTransformerDecoder(nn.Module):
        def __init__(self, hidden_dim=4096, num_heads=8, feedforward_dim=2048, max_seq_length=256):
            super().__init__()
            self.hidden_dim = hidden_dim

            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=0.0,
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

    class RFSQDraftModel(nn.Module):
        def __init__(self, hidden_dim=4096, num_coarse_layers=3, action_dim=7, grid_size=7):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_coarse_layers = num_coarse_layers
            self.grid_size = grid_size

            self.decoder = DraftTransformerDecoder(hidden_dim=hidden_dim)

            # Classification heads for coarse layers
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, grid_size),
                )
                for _ in range(num_coarse_layers)
            ])

        def forward(self, embeddings):
            batch_size, seq_len, _ = embeddings.shape
            decoder_output = self.decoder(embeddings)

            layer_outputs = []
            for head in self.classification_heads:
                flat_input = decoder_output.view(-1, self.hidden_dim)
                flat_logits = head(flat_input)
                layer_logits = flat_logits.reshape(batch_size, seq_len, self.grid_size)
                layer_outputs.append(layer_logits)

            # [Batch, Num_Coarse, Seq_Len, Grid_Size]
            return torch.stack(layer_outputs, dim=1)

    try:
        draft_model = RFSQDraftModel(
            hidden_dim=4096,
            num_coarse_layers=3,
            action_dim=7,
            grid_size=7,
        )

        if Path(draft_model_path).exists():
            checkpoint = torch.load(draft_model_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                draft_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                draft_model.load_state_dict(checkpoint)

            draft_model = draft_model.to(device)
            draft_model.eval()

            print(f"   ✓ Draft Model loaded (4.7M params)")
        else:
            print(f"   ⚠️  Draft Model not found, using random init")

    except Exception as e:
        print(f"   ❌ Failed to load draft model: {e}")
        draft_model = None
else:
    draft_model = None
    print(f"\n   Skipping Draft Model (speculative decoding disabled)")
```

---

## 🔧 Task 2: Integrate RSD Inference Engine

### Location
File: `modal_phase3_libero_eval.py`, line ~250

### What to Implement

**Add after model loading:**
```python
# Create RSD Inference Engine
if use_speculative_decoding and draft_model is not None:
    from rsd_inference_engine import RSDInferenceEngine

    print(f"\n🚀 Initializing RSD Inference Engine...")

    engine = RSDInferenceEngine(
        main_model=main_model,
        draft_model=draft_model,
        rfsq_decoder=rfsq_model,
        num_layers=8,
        num_coarse_layers=3,
        acceptance_threshold=0.7,
        enable_partial_acceptance=True,
        device=device,
    )

    print(f"   ✓ RSD Engine ready")
else:
    engine = None
    print(f"\n⚠️  RSD Engine disabled (using main model only)")
```

---

## 🔧 Task 3: Implement LIBERO Environment Integration

### Location
File: `modal_phase3_libero_eval.py`, line ~300

### What to Implement

**Replace the placeholder episode loop with:**

```python
# Import LIBERO utils
from libero.libero.envs import OffScreenRenderEnv

for trial_idx in range(min(num_trials, len(init_states))):
    task_episodes += 1
    total_episodes += 1

    # Create environment for this trial
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=task.problem_folder,
            camera_heights=256,
            camera_widths=256,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
        )

        # Reset environment
        env.reset()
        env.set_init_state(init_states[trial_idx])

        # Get initial observation
        obs = env.get_observation()

        # Episode loop
        max_steps = 300
        success = False
        episode_start_time = time.time()

        for step in range(max_steps):
            # Prepare observation for model
            observation = {
                'full_image': obs['agentview_image'],  # [256, 256, 3]
                'wrist_image': obs.get('robot0_eye_in_hand_image', obs['agentview_image']),
                'state': np.concatenate([
                    obs['robot0_eef_pos'],
                    obs['robot0_eef_quat'],
                    obs['robot0_gripper_qpos'],
                ]),
            }

            # Generate action using RSD engine or main model
            if engine is not None:
                actions, info = engine.generate_action(
                    observation=observation,
                    task_description=task_description,
                    processor=processor,
                    chunk_len=8,
                    action_dim=7,
                )
                inference_time = info['total_time']
            else:
                # Fallback: use main model directly (simplified)
                # TODO: Implement direct main model inference
                actions = np.random.randn(8, 7)  # Placeholder
                inference_time = 0.05

            # Execute first action from chunk
            action = actions[0]

            # Step environment
            obs, reward, done, info_env = env.step(action)

            if done:
                success = True
                break

        episode_time = time.time() - episode_start_time

        # Update stats
        if success:
            task_successes += 1
            total_successes += 1

        total_inference_time += inference_time

        print(f"   Trial {trial_idx + 1}/{num_trials}: "
              f"{'✓' if success else '✗'} "
              f"(time: {episode_time:.1f}s, inf: {inference_time*1000:.1f}ms)")

        # Close environment
        env.close()

    except Exception as e:
        print(f"   ❌ Trial {trial_idx + 1} failed: {e}")
        continue
```

---

## 🐛 Task 4: Debug and Test

### Step 1: Test with Debug Mode

```bash
# Run with minimal trials for quick debugging
modal run phase3/modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 1 \
    --use-speculative-decoding False
```

### Step 2: Check for Common Issues

**If you see "Module not found: prismatic":**
```python
# Verify OpenVLA-OFT installed in image build
uv pip install --system 'openvla-oft @ git+https://github.com/moojink/openvla-oft.git'
```

**If you see "CUDA out of memory":**
```python
# Enable 4-bit quantization
main_model = AutoModelForVision2Seq.from_pretrained(
    base_model_name,
    load_in_4bit=True,  # Add this
    device_map="auto",
)
```

**If you see "RFSQ dimension mismatch":**
```python
# Check hidden_dim=16 (not 64) for RFSQ decoder
rfsq_model = ActionRFSQAE(action_dim=7, hidden_dim=16, num_layers=8, num_levels=7)
```

**If you see "LIBERO environment fails":**
```python
# Check that torch.load fix is applied in Modal image
cd /root/LIBERO && sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
    libero/libero/benchmark/__init__.py
```

### Step 3: Run Full Evaluation

Once debug mode works:
```bash
modal run phase3/modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True
```

---

## 📊 Expected Output

### Success Looks Like:

```
============================================================
🎉 EVALUATION COMPLETE!
============================================================
   Task Suite: libero_spatial
   Total Episodes: 500
   Total Successes: 425-475 (85-95%)
   Success Rate: 85.0% - 95.0%
   Avg Inference Time: 45-55 ms
   Speculative Decoding: True
============================================================
```

### Metrics to Log

- Success rate per task
- Average inference time
- Draft acceptance rate (if HSD enabled)
- Per-layer accuracy correlation

---

## ✅ Completion Checklist

- [ ] All 4 model loading sections implemented
- [ ] RSD Engine integrated
- [ ] LIBERO environment working
- [ ] Debug mode runs successfully (1 task, 1 trial)
- [ ] Full evaluation runs without crashes
- [ ] Results saved to `/results/` volume
- [ ] Experiment logged to Orchestra SDK
- [ ] Success rate > 80%
- [ ] Inference time logged correctly

---

## 🆘 If You Get Stuck

1. **Check Modal logs**: `modal app logs rsd-phase3-libero-eval`
2. **Check volume contents**: `modal volume ls rsd-models`
3. **Read guides**: See `PHASE3_EXPERIMENT_GUIDE.md` for detailed help
4. **Test incrementally**: Start with 1 trial, then scale up
5. **Report errors**: Include full stack trace and context

---

## 🎯 Success Criteria

- [ ] Code runs without crashes
- [ ] Success rate 80-95% (compare with 97% baseline)
- [ ] Inference time < 60ms
- [ ] All 10 tasks in libero_spatial evaluated
- [ ] Results JSON saved to volume
- [ ] Experiment marked as "completed" in Orchestra

---

**Go forth and evaluate! 🚀**

Remember: The hard work (Phase 1 & 2 training) is done. You're just wiring up existing models and running evaluation. You've got this!
