# Phase 3: LIBERO Baseline Evaluation - Usage Guide

**Purpose**: Evaluate OpenVLA-OFT baseline model on LIBERO simulation tasks
**Platform**: Modal (Cloud A100 GPU)
**Script**: `modal_phase3_libero_eval_FIXED_v2.py`

---

## 🎯 What is Phase 3?

Phase 3 evaluates the **OpenVLA-OFT baseline model** on LIBERO robotic manipulation tasks to establish performance benchmarks before testing RSD (Residual Speculative Decoding) improvements.

### Key Components

- **Model**: `moojink/openvla-7b-oft-finetuned-libero-spatial` (from HuggingFace)
- **Environment**: LIBERO simulation (10 spatial manipulation tasks)
- **Evaluation**: 50 trials per task, measuring success rate
- **Expected Performance**: 85-95% success rate

---

## 🚀 Quick Start

### Prerequisites

1. **Modal CLI installed**:
   ```bash
   pip install modal
   modal setup
   ```

2. **HuggingFace token** (for model download):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=<your-token>
   ```

3. **Orchestra Supabase credentials** (if using experiment tracking):
   ```bash
   modal secret create orchestra-supabase <credentials>
   ```

### Run Evaluation

```bash
# Quick test (3 trials per task, ~15 minutes)
modal run modal_phase3_libero_eval_FIXED_v2.py --num-trials 3

# Full evaluation (50 trials per task, ~2-3 hours)
modal run modal_phase3_libero_eval_FIXED_v2.py --num-trials 50

# Different task suite
modal run modal_phase3_libero_eval_FIXED_v2.py \
  --task-suite libero_object \
  --num-trials 50
```

---

## 📊 Expected Output

### During Evaluation

```
🚀 Phase 3: LIBERO Evaluation (FIXED v2) - libero_spatial
   Speculative Decoding: DISABLED

✓ Random seed set to 7 for reproducibility

📦 Loading OpenVLA-OFT model using official functions...
✅ Model already has norm_stats with keys: ['libero_spatial_no_noops']
   ✓ Using unnorm_key: libero_spatial_no_noops
   ✓ Image resize size: 224

🏗️ Initializing LIBERO libero_spatial...
   Number of tasks: 10

🎯 Starting evaluation (50 trials per task)...

Task 1/10: pick up the black bowl on the plate and place it on the tray
   🔍 Image shapes after resize:
      full_image: (224, 224, 3)
      wrist_image: (224, 224, 3)
      state: (8,)
   Trial 1: ✓
   Trial 2: ✓
   ...
   Task Success Rate: 92.0% (46/50)

...

🎉 EVALUATION COMPLETE!
   Success Rate: 88.5%
```

### Results

Results are saved to `/results/libero_spatial_baseline_fixed_v2.json`:

```json
{
  "task_suite": "libero_spatial",
  "total_episodes": 500,
  "total_successes": 442,
  "final_success_rate": 0.884,
  "task_results": [
    {
      "task_id": 0,
      "task_description": "pick up the black bowl...",
      "success_rate": 0.92
    },
    ...
  ],
  "config": {
    "seed": 7,
    "unnorm_key": "libero_spatial_no_noops",
    "resize_size": 224,
    "model": "moojink/openvla-7b-oft-finetuned-libero-spatial"
  }
}
```

---

## 🔧 Configuration

### Available Task Suites

| Suite | Tasks | Description |
|-------|-------|-------------|
| `libero_spatial` | 10 | Spatial reasoning (default) |
| `libero_object` | 10 | Object manipulation |
| `libero_goal` | 10 | Goal-oriented tasks |
| `libero_10` | 10 | Long-horizon tasks |
| `libero_90` | 90 | All tasks |

### Key Parameters

```python
# In the script (modal_phase3_libero_eval_FIXED_v2.py)

# Model configuration
cfg = GenerateConfig(
    pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression=True,      # Use L1 action head
    use_proprio=True,            # Include proprioception
    center_crop=True,            # Center crop images
    num_open_loop_steps=8,       # Action chunk size
)

# Environment
SEED = 7                         # Random seed
max_steps = 220                  # Max steps per episode
```

---

## 🔍 Key Features (All Fixes Applied)

### ✅ Critical Fixes

1. **Image Resize**: 256x256 (env) → 224x224 (model input)
   - Uses JPEG encode/decode + lanczos3 interpolation
   - Matches training distribution

2. **Image Rotation**: 180° rotation to match training preprocessing
   ```python
   img = img[::-1, ::-1]  # CRITICAL
   ```

3. **Norm Stats Handling**: Defensive programming with fallback
   - Checks if model has norm_stats
   - Falls back to manual injection if needed
   - Uses official `check_unnorm_key()` verification

### ✅ Implementation Details

4. **Official Functions**: Uses openvla-oft official utilities
   - `get_action()` wrapper (adds torch.no_grad)
   - `get_libero_dummy_action()` for stabilization
   - `resize_image_for_policy()` for preprocessing

5. **EGL Rendering**: GPU-accelerated off-screen rendering
   - ~10x faster than CPU osmesa rendering
   - Configured in Docker image

6. **Seed Setting**: SEED=7 for reproducibility

---

## 📈 Performance Benchmarks

### Expected Results

| Metric | Expected | Notes |
|--------|----------|-------|
| **Success Rate** | 85-95% | LIBERO-Spatial with 50 trials |
| **Per-Task Variance** | ±5-10% | Some tasks harder than others |
| **Inference Time** | ~65-75ms | Per action prediction |

### Comparison to Paper

OpenVLA-OFT paper reports:
- LIBERO-Spatial: ~90% success rate
- LIBERO-Object: ~85% success rate
- LIBERO-Goal: ~80% success rate

Our implementation should match these results.

---

## 🐛 Troubleshooting

### Issue: Model download fails

**Solution**: Ensure HuggingFace token is set:
```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
```

### Issue: EGL rendering error

**Symptom**: `EGL device not found`

**Solution**: The Docker image includes EGL dependencies. If still failing:
- Check Modal GPU allocation
- Verify `libegl1-mesa-dev` in apt_install

### Issue: Low success rate (<50%)

**Check**:
1. Verify image shapes in output: `(224, 224, 3)` ✓
2. Check unnorm_key: `libero_spatial_no_noops` ✓
3. Ensure seed=7 for reproducibility

### Issue: Out of memory

**Solution**: Script uses A100 40GB GPU. If OOM:
- Reduce num_trials for testing
- Check for memory leaks (env.close() called)

---

## 📚 Technical Details

### Model Architecture

```
OpenVLA-OFT (7B parameters)
├── Vision Encoder: PrismaticVLM
├── LLM Backbone: LLaMA-2 7B
├── Action Head: L1 Regression MLP
└── Proprio Projector: MLP (8-dim → llm_dim)
```

### Evaluation Loop

```
For each task (10 tasks):
  For each trial (50 trials):
    1. Reset environment with fixed initial state
    2. Wait 10 steps for stabilization
    3. For each step (max 220):
       - Get observation (rotate + resize images)
       - Predict action chunk (8 actions)
       - Execute actions open-loop
       - Check if task succeeded
    4. Record success/failure
```

### Action Processing

```python
# 1. Model predicts action (normalized)
action = model.predict(observation)

# 2. Denormalize using dataset statistics
action = action * std + mean

# 3. Normalize gripper: [0,1] → [-1,+1]
action = normalize_gripper_action(action)

# 4. Invert gripper for OpenVLA: flip sign
action = invert_gripper_action(action)

# 5. Execute in environment
env.step(action)
```

---

## 📊 Cost Estimation

### Modal Pricing (A100 GPU)

- **Quick test (3 trials)**: ~$0.50 (15 minutes)
- **Full evaluation (50 trials)**: ~$5-8 (2-3 hours)

### Optimization Tips

1. Use quick tests for debugging
2. Run full evaluation only when confident
3. Modal auto-scales and stops billing when done

---

## 🎯 Next Steps After Phase 3

Once baseline performance is established:

1. **Phase 4**: Implement RSD (Residual Speculative Decoding)
2. **Compare**: RSD vs Baseline on same tasks
3. **Analyze**: Speed-accuracy tradeoffs

Expected RSD improvements:
- Inference speed: 1.5-2x faster
- Maintained accuracy: >90% of baseline

---

## 📝 File Overview

```
finalCode/
├── modal_phase3_libero_eval_FIXED_v2.py   # Main evaluation script
└── PHASE3_USAGE.md                         # This file
```

### Script Structure

```python
# modal_phase3_libero_eval_FIXED_v2.py

1. Modal app setup (lines 25-77)
   - Docker image with dependencies
   - Environment variables (EGL, CUDA)
   - Volume mounts

2. Evaluation function (lines 95-500)
   - Model loading
   - Norm stats handling
   - LIBERO initialization
   - Evaluation loop
   - Results saving

3. Local entrypoint (lines 502-522)
   - CLI argument parsing
   - Remote execution trigger
```

---

## ✅ Verification

Before running, verify:

- [ ] Modal CLI installed and authenticated
- [ ] HuggingFace secret configured
- [ ] Script downloaded: `modal_phase3_libero_eval_FIXED_v2.py`
- [ ] Network connection available (for model download)

After first run, verify:

- [ ] Model downloaded successfully
- [ ] Image shapes: `(224, 224, 3)` ✓
- [ ] Unnorm key: `libero_spatial_no_noops` ✓
- [ ] Success rate: 85-95% ✓

---

## 📖 References

- **OpenVLA-OFT Repository**: https://github.com/moojink/openvla-oft
- **LIBERO Benchmark**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **Paper**: "OpenVLA: An Open-Source Vision-Language-Action Model"
- **HuggingFace Model**: https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial

---

**Last Updated**: 2026-01-19
**Version**: v3.0 (FINAL - All fixes applied)
**Status**: ✅ Ready for production use
