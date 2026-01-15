# Phase 3 Experiment Guide: LIBERO Evaluation with RSD

## 📋 Overview

This guide explains how to run **Phase 3** of the RSD (Residual Speculative Decoding) experiment: evaluating the trained models on LIBERO benchmark tasks.

### What You Have So Far

✅ **Phase 1 Complete**: RFSQ AutoEncoder trained and saved
✅ **Phase 2 Complete**:
  - OpenVLA-OFT-RFSQ Main Model trained (90.9% token accuracy)
  - Draft Model trained (90.5% accuracy on coarse layers)

🎯 **Phase 3 Goal**: Evaluate RSD on LIBERO and measure:
  1. Success Rate (compare with OpenVLA-OFT baseline)
  2. Wall-clock inference time (prove speedup)
  3. Multimodal action handling (ambiguity test)

---

## 🚀 Quick Start

### Step 1: Run LIBERO Evaluation

```bash
# Run with speculative decoding enabled (RSD)
modal run RSD_Experiment/modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True

# Run baseline (no speculative decoding)
modal run RSD_Experiment/modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding False
```

### Step 2: Compare Results

The script will output:
- **Success Rate**: % of successful task completions
- **Avg Inference Time**: milliseconds per action chunk
- **Per-Task Breakdown**: success rates for each of the 10 tasks

---

## 📊 Understanding the Results

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Success Rate** | % of episodes where robot completes task | > 85% (compare with OFT baseline ~97%) |
| **Inference Time** | Time to generate 1 action chunk (8 timesteps) | < Baseline (faster due to HSD) |
| **Speedup** | `Baseline Time / RSD Time` | 1.3x - 1.6x expected |

### What Good Results Look Like

```
============================================================
🎉 EVALUATION COMPLETE!
============================================================
   Task Suite: libero_spatial
   Total Episodes: 500
   Total Successes: 450
   Success Rate: 90.0%              ← Close to baseline!
   Avg Inference Time: 45.2 ms      ← Faster than baseline (~70ms)
   Speculative Decoding: True
============================================================
```

### Interpreting Token Accuracy vs. Success Rate

- **Token Accuracy (90.9%)**: How well the model predicts discrete RFSQ tokens
- **Success Rate (target ~90%)**: How often the robot actually completes the task

⚠️ **Important**: Token accuracy ≠ Success rate!
- Token accuracy measures classification correctness
- Success rate measures task completion in simulation

---

## 🔧 Implementation Status

### What's Implemented

✅ **Modal infrastructure**: Volumes, GPU config, dependencies
✅ **RFSQ components**: Encoder, Decoder, Quantizer
✅ **LIBERO integration**: Environment setup, torch.load fix
✅ **Evaluation loop**: Task iteration, trial management
✅ **Logging**: Experiment tracking with Orchestra SDK

### What Needs Implementation

🚧 **Model Loading** (CRITICAL):
```python
# In modal_phase3_libero_eval.py, line ~170
# TODO: Load actual OpenVLA-OFT-RFSQ model
main_model = None  # Currently placeholder
```

**Action Required**: Implement proper model loading from HuggingFace checkpoint:
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load OpenVLA-OFT-RFSQ
model_name = "moojink/openvla-7b-oft-finetuned-libero-spatial"
main_model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load RFSQ classification head from your checkpoint
rfsq_head_path = "/models/openvla_oft_rfsq/best_model.pt"
checkpoint = torch.load(rfsq_head_path)
main_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

🚧 **Draft Model Loading**:
```python
# Load draft model architecture and weights
from RSD_Experiment.phase2.prismatic.models.draft_model import RFSQDraftModel

draft_model = RFSQDraftModel(
    hidden_dim=4096,
    num_coarse_layers=3,
    action_dim=7,
    grid_size=7,
)
draft_checkpoint = torch.load("/models/phase2_draft_model/best_draft_model.pt")
draft_model.load_state_dict(draft_checkpoint)
```

🚧 **RSD Inference Engine Integration**:
```python
# Integrate the RSDInferenceEngine from rsd_inference_engine.py
from rsd_inference_engine import RSDInferenceEngine

engine = RSDInferenceEngine(
    main_model=main_model,
    draft_model=draft_model,
    rfsq_decoder=rfsq_model,
    num_layers=8,
    num_coarse_layers=3,
    enable_partial_acceptance=True,
    device=device,
)

# Use in evaluation loop
actions, info = engine.generate_action(
    observation=obs,
    task_description=task_description,
    processor=processor,
    chunk_len=8,
    action_dim=7,
)
```

🚧 **LIBERO Environment**:
```python
# Proper environment creation and stepping
import robosuite
from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(
    bddl_file_name=task.problem_folder,
    camera_heights=256,
    camera_widths=256,
)

# Set initial state
env.reset()
env.set_init_state(init_states[trial_idx])

# Execute action
obs, reward, done, info = env.step(action)
```

---

## 🧪 Experimental Design

### Test Matrix

| Experiment | Config | Purpose |
|------------|--------|---------|
| **Baseline** | No HSD, Main Model only | Establish performance ceiling |
| **RSD Full** | HSD enabled, threshold=0.7 | Primary result |
| **RSD Conservative** | HSD enabled, threshold=0.9 | Test high-accuracy mode |
| **RSD Aggressive** | HSD enabled, threshold=0.5 | Test speedup limits |

Run all 4 experiments:
```bash
for threshold in 0.5 0.7 0.9; do
  modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True \
    --acceptance-threshold $threshold
done
```

### Task Suites to Test

1. **libero_spatial** (10 tasks): Spatial reasoning
2. **libero_object** (10 tasks): Object manipulation
3. **libero_goal** (10 tasks): Goal-conditioned tasks

Start with `libero_spatial` as it matches your training data.

---

## 📈 Expected Results

### Hypothesis

| Metric | Baseline (OFT) | RSD (Ours) | Explanation |
|--------|----------------|------------|-------------|
| Success Rate | 97.1% | **90-95%** | Slight drop due to quantization |
| Inference Time | ~70ms | **45-50ms** | 1.4x speedup from HSD |
| Batch Scalability | Poor (padding) | **Excellent** | Fixed-size tensors |

### Acceptance Strategy Performance

Expected partial acceptance stats:
- **Full Accept**: ~40% (draft perfectly matches main)
- **Partial Accept**: ~35% (2/3 coarse layers accepted)
- **Full Reject**: ~25% (main model predicts all layers)

Average accepted layers: **2.2 / 3** coarse layers

---

## 🐛 Troubleshooting

### Common Issues

**1. "Module not found: prismatic"**
```bash
# Solution: Ensure OpenVLA-OFT installed in eval image
uv pip install --system 'openvla-oft @ git+https://github.com/moojink/openvla-oft.git'
```

**2. "RFSQ decoder dimension mismatch"**
```python
# Issue: hidden_dim mismatch between encoder and decoder
# Solution: Ensure Phase 1 used hidden_dim=16, not 64
rfsq_model = ActionRFSQAE(action_dim=7, hidden_dim=16, num_layers=8, num_levels=7)
```

**3. "LIBERO torch.load pickle error"**
```bash
# Solution: Already fixed in evaluation image
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
  libero/libero/benchmark/__init__.py
```

**4. "CUDA out of memory"**
```python
# Solution: Use 4-bit quantization for main model
main_model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
)
```

---

## 📝 Next Steps After Phase 3

Once you have LIBERO results, proceed to:

### Day 9: Multimodal Ambiguity Test

Create scenarios with multiple valid action modes:
```python
# Example: "Pick up the block" when 2 identical blocks present
# Hypothesis: RSD can sample different modes, L1 regression averages to failure
```

### Day 10: Paper Writing

Generate plots for paper:
1. **Table 1**: LIBERO success rates
2. **Figure 1**: RFSQ architecture diagram
3. **Figure 2**: Multimodal action distribution (ambiguity test)
4. **Figure 3**: Iso-Latency plot (RSD vs FAST)

---

## 💡 Tips for Success

1. **Start Small**: Run 1 task, 5 trials first to debug quickly
2. **Check Logs**: Monitor Modal logs for CUDA errors
3. **Visualize Actions**: Save rollout videos to debug failures
4. **Compare Token Predictions**: Log draft vs main token agreement rates
5. **Profile Timing**: Break down inference into Draft/Main/Decode stages

---

## 📞 Getting Help

If stuck, check:
- Modal logs: `modal app logs rsd-phase3-libero-eval`
- Volume contents: `modal volume ls rsd-models`
- Experiment dashboard: Orchestra SDK web UI

---

## ✅ Success Criteria

You've succeeded when:

- [ ] Success rate > 85% (acceptable quantization drop from 97%)
- [ ] Inference time < 60ms (faster than baseline ~70ms)
- [ ] Speedup > 1.2x (provable acceleration)
- [ ] HSD acceptance rate > 60% (draft model is useful)
- [ ] No CUDA OOM errors (models fit on A100)

---

## 🎯 Final Checklist

Before running full evaluation:

- [ ] RFSQ Decoder loaded successfully
- [ ] Main Model (OpenVLA-OFT-RFSQ) loaded
- [ ] Draft Model loaded (if HSD enabled)
- [ ] LIBERO environment creates without errors
- [ ] Actions decoded to correct shape [8, 7]
- [ ] First episode runs end-to-end
- [ ] Timing metrics logged
- [ ] Results saved to volume

---

**Good luck with Phase 3! 🚀**

Remember: Even if success rate is slightly lower than baseline, the speedup and multimodal capability make RSD valuable. Document everything for the paper!
