# Evaluation Scripts

This directory contains two evaluation scripts for Phase 3 of the RSD project.

## Scripts Overview

### 1. `evaluate_phase3_rsd.py` - Full RSD Evaluation

Evaluates the complete Residual Speculative Decoding (RSD) pipeline on LIBERO benchmark.

**What it does:**
- Loads 3 models: RFSQ tokenizer, Draft model (L0-L2), Conditional Main model (all layers)
- Loads OpenVLA-OFT policy for feature extraction
- Implements speculative decoding with mode locking:
  1. Draft model predicts coarse tokens (L0-L2) - FAST
  2. Conditional Main model uses draft tokens to predict all layers (L0-L7) - ACCURATE
  3. Verify draft tokens against main model predictions
  4. Decode final tokens to actions using RFSQ decoder
- Tracks success rate, latency, acceptance rate, and adaptive behavior
- Saves detailed results to JSON

**Usage:**
```bash
# Full evaluation on libero_spatial (default)
python evaluate_phase3_rsd.py --task-suite libero_spatial --num-trials 50

# Quick test (single task, few trials)
python evaluate_phase3_rsd.py --num-trials 3

# Disable speculative decoding (baseline)
python evaluate_phase3_rsd.py --use-speculative-decoding False
```

**Environment Variables:**
- `OPENVLA_CHECKPOINT`: Path to OpenVLA-OFT checkpoint (default: moojink/openvla-7b-oft-finetuned-libero-spatial)
- `RFSQ_MODEL`: Path to trained RFSQ tokenizer (default: /models/rfsq_robust_best.pt)
- `DRAFT_MODEL`: Path to trained Draft model (default: /models/best_draft_with_projection.pt)
- `MAIN_MODEL`: Path to trained Conditional model (default: /models/openvla_rfsq_conditional/best_rfsq_head.pt)
- `RESULTS_DIR`: Output directory for results (default: ./results)
- `LOG_DIR`: Output directory for logs (default: ./logs)
- `LIBERO_PATH`: Path to LIBERO installation (default: /root/LIBERO)

**Outputs:**
- `results/{task_suite}_rsd_results.json`: Detailed results with per-task breakdown
- `logs/phase3_rsd_{task_suite}_YYYYMMDD_HHMMSS/`: Experiment logs
  - `metrics.jsonl`: Step-by-step metrics
  - `summary.json`: Final summary
  - `log.txt`: Text log

**Key Metrics:**
- Success rate (overall and per-task)
- Draft acceptance rate (how often draft tokens are correct)
- Mode locking rate (how often conditional fusion is used)
- Average inference time (draft, main, decode)
- Fallback rate (how often system falls back to vanilla OpenVLA)

---

### 2. `evaluate_openvla_baseline.py` - Baseline Evaluation with Video Recording

Evaluates vanilla OpenVLA-OFT model WITHOUT RSD components.

**What it does:**
- Loads only OpenVLA-OFT model (no RSD components)
- Runs standard policy inference on LIBERO benchmark
- Records ALL episodes as videos
- Saves ONLY failure case videos by default (for debugging)
- Computes success rate and detailed episode logs
- Serves as baseline for comparison with RSD

**Usage:**
```bash
# Full evaluation (saves only failure videos)
python evaluate_openvla_baseline.py --task-suite libero_spatial --num-trials 50

# Save all videos (successes + failures)
python evaluate_openvla_baseline.py --save-all-videos

# Disable video recording (faster)
python evaluate_openvla_baseline.py --no-videos

# Quick test
python evaluate_openvla_baseline.py --num-trials 3
```

**Environment Variables:**
- `OPENVLA_CHECKPOINT`: Path to OpenVLA-OFT checkpoint
- `RESULTS_DIR`: Output directory for results
- `LOG_DIR`: Output directory for logs
- `LIBERO_PATH`: Path to LIBERO installation

**Outputs:**
- `results/baseline_results.json`: Overall results with per-task breakdown
- `results/baseline_episode_logs.json`: Detailed logs for every episode
- `results/baseline_failures/*.mp4`: Failure case videos (default)
- `results/baseline_all/*.mp4`: All episode videos (if `--save-all-videos`)
- `logs/openvla_baseline_{task_suite}_YYYYMMDD_HHMMSS/`: Experiment logs

**Key Metrics:**
- Success rate (overall and per-task)
- Average inference time
- Number of failures (with corresponding videos)
- Per-episode success/failure status

---

## Comparison Workflow

To compare RSD against baseline:

```bash
# 1. Run baseline evaluation
python evaluate_openvla_baseline.py --task-suite libero_spatial --num-trials 50

# 2. Run RSD evaluation
python evaluate_phase3_rsd.py --task-suite libero_spatial --num-trials 50

# 3. Compare results
python -c "
import json
baseline = json.load(open('results/baseline_results.json'))
rsd = json.load(open('results/libero_spatial_rsd_results.json'))

print(f'Baseline Success Rate: {baseline[\"final_success_rate\"]:.1%}')
print(f'RSD Success Rate: {rsd[\"final_success_rate\"]:.1%}')
print(f'Draft Acceptance Rate: {rsd[\"rsd_stats\"][\"draft_acceptance_rate\"]:.1%}')
"
```

---

## Dependencies

Both scripts require:
- PyTorch with CUDA
- OpenVLA-OFT (from openvla-oft repo)
- LIBERO (robosuite==1.4.0)
- Transformers, timm, pillow, etc.

**SLURM Setup:**
See `slurm/batch_scripts/` for SLURM job scripts that handle all dependencies.

---

## Model Files Required

### For RSD Evaluation (`evaluate_phase3_rsd.py`):
1. **RFSQ Tokenizer** (`rfsq_robust_best.pt`):
   - Trained in Phase 1
   - Path: `$RFSQ_MODEL` or `/models/rfsq_robust_best.pt`

2. **Draft Model** (`best_draft_with_projection.pt`):
   - Trained in Phase 2 (predicts L0-L2)
   - Path: `$DRAFT_MODEL` or `/models/best_draft_with_projection.pt`

3. **Conditional Main Model** (`best_rfsq_head.pt`):
   - Trained in Phase 2 (predicts all layers with conditioning)
   - Path: `$MAIN_MODEL` or `/models/openvla_rfsq_conditional/best_rfsq_head.pt`

4. **OpenVLA-OFT Checkpoint**:
   - HuggingFace model ID or local path
   - Default: `moojink/openvla-7b-oft-finetuned-libero-spatial`

### For Baseline Evaluation (`evaluate_openvla_baseline.py`):
1. **OpenVLA-OFT Checkpoint** (same as above)

---

## Understanding the Results

### RSD Results JSON Structure:
```json
{
  "task_suite": "libero_spatial",
  "use_speculative_decoding": true,
  "model_type": "conditional_rsd",
  "total_episodes": 500,
  "total_successes": 425,
  "final_success_rate": 0.85,
  "task_results": [
    {
      "task_id": 0,
      "task_description": "put the spatula on the plate",
      "successes": 45,
      "episodes": 50,
      "success_rate": 0.9
    },
    ...
  ],
  "rsd_stats": {
    "total_predictions": 12500,
    "draft_acceptance_rate": 0.72,
    "mode_locking_rate": 1.0,
    "avg_draft_time_ms": 8.5,
    "avg_main_time_ms": 12.3,
    "avg_decode_time_ms": 2.1
  }
}
```

### Baseline Results JSON Structure:
```json
{
  "task_suite": "libero_spatial",
  "model_type": "openvla_baseline",
  "total_episodes": 500,
  "total_successes": 400,
  "final_success_rate": 0.8,
  "task_results": [...],
  "episode_logs": [
    {
      "task_id": 0,
      "trial_idx": 0,
      "success": false,
      "steps": 220
    },
    ...
  ],
  "policy_stats": {
    "total_predictions": 12000,
    "avg_inference_time_ms": 25.4
  }
}
```

---

## Troubleshooting

### ImportError: Cannot import openvla utilities
**Solution:** Make sure openvla-oft is installed:
```bash
cd /path/to/openvla-oft
pip install -e .
```

### Model file not found
**Solution:** Check environment variables or use `--rfsq-model`, `--draft-model`, `--main-model` flags

### LIBERO not found
**Solution:** Set `LIBERO_PATH` environment variable:
```bash
export LIBERO_PATH=/path/to/LIBERO
```

### GPU OOM
**Solution:** Reduce batch size or use smaller models (not applicable for evaluation, batch size is always 1)

### Video recording fails
**Solution:** Install imageio with ffmpeg:
```bash
pip install imageio[ffmpeg]
```

---

## Performance Tips

1. **Use CUDA:** Both scripts run much faster on GPU
2. **Disable videos for quick testing:** Use `--no-videos` flag
3. **Reduce trials for debugging:** Use `--num-trials 3` for quick tests
4. **Check model loading time:** First run is slower due to model downloads

---

## Expected Runtime

On A100 GPU:
- **Baseline:** ~30-60 minutes for 500 episodes (10 tasks × 50 trials)
- **RSD:** ~30-60 minutes for 500 episodes (similar to baseline due to efficient speculative decoding)

The actual runtime depends on:
- Task complexity (some tasks fail early, others run full 220 steps)
- GPU availability
- Video recording overhead (~10% slowdown)
