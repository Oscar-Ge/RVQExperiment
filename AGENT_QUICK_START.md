# Agent Quick Start Guide

**Purpose**: Train Phase 1 (Robust RFSQ) → Phase 2 (Draft Model) → Phase 3 (Evaluation) step-by-step to avoid token waste.

**Context**: Previous experiment used Naive RFSQ (broken, only 3/8 layers effective). Now using Robust RFSQ with LayerNorm strategy (all 8 layers effective, -44% MSE).

---

## 📦 Setup

### Clone Repository
```bash
git clone https://github.com/Oscar-Ge/RVQExperiment.git
cd RVQExperiment
```

### Files You Need to Know
- `phase1_improved/` - Robust RFSQ training code (NEW, needs training)
- `phase2_draft_retrain/` - Draft model training (updated to use Robust RFSQ)
- `phase3/` - LIBERO evaluation (updated to use Robust RFSQ)
- `mock_test/` - Integration tests (run before expensive training)

### Modal Volume Structure
```
/models/
├── rfsq_robust_best.pt              # ← Phase 1 output (YOU WILL CREATE)
├── best_draft_with_projection.pt    # ← Phase 2 output (YOU WILL CREATE)
└── openvla_rfsq_robust/
    └── best_rfsq_head.pt            # ← Phase 2 output (YOU WILL CREATE)
```

---

## 🧪 Step 0: Pre-flight Check (RECOMMENDED)

**Why**: Previous experiment failed at integration. Run mock test first (35 seconds) to avoid wasting 10+ hours of training.

```bash
# Generate mock checkpoints
python mock_test/generate_mock_checkpoints.py --output-dir ./mock_models

# Run integration test
python mock_test/test_phase3_integration.py --models-dir ./mock_models --device cpu
```

**Expected**: All tests pass ✅

**If failed**: Fix integration issues BEFORE training on A100.

---

## 🔥 Step 1: Train Robust RFSQ

### 1.1 What This Does
- Trains an 8-layer RFSQ encoder/decoder with LayerNorm strategy
- Output: `rfsq_robust_best.pt` (will be used by Phase 2 and Phase 3)
- Time: 2-3 hours on A100

### 1.2 Training Script Location
`phase1_improved/modal_train_rfsq_robust.py`

### 1.3 Run Training
```bash
# On Modal (or your GPU environment)
modal run phase1_improved/modal_train_rfsq_robust.py
```

### 1.4 Monitor
- Training logs will show MSE decreasing
- Target MSE: < 0.012 (Naive RFSQ was 0.018)
- Expected MSE: ~0.010

### 1.5 Verify Output
Check that `/models/rfsq_robust_best.pt` exists in Modal volume with:
- `model_state_dict` - RFSQ model weights
- `train_mse` - Training MSE (should be < 0.012)

**Checkpoint**: Phase 1 complete when `rfsq_robust_best.pt` exists and MSE < 0.012.

---

## 🎯 Step 2: Train Draft Model + Main Model RFSQ Head

### 2.1 What This Does
- Trains Draft Model (predicts coarse layers L0-L2)
- Trains Main Model RFSQ Head (predicts all layers L0-L7)
- Uses `rfsq_robust_best.pt` from Phase 1 to generate token labels
- Output: `best_draft_with_projection.pt` and `openvla_rfsq_robust/best_rfsq_head.pt`
- Time: 6-10 hours on A100

### 2.2 Training Script Location
`phase2_draft_retrain/modal_train_draft_with_projection.py`

### 2.3 Prerequisites
⚠️ **MUST HAVE**: `/models/rfsq_robust_best.pt` from Step 1

### 2.4 Run Training
```bash
# On Modal
modal run phase2_draft_retrain/modal_train_draft_with_projection.py
```

### 2.5 Monitor
- Draft Model accuracy (coarse layers L0-L2): target > 90%
- Main Model accuracy (all layers L0-L7): target > 92%

### 2.6 Verify Output
Check Modal volume `/models/` contains:
- `best_draft_with_projection.pt` - Draft model checkpoint
- `openvla_rfsq_robust/best_rfsq_head.pt` - Main model RFSQ head

**Checkpoint**: Phase 2 complete when both checkpoints exist with good accuracy.

---

## 🤖 Step 3: Run LIBERO Evaluation

### 3.1 What This Does
- Loads all 3 checkpoints (Phase 1, Phase 2 outputs)
- Runs RSD (Residual Speculative Decoding) pipeline
- Evaluates on LIBERO benchmark (10 tasks, 50 episodes each)
- Output: Success rates, action metrics, logs
- Time: 2-3 hours

### 3.2 Evaluation Script Location
`phase3/modal_phase3_libero_eval.py`

### 3.3 Prerequisites
⚠️ **MUST HAVE**:
- `/models/rfsq_robust_best.pt` (Phase 1)
- `/models/best_draft_with_projection.pt` (Phase 2)
- `/models/openvla_rfsq_robust/best_rfsq_head.pt` (Phase 2)

### 3.4 Run Evaluation
```bash
# On Modal
modal run phase3/modal_phase3_libero_eval.py
```

### 3.5 Expected Results
- Overall success rate: > 90% (previous Naive RFSQ: 87%)
- Fine-grained task success rate: > 85% (previous: 75-78%)
- Token acceptance rate: 80-85%

### 3.6 Verify Output
Check evaluation logs for:
- Per-task success rates
- Overall metrics
- No shape mismatch errors
- No RFSQ decoding failures

**Checkpoint**: Phase 3 complete when evaluation runs successfully with improved success rates.

---

## 🚨 Troubleshooting

### Issue: Phase 2 can't load RFSQ checkpoint
**Error**: `FileNotFoundError: /models/rfsq_robust_best.pt`
**Fix**: Complete Step 1 first. Phase 2 depends on Phase 1 output.

### Issue: Phase 3 can't load checkpoints
**Error**: Missing checkpoint files
**Fix**: Complete Steps 1 and 2 first. Phase 3 needs all checkpoints.

### Issue: Shape mismatch in Phase 3
**Error**: `RuntimeError: size mismatch`
**Fix**: Run mock test (Step 0) to verify integration BEFORE training.

### Issue: Low success rate in Phase 3
**Symptom**: Success rate < 85%
**Debug**:
1. Check Phase 1 MSE: should be < 0.012
2. Check Phase 2 accuracies: Draft > 90%, Main > 92%
3. Check logs for RFSQ decoding errors

---

## 📊 Expected Timeline

| Phase | Time | Cost (A100) | Output |
|-------|------|-------------|--------|
| **Step 0** (Mock Test) | 35 sec | $0.01 | Integration verified |
| **Step 1** (Robust RFSQ) | 2-3 hrs | $5-8 | rfsq_robust_best.pt |
| **Step 2** (Draft + Main) | 6-10 hrs | $15-25 | Draft + Main checkpoints |
| **Step 3** (Evaluation) | 2-3 hrs | $5-8 | Success rates, metrics |
| **TOTAL** | ~10-16 hrs | **$25-41** | Complete evaluation |

---

## 🎯 Success Criteria

### Phase 1 Success ✅
- `rfsq_robust_best.pt` exists
- MSE < 0.012 (target: ~0.010)

### Phase 2 Success ✅
- Both checkpoints exist
- Draft accuracy > 90%
- Main accuracy > 92%

### Phase 3 Success ✅
- Overall success rate > 90%
- Fine-grained tasks > 85%
- No integration errors

---

## 💡 Tips for Agent

1. **Run Step 0 first**: Mock test takes 35 seconds, can save 10+ hours if integration is broken.

2. **Check dependencies**: Each phase depends on previous outputs. Don't skip steps.

3. **Monitor logs**: Watch for MSE/accuracy during training. If numbers look bad, stop early.

4. **Verify checkpoints**: After each phase, verify checkpoint files exist in Modal volume.

5. **Compare results**: Compare Phase 3 results with Naive RFSQ baseline:
   - Naive RFSQ: 87% overall, 75-78% fine-grained
   - Robust RFSQ target: 90-92% overall, 85%+ fine-grained

---

## 📖 Detailed Documentation

If you need more context:
- `MIGRATION_TO_ROBUST_RFSQ.md` - Why we're doing this
- `AGENT_ACTION_PLAN.md` - Detailed action plan
- `phase1_improved/AGENT_GUIDE.md` - Phase 1 detailed guide
- `mock_test/README.md` - Mock testing guide

---

**Ready to start? Begin with Step 0 (Mock Test) → Step 1 (Train RFSQ) → Step 2 (Train Draft/Main) → Step 3 (Evaluate)**
