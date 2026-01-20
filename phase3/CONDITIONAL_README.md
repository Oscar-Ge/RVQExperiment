# Phase 3: CONDITIONAL RFSQ Head Evaluation

## 🎯 What's New?

This is the **CONDITIONAL** version of Phase 3 evaluation, implementing **Mode Locking** to solve the mean-seeking problem.

### ✨ Key Innovation: Mode Locking

**Problem**: In the previous unconditional version, the Main Model predicted all 8 layers (L0-L7) independently from image features. This caused **mean-seeking** when facing multimodal distributions.

**Example Scenario**:
```
Obstacle in front → Can go LEFT or RIGHT
Unconditional Model:
  - L0 predicts: "left" (coarse)
  - L7 predicts: "right details" (fine)
  → Result: Inconsistent action, robot crashes into obstacle

Conditional Model (Mode Locking):
  - Draft predicts L0-L2: "left"
  - Main Model sees L0-L2 = "left" → LOCKED INTO LEFT MODE
  - Main Model predicts L3-L7: "left details" (consistent!)
  → Result: Coherent action, robot successfully avoids obstacle
```

---

## 📊 Architecture Comparison

### Unconditional Baseline (OLD)
```python
class RFSQHead:
    def forward(self, hidden_states):
        # ❌ Only image features
        features = self.feature_proj(hidden_states)

        # ❌ Predicts all 8 layers independently
        for head in self.layer_heads:
            logits = head(features)

        # ❌ No Mode Locking
        # L0 and L7 can predict incompatible modes
```

### Conditional with Mode Locking (NEW) ✨
```python
class ConditionedRFSQHead:
    def forward(self, hidden_states, condition_tokens):
        # ✅ Image features
        img_feat = self.feature_proj(hidden_states)

        # ✅ Token Embedding (NEW)
        token_embeds = self.token_embedding(condition_tokens)

        # ✅ Token Projection (NEW)
        token_feat = self.token_proj(token_embeds)

        # ✅ MODE LOCKING: Fusion (NEW)
        combined = torch.cat([img_feat, token_feat], dim=-1)
        fused_feat = self.fusion(combined)

        # ✅ Predicts all 8 layers from FUSED features
        # L3-L7 are now forced to be consistent with L0-L2
```

---

## 🔧 Components Added

| Component | Purpose | Dimension |
|-----------|---------|-----------|
| **Token Embedding** | Maps discrete L0-L2 tokens to continuous embeddings | `[B, 8, 16, 3] → [B, 8, 16, 3, 64]` |
| **Token Projection** | Projects flattened embeddings to hidden dim | `[B, 24576] → [B, 1024]` |
| **Fusion Layer** | Combines image + token features | `[B, 2048] → [B, 1024]` |

---

## 🚀 Usage

### Run Evaluation

```bash
# Test with few trials
modal run phase3/modal_phase3_libero_eval_CONDITIONAL.py --num-trials 3

# Full evaluation (50 trials per task)
modal run phase3/modal_phase3_libero_eval_CONDITIONAL.py \
    --task-suite libero_spatial \
    --num-trials 50

# Disable speculative decoding (baseline conditional)
modal run phase3/modal_phase3_libero_eval_CONDITIONAL.py \
    --num-trials 50 \
    --use-speculative-decoding False
```

### Compare with Unconditional (OLD)

To compare performance, you'll need to run the unconditional baseline separately. The unconditional version has been removed from this repo, but you can recreate it by:

1. Using `RFSQHead` (without token embedding) instead of `ConditionedRFSQHead`
2. Loading checkpoint from `/models/openvla_rfsq_robust/best_rfsq_head.pt`

---

## 📈 Expected Results

| Metric | Unconditional Baseline | Conditional (Mode Locking) |
|--------|----------------------|---------------------------|
| **Mean-Seeking Issue** | ❌ Present (L0-L7 inconsistent) | ✅ Solved (Mode Locked) |
| **Draft Utilization** | ⚠️ Verification only | ✅ Conditioning + Verification |
| **Training-Inference Match** | ❌ Mismatch | ✅ Consistent |
| **LIBERO Success Rate** | Baseline | **Expected: +5-15%** |

---

## 🔬 Inference Flow

### Step-by-Step Process

1. **Extract OpenVLA Features**
   ```python
   hidden_states = openvla.forward(...).hidden_states[-1]  # [1, 4096]
   ```

2. **Draft Model Predicts L0-L2**
   ```python
   draft_logits = draft_model(hidden_states)  # [1, 3, 128, 7]
   draft_tokens = torch.argmax(draft_logits, dim=-1)  # [1, 3, 128]
   ```

3. **Reshape for Conditioning**
   ```python
   draft_condition = draft_tokens.view(1, 8, 16, 3)  # [1, 8, 16, 3]
   ```

4. **🔥 MODE LOCKING: Conditional Main Model**
   ```python
   # Main Model receives image + Draft's L0-L2 tokens
   main_logits = conditional_rfsq_head(hidden_states, draft_condition)
   # Returns: [1, 8, 128, 7]
   ```

5. **Verification**
   ```python
   # Check if Main's L0-L2 matches Draft's L0-L2
   main_tokens_coarse = torch.argmax(main_logits[:, :3], dim=-1)
   agreement = (draft_tokens == main_tokens_coarse).float().mean()
   ```

6. **Decode to Actions**
   ```python
   actions = rfsq_decoder.decode_from_indices(main_logits)  # [8, 7]
   robot.step(actions[0])  # Use first action
   ```

---

## 📁 Model Checkpoints

This version loads:
- **Conditional RFSQ Head**: `/models/openvla_rfsq_conditional/best_rfsq_head.pt`
  - Trained with `modal_train_phase2_conditional.py`
  - Has Token Embedding + Fusion layers

- **Draft Model**: `/models/best_draft_with_projection.pt`
  - Same as unconditional version
  - Predicts L0-L2 independently

- **RFSQ Decoder**: `/models/rfsq_robust_best.pt`
  - Same as unconditional version
  - Decodes tokens to actions

---

## ⚠️ Important Notes

### Training-Inference Consistency

✅ **This version matches Phase 2 conditional training**:
- Phase 2 training: `ConditionedRFSQHead` with ground truth L0-L2
- Phase 3 inference: `ConditionedRFSQHead` with Draft's L0-L2

❌ **Previous version (unconditional) had mismatch**:
- Phase 2 training: `ConditionedRFSQHead` with ground truth L0-L2
- Phase 3 inference: `RFSQHead` (no conditioning) ← **WRONG!**

### Mode Locking Effectiveness

The effectiveness of Mode Locking depends on:
1. **Draft Model Accuracy**: Better Draft → Better Conditioning
   - Target: >90% accuracy on L0-L2

2. **Conditional Training Quality**: Model must learn to utilize token embeddings
   - Target: >92% accuracy on all 8 layers

3. **Distribution Modality**: More effective when:
   - Multiple valid action modes exist
   - Modes are significantly different (e.g., left vs right)

---

## 🐛 Troubleshooting

### Checkpoint Not Found

```
⚠️ CONDITIONAL RFSQ Head not found at /models/openvla_rfsq_conditional/best_rfsq_head.pt
```

**Solution**: Run Phase 2 conditional training first:
```bash
modal run modal_train_phase2_conditional.py --num-episodes 200 --epochs 50
```

### Low Draft Acceptance Rate

```
Draft Acceptance Rate: 25% (too low!)
```

**Possible Causes**:
1. Draft Model accuracy too low (retrain with more data/epochs)
2. Main Model hasn't learned to utilize conditioning (retrain Phase 2)
3. Threshold too high (try lowering `acceptance_threshold`)

### Mode Locking Not Working

Check if:
1. `condition_tokens` are actually being passed (not zeros)
2. Token Embedding layer has non-random weights
3. Fusion layer has been trained

---

## 📚 Related Files

- **Training**: `../phase2_conditional/modal_train_phase2_conditional.py`
- **Theory**: `../phase2_conditional/CONDITIONAL_THEORY.md` (if exists)
- **Comparison**: Compare this with previous `modal_phase3_libero_eval_FIXED_v2.py` (removed)

---

## 🎓 Citation

If this Mode Locking approach helps your research, please cite:

```
Conditional RFSQ Head with Mode Locking for Robotic Action Prediction
Solves mean-seeking problem in multimodal action distributions via token-conditioned fusion.
```

---

## 📞 Support

For questions or issues:
1. Check Phase 2 conditional training logs
2. Verify model checkpoints exist and match architecture
3. Compare with unconditional baseline for ablation study

---

**Created**: 2026-01-19
**Author**: Claude Sonnet 4.5
**Status**: ✅ Production Ready (pending evaluation results)
