# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains experimental code for **Adaptive Inference for Vision-Language-Action Models via Residual Speculative Decoding**, a research project that explores compressing robot action sequences using DCT and RVQ (Residual Vector Quantization) tokenizers, then implementing adaptive inference to dynamically adjust computational cost based on task complexity.

**Research Goal**: Achieve 30-50% inference speedup with <3% accuracy loss by using coarse RVQ layers (1-2) for simple motions and fine layers (1-8) for complex manipulations.

## Repository Structure

The codebase is organized into 3 phases:

### Phase 1: RVQ Tokenizer (✅ Completed)
- **Core Files**: `rvq_tokenizer.py`, `train_rvq_tokenizer.py`, `analyze_rvq_compression.py`, `test_rvq_tokenizer.py`
- **Baseline**: `analyze_libero_actions.py` (DCT compression baseline)
- **Guide**: `RVQ_PHASE1_GUIDE.md`

### Phase 2: VLA-RVQ Training (⚠️ Architecture complete, needs π0.5 integration)
- **Core Files**: `train_vla_rvq_policy.py`, `pi05_feature_extractor.py`, `test_pi05_integration.py`
- **Guide**: `PHASE2_3_GUIDE.md`, `PI05_INTEGRATION_GUIDE.md`

### Phase 3: Adaptive Inference (✅ Implementation complete, needs trained model)
- **Core Files**: `adaptive_inference.py`, `evaluate_adaptive_policy.py`
- **Guide**: `PHASE2_3_GUIDE.md`

### Supporting Infrastructure
- **`basic-run/`**: π0.5 LIBERO benchmark baseline code
- **`openpi/`**: OpenPI library (π0.5 policy implementation)

## Key Commands

### Phase 1: RVQ Tokenizer Training
```bash
# Unit tests (verify implementation)
python test_rvq_tokenizer.py

# Train RVQ tokenizer on LIBERO actions
python train_rvq_tokenizer.py \
    --task_suite libero_spatial \
    --num_episodes 50 \
    --epochs 100 \
    --device cuda

# Analyze compression performance
python analyze_rvq_compression.py \
    --model rvq_tokenizer.pt \
    --num_episodes 20 \
    --device cuda

# DCT baseline (for comparison)
python analyze_libero_actions.py \
    --task_suite libero_spatial \
    --num_episodes 20
```

### Phase 2: VLA-RVQ Policy Training
```bash
# Test π0.5 integration (critical first step)
python test_pi05_integration.py --device cuda --rvq_model rvq_tokenizer.pt

# Inspect π0.5 model structure (if feature extraction fails)
python test_pi05_integration.py --inspect --device cuda

# Train VLA policy to predict RVQ tokens
python train_vla_rvq_policy.py \
    --rvq_model rvq_tokenizer.pt \
    --num_episodes 100 \
    --epochs 50 \
    --freeze_backbone \
    --device cuda
```

### Phase 3: Adaptive Inference Evaluation
```bash
# Evaluate adaptive policy
python evaluate_adaptive_policy.py \
    --rvq_model rvq_tokenizer.pt \
    --vla_model vla_rvq_policy.pt \
    --num_episodes 50 \
    --monitor_types entropy,variance \
    --device cuda
```

## Architecture Overview

### RVQ Tokenizer (Phase 1)
```
Actions [chunk_size, 7]
  ↓ Encoder
Hidden [chunk_size, 64]
  ↓ Residual VQ (8 layers)
Discrete Tokens [8 layers × chunk_size]
  ↓ Decoder
Reconstructed Actions [chunk_size, 7]
```

**Key Insight**: Layer 1-2 capture coarse motion (MSE ≈ 0.01), Layer 3-8 capture fine details (MSE < 0.001).

### VLA-RVQ Policy (Phase 2)
```
[Image + Language + State]
  ↓ π0.5 Backbone (Vision-Language Encoder)
Features [batch, feature_dim]  ← Extracted via hooks
  ↓ RVQ Prediction Head (new, trainable)
Logits [batch, 8 layers, chunk_size, 7 dims, 256 vocab]
  ↓ Cross-Entropy Loss
Discrete RVQ Tokens
  ↓ RVQ Decoder (frozen)
Actions [chunk_size, 7]
```

**Critical Component**: `pi05_feature_extractor.py` uses PyTorch forward hooks to extract intermediate features from π0.5's encoder layers.

### Adaptive Inference (Phase 3)
```
1. Draft: Predict coarse tokens (Layer 1-2) → Fast
2. Monitor: Compute complexity metric (entropy/variance/distance)
3. Refine: If metric > threshold, predict all tokens (Layer 1-8) → Accurate
```

**Monitor Strategies**:
- **Entropy**: Prediction uncertainty (logits entropy)
- **Variance**: Action stability (recent action variance)
- **Distance**: Proximity to object (state-based)

## Critical Implementation Details

### RVQ Tokenizer Architecture
- **VectorQuantizer**: Single-layer VQ with EMA updates (`decay=0.99`)
- **ResidualVectorQuantizer**: Multi-layer RVQ where each layer quantizes the residual from previous layers
- **Residual Dropout**: During training, randomly drop deep layers (`prob=0.1`) to force shallow layers to encode more information
- **Action Normalization**: Normalize to [-1, 1] before encoding, denormalize after decoding
- **DCT Coefficient Range Bug**: Previously assumed DCT coefficients ∈ [-1, 1], but actual range is ≈ [-4, 4] due to DC component. RVQ learns this automatically via `fit()`.

### π0.5 Feature Extraction
The `Pi05FeatureExtractor` class uses a hook-based approach with 3 fallback strategies:
1. **Vision encoder hook** (priority): Hook after `vision_encoder` or `visual_encoder`
2. **Language model backbone**: Hook at middle transformer layer
3. **Fallback**: Hook first large Linear layer (out_features ≥ 512)

**Usage Pattern**:
```python
extractor = Pi05FeatureExtractor(pi05_policy)
features = extractor.extract_features(obs_dict)  # [batch, feature_dim]
# feature_dim is auto-detected (typically 512-2048)
```

### Training Considerations
- **Phase 1**: Typical training time 30-60 min (50 episodes, 100 epochs)
- **Phase 2**: Requires 100+ episodes for freeze_backbone, 500+ for fine-tuning; training time 2-4 hours
- **Target Metrics**:
  - Phase 1: MSE < 0.01 for Layer 1-2, MSE < 0.001 for Layer 1-8
  - Phase 2: Token prediction accuracy > 70%
  - Phase 3: Success rate ≈ Dense RVQ, Latency ↓ 30-50%, Adaptive rate 20-30%

## Common Issues and Solutions

### Issue: Feature extraction fails
**Symptom**: `RuntimeError: Failed to extract features`
**Solution**:
1. Run `python test_pi05_integration.py --inspect` to view model structure
2. Manually configure hooks in `pi05_feature_extractor.py::_setup_feature_extraction()`

### Issue: RVQ MSE too high
**Symptom**: MSE > 0.1 even with all layers
**Causes**:
- Insufficient training (increase epochs)
- Model capacity too small (increase `hidden_dim` or `codebook_size`)
- Wrong chunk_size (π0.5-libero uses chunk_size=10, not 16)

### Issue: GPU OOM during training
**Solutions**:
- Reduce batch_size: `--batch_size 8`
- Reduce hidden_dim: `--hidden_dim 256`
- Use gradient accumulation (requires code modification)

### Issue: Token prediction accuracy < 50%
**Causes**:
- Poor feature extraction (features may be all zeros or constant)
- Insufficient training data
- Learning rate mismatch

**Debug**:
```python
# Check feature quality
print(f"Features mean: {features.mean()}")  # Should not be 0
print(f"Features std: {features.std()}")    # Should not be 0
```

## Data Flow

### Training Data Collection
1. Run π0.5 policy on LIBERO environment
2. Collect action chunks (10 steps × 7 dimensions)
3. Encode actions to RVQ tokens using trained tokenizer
4. Store observations + prompts + actions + tokens

### Action Representation
- **Chunk size**: 10 steps (π0.5-libero config)
- **Action dimensions**: 7 (x, y, z, roll, pitch, yaw, gripper)
- **State representation**: Concatenate [eef_pos, quat2axisangle(eef_quat), gripper_qpos]

### Evaluation Metrics
- **Success Rate**: % of episodes that complete task
- **Latency**: Average inference time per step
- **Speedup**: Baseline latency / Adaptive latency
- **Adaptive Rate**: % of steps using fine layers (target: 20-30%)

## Dependencies

**Core**:
- PyTorch (with CUDA)
- NumPy, SciPy (≥1.4.0 for DCT)
- Matplotlib (visualization)

**π0.5/LIBERO**:
- openpi, openpi_client
- LIBERO (robosuite==1.4.0)
- basic-run scripts (π0.5 benchmark utilities)

**Environment Setup**:
```bash
# Verify scipy DCT support
python -c "from scipy.fft import dct, idct; print('✅ scipy DCT available')"

# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Testing Strategy

### Unit Tests
- `test_rvq_tokenizer.py`: Tests VQ, RVQ, round-trip encoding, different layer counts
- `test_pi05_integration.py`: Tests π0.5 loading, inference, feature extraction, VLA-RVQ creation

### Integration Tests
Run full pipeline end-to-end on small dataset (5-10 episodes) before full training.

### Validation
- **Phase 1**: Visual inspection of `rvq_compression_analysis.png` (MSE vs layers plot)
- **Phase 2**: Monitor token accuracy during training (should reach >70%)
- **Phase 3**: Compare `adaptive_comparison.png` against baselines

## File Naming Conventions

- `*_tokenizer.py`: Tokenizer implementations
- `train_*.py`: Training scripts
- `analyze_*.py`: Analysis/evaluation scripts
- `test_*.py`: Unit tests
- `*_GUIDE.md`: User documentation
- `*.pt`: Saved model checkpoints

## Known Limitations

1. **π0.5 API Access**: Requires modifying OpenPI code or using hooks to extract features (not officially supported)
2. **Action Chunk Size**: π0.5-libero uses 10-step chunks (not 16 as initially assumed)
3. **Training Data**: Freeze backbone approach requires 100+ episodes; fine-tuning requires 500+
4. **Generalization**: Threshold tuning may be task-specific; single-task training may not generalize

## Research Context

This is a proof-of-concept for **Residual Speculative Decoding** applied to VLA models. The core hypothesis: VLA models can be accelerated by recognizing that robot manipulation has varying complexity - simple reaching motions only need coarse predictions (Layer 1-2), while precise grasping needs fine predictions (Layer 1-8).

**Key Innovation**: First work to leverage RVQ's hierarchical structure for speculative decoding in VLA, enabling task-aware adaptive computation.
