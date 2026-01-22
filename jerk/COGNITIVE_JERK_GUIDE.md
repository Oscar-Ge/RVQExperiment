# Cognitive Jerk Analysis Guide

## Overview

The cognitive jerk analysis script (`analyze_cognitive_jerk.py`) is a research verification tool that tracks how the VLA model's internal state and action plans change during LIBERO task execution.

**Key Innovation**: Dense prediction mode that decouples monitoring frequency from control frequency, enabling detection of when the model "changes its mind" between action chunks.

## What It Measures

### 1. Cognitive Jerk (Dense - every step)
**Definition**: Cosine distance between consecutive 4096-dimensional hidden states from the final transformer layer.

**Interpretation**:
- Low values (0.01-0.1): Model state is stable
- High values (>0.3): Significant cognitive shift occurring
- Expected spikes: At object contact, grasp moments, task transitions

**Formula**: `jerk = 1 - cosine_similarity(hidden[t], hidden[t-1])`

### 2. Action Plan Instability (Dense - every step)
**Definition**: L2 norm of the difference between the overlapping regions of consecutive action plans.

**Interpretation**:
- Low values: Model has consistent plan from step to step
- High values: Model is "changing its mind" - re-planning even when not executing new actions
- Temporal consistency check: `||plan[t][0:7] - plan[t-1][1:8]||`

**Why This Matters**: Since we predict actions at every step but only execute every 8 steps, this reveals whether the model would have done something different if we asked it mid-execution.

### 3. Gripper Events
**Definition**: Detection of gripper state transitions (open ↔ closed).

**Interpretation**:
- Closing events: Model attempting to grasp object
- Opening events: Model releasing object
- Proxy for task progress and manipulation moments

## Usage

### Basic Run
```bash
modal run analyze_cognitive_jerk.py
```

### Expected Runtime
- Single task, single trial: 2-5 minutes
- ~230 timesteps total (10 stabilization + up to 220 execution)
- GPU: A100 (required for 7B VLA model)

## Outputs

### 1. Visualization (`jerk_analysis_task0.png`)
4-panel time-series plot:

**Panel 1: Cognitive Jerk**
- Blue line: dense jerk values
- Red dots: high jerk moments (>90th percentile)
- Gray dashed lines: prediction timesteps (queue refills)

**Panel 2: Action Plan Instability**
- Purple line: temporal consistency metric
- Red dots: high instability moments
- Shows when model is "changing mind"

**Panel 3: Gripper State**
- Orange line: gripper position over time
- Red triangles (▼): closing events
- Green triangles (▲): opening events

**Panel 4: Timeline**
- Purple ticks: when new actions predicted
- Star: final outcome (green = success, red = failure)

### 2. Raw Data (`jerk_metrics_task0.json`)
JSON file containing all metrics for later analysis:
```json
{
  "timesteps": [10, 11, 12, ...],
  "cognitive_jerk": [0.021, 0.018, ...],
  "action_change": [0.5, 0.3, ...],
  "gripper_state": [-0.8, -0.7, ...],
  "gripper_events": [null, "closing", ...],
  "prediction_steps": [10, 18, 26, ...],
  "task_description": "put the red mug on the plate",
  "success": true
}
```

### 3. Console Summary
Printed statistics including:
- Mean, std, min, max, 90th percentile for all metrics
- Gripper event counts
- **Correlation analysis**: Cognitive jerk ↔ action instability
  - Strong correlation (>0.5): Validates hypothesis that jerk → instability
  - Weak correlation: Suggests independent factors

## Expected Results

### Typical Values
- **Cognitive Jerk**: 0.01-0.5 (typical), >0.3 (high)
- **Action Instability**: 0.1-2.0 (typical), >2.0 (high)
- **Gripper Range**: -1.0 (open) to +1.0 (closed)

### Hypothesis Validation
If the hypothesis is correct, you should observe:
1. ✅ Cognitive jerk spikes at task-critical moments (grasp, release)
2. ✅ Action instability spikes correlate with jerk spikes
3. ✅ Gripper events often co-occur with high jerk

## Architecture Details

### Dense Prediction Strategy
```python
for step in range(max_steps):
    # 1. Heavy compute: extract 4096-dim hidden state
    hidden = extract_hidden_state(obs, task_desc)  # 7B backbone forward pass

    # 2. Cheap compute: predict hypothetical actions
    plan = predict_actions(obs, task_desc)  # Lightweight action head

    # 3. Compute metrics (dense)
    jerk = cosine_distance(hidden[t], hidden[t-1])
    instability = norm(plan[t][:-1] - plan[t-1][1:])

    # 4. Control (sparse - only refill when empty)
    if queue.empty():
        queue.extend(plan)

    # 5. Execute action
    action = queue.pop()
    env.step(action)
```

### Why Dense Prediction?
- **Monitoring frequency**: Every step (1 Hz at policy frequency)
- **Control frequency**: Every 8 steps (chunk size)
- **Key insight**: Since we're already running the expensive 7B backbone at every step for hidden states, the marginal cost of running the action head (<1ms) is negligible
- **Benefit**: Reveals model's "internal monologue" - what it would do if queried mid-execution

## Troubleshooting

### Issue: No cognitive jerk data
**Cause**: Hidden state extraction failing
**Solution**: Check that OpenVLA model loaded correctly, verify GPU available

### Issue: Action instability always near zero
**Cause**: Model predicting identical plans at every step
**Interpretation**: Either the task is very simple, or the model is "stuck"

### Issue: High jerk but low instability (or vice versa)
**Interpretation**: Hidden state changing but action plan stable (or opposite)
- Could indicate robustness (plan stable despite internal changes)
- Or could indicate issue with feature extraction

### Issue: Correlation analysis shows negative correlation
**Interpretation**: Unusual - suggests jerk and instability are decoupled
- May indicate model architecture issue
- Or task-specific behavior (e.g., deliberate re-planning at stable moments)

## Research Insights

### What This Reveals About VLA Models
1. **Cognitive dynamics**: How internal representations evolve during manipulation
2. **Plan stability**: Whether the model commits to plans or constantly re-evaluates
3. **Task difficulty**: High jerk/instability → model uncertain
4. **Critical moments**: Spikes reveal when model "concentrates"

### Potential Applications
- **Adaptive inference**: Use jerk as confidence signal for speculative decoding
- **Failure prediction**: High instability may predict upcoming errors
- **Task difficulty estimation**: Aggregate jerk as proxy for task complexity
- **Model comparison**: Compare jerk profiles across different VLA architectures

## Next Steps

### Extending the Analysis
1. **Multi-task analysis**: Run on all 10 LIBERO-Spatial tasks
2. **Success vs failure**: Compare jerk patterns for successful vs failed episodes
3. **Layer-wise analysis**: Track hidden states from multiple layers
4. **Attention visualization**: Correlate jerk with attention map changes

### Integration with Adaptive Inference
Use jerk thresholds to trigger refinement:
```python
if cognitive_jerk > threshold:
    # Use full RVQ layers (L0-L7) for precise control
else:
    # Use coarse layers (L0-L2) for efficiency
```

## Citation
If you use this analysis in research, please cite:
```
@misc{cognitive-jerk-analysis,
  title={Cognitive Jerk Analysis for Vision-Language-Action Models},
  note={Part of RVQ Adaptive Inference project},
  year={2025}
}
```
