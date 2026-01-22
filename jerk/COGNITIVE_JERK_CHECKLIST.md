# Cognitive Jerk Analysis - Quick Validation Checklist

Use this checklist to verify the analysis is working correctly.

## Pre-Run Checks

- [ ] Modal CLI installed and authenticated (`modal token set`)
- [ ] GPU quota available (requires A100)
- [ ] Environment variables set:
  - `ORCHESTRA_SDK_PATH` (points to `/root/vm_worker/src`)
  - `AGENT_ID`, `PROJECT_ID`, `USER_ID` (optional but recommended)

## Expected Console Output

### 1. Model Loading Phase
```
📦 Loading OpenVLA-OFT model...
Using device: cuda
✅ Model loaded successfully
```

### 2. Environment Initialization
```
🎮 Initializing LIBERO environment...
Task: put the red mug on the plate
BDDL file: /root/LIBERO/libero/libero/bddl_files/...
```

### 3. First Step Diagnostics (Step 10)
```
🔄 Stabilization complete, resetting tracking variables
🔍 Image shapes after resize:
   full_image: (224, 224, 3)
   wrist_image: (224, 224, 3)
   state: (7,)
🔍 Action chunk size: 8
🔍 Refilling action queue
🔍 First cognitive jerk: 0.0XXXXX
🔍 First gripper value: X.XXX (expecting range [-1, 1])
```

### 4. Execution Progress
```
🔍 First action instability: X.XXXXX
🤏 Gripper event at step XX: closing
🤏 Gripper event at step YY: opening
✅ Task completed at step ZZZ
```

### 5. Data Saving
```
💾 Saving raw metrics data...
   ✅ Raw data saved to jerk_metrics_task0.json
📊 Generating visualization...
   📊 Plot saved to jerk_analysis_task0.png
```

## Validation Metrics

### ✅ Success Criteria

| Metric | Expected Range | Check |
|--------|---------------|-------|
| Total steps | 50-230 | [ ] |
| Cognitive jerk mean | 0.01-0.5 | [ ] |
| Cognitive jerk max | 0.05-2.0 | [ ] |
| Action instability mean | 0.1-2.0 | [ ] |
| Gripper state range | [-1.0, 1.0] | [ ] |
| Gripper events | 2-10 total | [ ] |
| Correlation (jerk ↔ instability) | >0.2 | [ ] |

### ⚠️ Warning Signs

| Issue | Symptom | Likely Cause |
|-------|---------|--------------|
| No jerk data | `cognitive_jerk: []` | Hidden state extraction failed |
| Jerk always ~2.0 | All values near max | Hidden states are random/uncorrelated |
| Jerk always ~0.0 | All values near zero | Same hidden state repeated |
| Instability always 0 | All action plans identical | Action prediction stuck |
| No gripper events | Empty list | Gripper values constant (unusual) |
| Negative correlation | <-0.1 | Metrics decoupled (investigate) |

## Output File Checks

### 1. Visualization (`jerk_analysis_task0.png`)

- [ ] **Panel 1**: Blue line with red dots (high jerk moments)
- [ ] **Panel 2**: Purple line with filled area (instability)
- [ ] **Panel 3**: Orange line with colored triangles (gripper events)
- [ ] **Panel 4**: Timeline with purple ticks and colored star (success/failure)
- [ ] All panels share x-axis (timesteps aligned)
- [ ] Gray dashed lines appear every ~8 steps (prediction markers)

### 2. Raw Data (`jerk_metrics_task0.json`)

```bash
# Verify file structure
cat jerk_metrics_task0.json | python -m json.tool | head -20
```

Expected keys:
- [ ] `timesteps`: list of integers
- [ ] `cognitive_jerk`: list of floats in [0, 2]
- [ ] `action_change`: list of floats (positive)
- [ ] `gripper_state`: list of floats in [-1, 1]
- [ ] `gripper_events`: list of strings/null
- [ ] `prediction_steps`: list of integers
- [ ] `task_description`: string
- [ ] `success`: boolean

### 3. Data Consistency Checks

```python
import json
with open('jerk_metrics_task0.json') as f:
    data = json.load(f)

# Check alignment
len(data['timesteps']) == len(data['cognitive_jerk'])  # Should be True
len(data['action_change']) == len(data['cognitive_jerk']) - 1  # Should be True (offset)
len(data['gripper_state']) == len(data['timesteps'])  # Should be True
len(data['prediction_steps']) * 8 ≈ len(data['timesteps'])  # Approximately
```

## Common Issues & Fixes

### Issue: "Hidden state extraction failed"
```
✅ Fix: Check GPU availability and model loading
modal run --help  # Verify modal works
nvidia-smi  # Check GPU in container
```

### Issue: "Action prediction failed"
```
✅ Fix: Verify OpenVLA processor working
# Check processor import
from experiments.robot.openvla_utils import get_processor
```

### Issue: "Correlation analysis not shown"
```
✅ Fix: Need at least 2 data points
# Check: len(cognitive_jerk) > 1 and len(action_change) > 0
```

### Issue: Plot shows empty panels
```
✅ Fix: Check metric lists are populated
# If cognitive_jerk is empty, hidden state extraction failed
# If action_change is empty, action prediction failed
```

## Performance Benchmarks

| Phase | Expected Time | Check |
|-------|--------------|-------|
| Model loading | 30-60 sec | [ ] |
| Environment init | 5-10 sec | [ ] |
| Per-step inference | ~500ms | [ ] |
| Total runtime | 2-5 min | [ ] |

## Debugging Commands

### Check Modal status
```bash
modal app list
modal volume list
```

### Check outputs
```bash
# List results
modal volume get rsd-results jerk_analysis_task0.png ./
modal volume get rsd-results jerk_metrics_task0.json ./

# Or check /tmp directory
ls /tmp/jerk_*
```

### View logs
```bash
# Modal automatically shows logs, but you can also:
modal app logs rsd-phase3-libero-eval-conditional-fixed
```

## Success Confirmation

✅ **Analysis is successful if:**
1. Console shows all diagnostic prints
2. Both output files created
3. Metrics within expected ranges
4. Visualization shows clear time-series patterns
5. Correlation analysis printed (if enough data)
6. No errors in console output

## Next Steps After Validation

Once validation passes:
1. Run on multiple tasks (modify script to loop over tasks)
2. Compare success vs failure patterns
3. Analyze correlation strength across tasks
4. Use jerk thresholds for adaptive inference
5. Integrate with Phase 3 speculative decoding

---

**Quick Test**: If you want to verify the script syntax without running:
```bash
python analyze_cognitive_jerk.py --help  # Won't work (modal only)
modal run analyze_cognitive_jerk.py --dry-run  # Check if modal recognizes it
```
