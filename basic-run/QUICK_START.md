# Quick Start Guide for PI0.5-LIBERO Benchmark

## What Changed?

Your script has been updated to follow the **official OpenPI LIBERO example** (`openpi/examples/libero/main.py`). Key changes:

### 1. **Proper Policy Loading**
```python
# OLD (incorrect):
policy = Policy.from_config(config=config, checkpoint_dir=checkpoint_dir, seed=42)

# NEW (correct):
policy = _policy_config.create_trained_policy(
    train_config=config,
    checkpoint_dir=checkpoint_dir,
    default_prompt=default_prompt,
    pytorch_device=pytorch_device,
)
```

### 2. **Correct Observation Preprocessing**
- Images are rotated 180 degrees: `img[::-1, ::-1]`
- Images resized with padding: `image_tools.resize_with_pad(img, 224, 224)`
- State includes: `eef_pos + quat2axisangle(eef_quat) + gripper_qpos`
- Observations wait 10 steps for objects to stabilize

### 3. **Action Replanning**
- Actions are replanned every 5 steps (configurable)
- Uses action chunking with deque for smooth execution

### 4. **Environment Setup**
- Proper BDDL file path resolution
- Environment seeding for reproducibility
- Correct camera resolution (256x256)

## Installation

```bash
# 1. Install OpenPI
cd openpi
pip install -e .

# 2. Install LIBERO and dependencies
pip install libero
pip install 'robosuite==1.4.0'  # or 1.3.0 if compatibility issues

# 3. Install additional requirements
pip install -r requirements_libero.txt
```

## Usage

### Check Environment
```bash
python run_pi05_libero_benchmark.py --check-env
```

Expected output:
```
================================================================================
ENVIRONMENT CHECK
================================================================================
Python: 3.10.x
openpi available: ✓ YES
  pi05_libero config found: ✓
LIBERO available: ✓ YES
robosuite available: ✓ YES
  robosuite version: 1.4.0
...
================================================================================
```

### Run Benchmark

```bash
# Quick test on libero_spatial (10 tasks, 10 episodes each)
python run_pi05_libero_benchmark.py --task_suite libero_spatial --num_episodes 10

# Full benchmark on libero_10 (50 episodes per task)
python run_pi05_libero_benchmark.py --task_suite libero_10 --num_episodes 50

# Use local checkpoint
python run_pi05_libero_benchmark.py \
    --checkpoint_dir ./checkpoints/pi05_libero \
    --task_suite libero_spatial \
    --num_episodes 10

# Quiet mode (less verbose output)
python run_pi05_libero_benchmark.py --task_suite libero_object --quiet
```

## Expected Results

Based on the OpenPI paper, you should see:

| Task Suite | Expected Success Rate |
|------------|----------------------|
| libero_spatial | ~98.8% |
| libero_object | ~98.2% |
| libero_goal | ~98.0% |
| libero_10 | ~92.4% |
| **Average** | **~96.85%** |

Example output:
```
================================================================================
BENCHMARK SUMMARY
================================================================================

Task Suite: libero_spatial
Overall Success Rate: 98.5%
Total Successes: 98/100

Per-Task Results:
ID   Task Name                                      Success Rate    Successes
--------------------------------------------------------------------------------
0    LIBERO_SPATIAL_0                               100.0%           10/10
1    LIBERO_SPATIAL_1                                90.0%            9/10
2    LIBERO_SPATIAL_2                               100.0%           10/10
...
================================================================================
```

## Troubleshooting

### Issue: ImportError for openpi_client

**Error:**
```
ModuleNotFoundError: No module named 'openpi_client'
```

**Solution:**
```bash
cd openpi
pip install -e packages/openpi-client/
```

### Issue: LIBERO import fails

**Error:**
```
ImportError: cannot import name 'SingleArmEnv'
```

**Solution:** This is a robosuite version compatibility issue.
```bash
# Try robosuite 1.3.0 instead
pip uninstall robosuite
pip install 'robosuite==1.3.0'
```

### Issue: Images not rotating correctly

**Symptom:** Low success rate (< 50%)

**Cause:** The OpenPI training data preprocesses images by rotating them 180 degrees. If this isn't done during inference, the model won't recognize the scenes properly.

**Verification:** Check that this line exists in evaluate_task():
```python
img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
```

### Issue: CUDA out of memory

**Solutions:**
1. Use CPU: `pytorch_device="cpu"` (slower but works)
2. Close other GPU programs
3. Reduce batch size (already minimal at 1)

### Issue: Checkpoint download fails

**Error:**
```
Error downloading from gs://openpi-assets/...
```

**Solutions:**
1. Check internet connection
2. Try downloading manually:
   ```bash
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero ./checkpoints/
   ```
3. Use local checkpoint:
   ```bash
   python run_pi05_libero_benchmark.py --checkpoint_dir ./checkpoints/pi05_libero
   ```

## Key Differences from Original

| Aspect | Original Script | Updated Script |
|--------|----------------|----------------|
| Policy Loading | Custom model loading | `policy_config.create_trained_policy()` |
| Image Preprocessing | Simple resize | Rotate 180° + resize with padding |
| State Observation | Only eef_pos | eef_pos + axis_angle + gripper |
| Action Execution | Immediate | Action chunking with replanning |
| Object Settling | None | 10-step wait period |
| Imports | `Policy`, `libero_policy` | `policy_config`, `image_tools` |

## Architecture Flow

```
1. Download checkpoint from GCS
   ↓
2. Load pi05_libero config
   ↓
3. Create policy with transforms and normalization
   ↓
4. Load LIBERO task suite
   ↓
5. For each task:
   - Create environment
   - For each episode:
     * Reset environment
     * Wait for objects to settle (10 steps)
     * Get observation → preprocess images → get state
     * Query policy for action chunk
     * Execute actions with replanning every 5 steps
     * Record success/failure
   ↓
6. Report results
```

## Performance Tips

1. **Use GPU:** Significantly faster than CPU
   ```python
   policy = load_pi05_libero_policy(pytorch_device="cuda")
   ```

2. **Reduce episodes for testing:** Start with 5-10 episodes per task
   ```bash
   python run_pi05_libero_benchmark.py --num_episodes 5
   ```

3. **Test on smaller suites first:** libero_spatial (10 tasks) before libero_90 (90 tasks)

4. **Cache checkpoint:** First run downloads, subsequent runs use cache (~/.cache/openpi/)

## References

- **OpenPI Repository:** https://github.com/physical-intelligence/openpi
- **OpenPI LIBERO Example:** `openpi/examples/libero/main.py`
- **LIBERO Benchmark:** https://github.com/Lifelong-Robot-Learning/LIBERO
- **Policy Config:** `openpi/src/openpi/policies/policy_config.py`
- **Serve Policy Script:** `openpi/scripts/serve_policy.py`

## Next Steps

1. **Run environment check:**
   ```bash
   python run_pi05_libero_benchmark.py --check-env
   ```

2. **Quick test (5 minutes):**
   ```bash
   python run_pi05_libero_benchmark.py --task_suite libero_spatial --num_episodes 5
   ```

3. **Full benchmark (1-2 hours):**
   ```bash
   python run_pi05_libero_benchmark.py --task_suite libero_spatial --num_episodes 50
   ```

Good luck with your benchmarking! 🚀
