# Phase 3 QUICK TEST Mode

## ⚡ What is QUICK TEST Mode?

The Phase 3 evaluation code has been configured to run in **QUICK TEST mode** to save GPU resources during development and debugging.

### Current Configuration

- **Tasks**: Only Task 0 (first task of libero_spatial)
- **Trials**: Only 1 trial (instead of 50)
- **Runtime**: ~2-5 minutes (instead of hours)
- **Purpose**: Fast verification that code works correctly

---

## 🚀 Running QUICK TEST

```bash
# Test conditional RSD with Mode Locking (QUICK TEST - 1 task, 1 trial)
modal run phase3/modal_phase3_libero_eval_CONDITIONAL_FIXED.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True
```

**Note**: Even though you pass `--num-trials 50`, the code will only run **1 trial** due to QUICK TEST mode.

---

## 🔄 How to Revert to FULL Evaluation

When you're ready to run the complete evaluation on all tasks:

### Option 1: Manual Edit (Recommended)

Edit `modal_phase3_libero_eval_CONDITIONAL_FIXED.py`:

1. **Line 1009** - Change task loop:
   ```python
   # FROM:
   for task_id in range(1):  # Changed from range(num_tasks)

   # TO:
   for task_id in range(num_tasks):
   ```

2. **Line 1031** - Change trial loop:
   ```python
   # FROM:
   for trial_idx in range(1):  # Changed from range(min(num_trials, len(init_states)))

   # TO:
   for trial_idx in range(min(num_trials, len(init_states))):
   ```

3. **Line 999-1002** - Update print messages:
   ```python
   # FROM:
   print(f"\n🎯 Starting QUICK TEST evaluation (Task 0, 1 trial only)...\n")
   print(f"   ⚠️  This is a QUICK TEST to verify code works!")
   print(f"   ⚠️  For full evaluation, modify code to run all {num_tasks} tasks with {num_trials} trials each\n")

   # TO:
   print(f"\n🎯 Starting FULL evaluation ({num_trials} trials per task)...\n")
   ```

### Option 2: Git Revert (If you want to go back to original version)

```bash
cd RVQExperiment
git log --oneline  # Find commit before "QUICK TEST mode"
git revert 63cdd65  # Revert the QUICK TEST commit
git push origin main
```

---

## 📊 Comparison

| Mode | Tasks | Trials/Task | Total Episodes | Est. Runtime | GPU Hours |
|------|-------|-------------|----------------|--------------|-----------|
| **QUICK TEST** | 1 | 1 | 1 | ~2-5 min | ~0.05 |
| **FULL EVAL** | 10 | 50 | 500 | ~10-20 hrs | ~15 |

---

## ⚠️ Important Notes

1. **QUICK TEST is NOT for benchmarking**
   - Success rate from 1 trial is meaningless
   - Only use QUICK TEST to verify code runs without errors

2. **When to use QUICK TEST**:
   - ✅ Testing new code changes
   - ✅ Debugging runtime errors
   - ✅ Validating environment setup
   - ✅ Checking model loading

3. **When to use FULL EVAL**:
   - ✅ Final performance benchmarking
   - ✅ Comparing different models
   - ✅ Publishing results
   - ✅ After code is verified to work

---

## 🎯 Expected Output (QUICK TEST)

```
🚀 Phase 3: CONDITIONAL RSD + Official Fixes - libero_spatial
   ✨ Mode Locking: ENABLED
   🔧 Official Fixes: ENABLED
   Speculative Decoding: ENABLED

🎯 Starting QUICK TEST evaluation (Task 0, 1 trial only)...

   ⚠️  This is a QUICK TEST to verify code works!
   ⚠️  For full evaluation, modify code to run all 10 tasks with 50 trials each

================================================================================
Task 1/10: put the black bowl on top of the wooden cabinet
================================================================================
   🔍 Image shapes after resize:
      full_image: (224, 224, 3)
      wrist_image: (224, 224, 3)
      state: (8,)
   🔍 Action chunk returned 8 actions
      Row 0: [0.123, -0.045, 0.234, 0.012, -0.034, 0.056, -1.000]
      Row 1: [0.145, -0.038, 0.221, 0.009, -0.041, 0.062, -1.000]
      Row 2: [0.167, -0.031, 0.208, 0.006, -0.048, 0.068, -1.000]
   🔍 Processed action: [0.123, -0.045, 0.234, 0.012, -0.034, 0.056, 1.000]
   Trial 1: ✓  (or ✗)

   Task Success Rate: 100.0% (1/1)  (or 0.0% if failed)

================================================================================
🎉 CONDITIONAL+FIXED EVALUATION COMPLETE!
================================================================================
   Success Rate: 100.0% (1/1)  (or 0.0%)
   Draft Acceptance Rate: 68.5%
   Mode Locking Rate: 100.0%
   Fallback Rate: 0.0%
================================================================================
```

---

## 🐛 Troubleshooting

### Issue: Code still runs all tasks

**Solution**: Make sure you edited the correct file and the changes were saved.

```bash
# Check if changes are in the file
grep "range(1)" phase3/modal_phase3_libero_eval_CONDITIONAL_FIXED.py
# Should show 2 matches (line 1009 and 1031)
```

### Issue: Want to test specific task (not Task 0)

**Edit Line 1009**:
```python
# Test Task 3 instead of Task 0:
for task_id in range(3, 4):  # Will run task_id = 3

# Or directly:
task_id = 3  # Skip the loop entirely
```

### Issue: Want to run 3 trials instead of 1

**Edit Line 1031**:
```python
# Run 3 trials:
for trial_idx in range(3):  # Changed from range(1)
```

---

## 📝 Summary

✅ **Current State**: QUICK TEST mode (1 task, 1 trial)
✅ **Purpose**: Fast debugging and verification
✅ **How to change**: Edit lines 1009 and 1031
✅ **Commit**: `63cdd65` - "Phase 3: Enable QUICK TEST mode"

**Remember**: Always use FULL EVAL for final benchmarking!
