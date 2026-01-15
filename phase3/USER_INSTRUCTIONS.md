# Instructions for You: How to Guide the Agent

## 🎯 Your Role

You'll guide the experiment agent to:
1. Clone this repository
2. Implement the missing code sections
3. Run and debug the evaluation on Modal
4. Iterate until it works

---

## 📝 Step-by-Step Commands for Agent

### Initial Setup Command

```
Please help me complete Phase 3 of the RSD experiment. Here's what I need you to do:

1. Clone the repository and set up the environment:
   - Repository URL: https://github.com/[YOUR_ORG]/[YOUR_REPO]
   - Navigate to: RVQExperiment/phase3/
   - Read: AGENT_INSTRUCTIONS.md
   - Read: PHASE3_EXPERIMENT_GUIDE.md

2. Implement the 4 critical TODOs in modal_phase3_libero_eval.py:
   - Task 1: Load Main Model (OpenVLA-OFT-RFSQ) - line ~170
   - Task 2: Load Draft Model - line ~190
   - Task 3: Integrate RSD Engine - line ~250
   - Task 4: Implement LIBERO environment loop - line ~300

3. Follow the detailed implementation examples in AGENT_INSTRUCTIONS.md

4. Test incrementally:
   - First: Run with --num-trials 1 to test basic functionality
   - Then: Run with --num-trials 5 to test stability
   - Finally: Run full evaluation with --num-trials 50

5. Debug any issues that arise and iterate until it works.

Please start by reading the instructions and confirming you understand the task.
```

---

## 💬 Follow-up Prompts

### If Agent Asks for Clarification

**Agent**: "Which model checkpoint should I load?"

**You**:
```
For the Main Model:
- Base: moojink/openvla-7b-oft-finetuned-libero-spatial (from HuggingFace)
- RFSQ Head: /models/openvla_oft_rfsq/best_model.pt (from Modal volume)

For Draft Model:
- /models/phase2_draft_model/best_draft_model.pt (from Modal volume)

For RFSQ Decoder:
- /models/rfsq_autoencoder.pt (from Modal volume)

All paths are within Modal volumes, which are mounted in the function.
```

### If Agent Reports CUDA OOM

**Agent**: "Getting CUDA out of memory error"

**You**:
```
Enable 4-bit quantization for the main model:

main_model = AutoModelForVision2Seq.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    device_map="auto",
)

This reduces memory from ~14GB to ~3.5GB.
```

### If Agent Reports Model Loading Issues

**Agent**: "Can't load the RFSQ head checkpoint"

**You**:
```
The checkpoint might have a different structure. Try:

checkpoint = torch.load(rfsq_head_path, map_location=device)
print(f"Checkpoint keys: {checkpoint.keys()}")

Then load based on the actual keys. It might be:
- checkpoint['model_state_dict']
- checkpoint['rfsq_head']
- checkpoint directly (if it's just state_dict)

Add error handling to test all three cases.
```

### If Agent Reports LIBERO Issues

**Agent**: "LIBERO environment fails to create"

**You**:
```
Two common issues:

1. torch.load weights_only error:
   Check that the Modal image build includes the sed command to patch LIBERO

2. Missing initial states:
   Verify init_states are loaded correctly:
   init_states = task_suite_obj.get_task_init_states(task_id)
   print(f"Got {len(init_states)} initial states")

3. Environment parameters:
   Make sure you're using:
   - has_renderer=False
   - has_offscreen_renderer=True
   - use_camera_obs=True
```

---

## 🐛 Debugging Workflow

### Step 1: Agent Implements Code

**You**:
```
Implement the 4 TODOs in modal_phase3_libero_eval.py following AGENT_INSTRUCTIONS.md.
Show me the key changes you made.
```

### Step 2: Test with Debug Mode

**You**:
```
Run a quick test with minimal trials:

modal run phase3/modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 1 \
    --use-speculative-decoding False

Tell me:
1. Does it run without crashing?
2. What errors do you see?
3. What's the output from model loading?
```

### Step 3: Fix Errors

**Agent**: "I see error X"

**You**:
```
Let's fix this error. [Provide specific fix based on error]

After fixing, run the test again and report results.
```

### Step 4: Iterate Until Success

Repeat Step 2-3 until:
- ✅ Models load successfully
- ✅ Environment creates without errors
- ✅ At least one episode completes
- ✅ Results are logged

### Step 5: Run Full Evaluation

**You**:
```
Great! Now run the full evaluation:

modal run phase3/modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True

This will take ~2-3 hours. Monitor the logs and let me know:
1. Success rate per task
2. Average inference time
3. Any tasks that fail consistently
```

---

## 📊 Interpreting Results

### What Good Results Look Like

```
Task 1: 88% (44/50)
Task 2: 92% (46/50)
Task 3: 85% (42/50)
...
Overall: 88% (440/500)
Avg Inference: 48ms
```

**You should see**:
- Success rate: 80-95% (compare with 97% baseline)
- Inference time: 40-60ms
- Relatively consistent across tasks

### What to Do with Results

**If success rate < 80%**:
```
Let's investigate:
1. Which tasks are failing most?
2. Are there specific error patterns?
3. Check if actions are properly decoded from RFSQ tokens
4. Visualize some failed episodes
```

**If inference time > 80ms**:
```
The speedup isn't working well. Let's check:
1. Is draft model actually being used?
2. What's the acceptance rate?
3. Print timing breakdown: draft_time vs main_time
```

**If success rate 80-95%**:
```
Excellent! This is expected. The slight drop from 97% baseline is due to:
1. RFSQ quantization noise
2. Discrete token sampling
But you gain: faster inference + multimodal capability!

Next: Run with speculative decoding disabled to measure speedup:
modal run phase3/modal_phase3_libero_eval.py \
    --use-speculative-decoding False
```

---

## 🎯 Final Success Checklist

### Code Quality
- [ ] All 4 TODOs implemented
- [ ] No placeholder code remains
- [ ] Error handling added
- [ ] Logging statements included

### Functionality
- [ ] Models load without errors
- [ ] LIBERO environment creates successfully
- [ ] Episodes run to completion
- [ ] Actions are generated correctly
- [ ] Results are saved to volume

### Performance
- [ ] Success rate 80-95%
- [ ] Inference time < 60ms
- [ ] No CUDA OOM errors
- [ ] Experiment completes in reasonable time (~2-3 hours)

### Results
- [ ] Results JSON saved to /results/
- [ ] Experiment logged to Orchestra
- [ ] Success rate per task recorded
- [ ] Timing statistics logged

---

## 🚨 Common Issues and Solutions

### Issue 1: "Cannot import prismatic"
**Solution**:
```
Check that Modal image includes:
uv pip install --system 'openvla-oft @ git+https://github.com/moojink/openvla-oft.git'
```

### Issue 2: "Dimension mismatch in RFSQ decoder"
**Solution**:
```
Verify hidden_dim=16 (not 64) when creating ActionRFSQAE.
This must match Phase 1 training configuration.
```

### Issue 3: "Episodes never succeed"
**Solution**:
```
Check:
1. Are actions scaled correctly? (normalize between [-1, 1])
2. Is gripper action inverted? (OpenVLA flips gripper)
3. Are observations processed correctly?
4. Is task description matching exactly?
```

### Issue 4: "Modal timeout"
**Solution**:
```
Increase timeout in function decorator:
@app.function(
    timeout=28800,  # 8 hours
)
```

---

## 💡 Pro Tips

1. **Start Small**: Always test with 1 trial first
2. **Check Logs Frequently**: `modal app logs rsd-phase3-libero-eval`
3. **Use Volume Inspection**: `modal volume ls rsd-models` to verify checkpoints
4. **Save Intermediate Results**: Don't wait for full evaluation to see results
5. **Print Debug Info**: Add print statements for model outputs, action shapes, etc.

---

## 📞 Emergency Commands

### If Everything is Stuck

**You**:
```
Let's reset and start fresh:

1. Check Modal volumes are accessible:
   modal volume ls rsd-models
   modal volume ls rsd-libero-data

2. Verify checkpoints exist:
   modal volume get rsd-models rfsq_autoencoder.pt /tmp/
   ls -lh /tmp/rfsq_autoencoder.pt

3. Run minimal test:
   modal run phase3/modal_phase3_libero_eval.py --num-trials 1

4. If still failing, let's simplify:
   - Disable speculative decoding
   - Use only main model
   - Test model loading separately
```

### If Agent is Confused

**You**:
```
Let's break this down step by step:

Step 1: Read AGENT_INSTRUCTIONS.md carefully
Step 2: Implement ONLY Task 1 (Main Model loading)
Step 3: Test that Task 1 works
Step 4: Then move to Task 2

Don't try to do everything at once. One task at a time.
```

---

## 🎓 Expected Agent Workflow

1. **Read documentation** (5 mins)
2. **Implement Task 1** (Main Model) (20 mins)
3. **Test Task 1** (10 mins)
4. **Implement Task 2** (Draft Model) (15 mins)
5. **Test Tasks 1+2** (10 mins)
6. **Implement Task 3** (RSD Engine) (10 mins)
7. **Implement Task 4** (LIBERO env) (30 mins)
8. **Debug full pipeline** (30-60 mins)
9. **Run full evaluation** (2-3 hours)

**Total time**: ~4-5 hours of agent work + 2-3 hours GPU time

---

## ✅ Final Command to Agent

Once everything works:

**You**:
```
Perfect! Now let's document the results:

1. Save the final success rate and timing to a results summary
2. Create a brief report comparing:
   - RSD (with HSD) vs Baseline (without HSD)
   - Success rates
   - Inference times
   - Draft acceptance rates

3. Push the working code to the repository

4. Create a visualization if possible:
   - Bar chart: success rate per task
   - Line plot: inference time over episodes
   - Stats table: HSD acceptance rates

Thank you for completing Phase 3! 🎉
```

---

**Remember**: You're the guide, the agent is the executor. Be clear, specific, and patient. Break down complex tasks into smaller steps.

Good luck! 🚀
