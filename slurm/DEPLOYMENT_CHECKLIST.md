# Great Lakes Deployment Checklist

Use this checklist when deploying to Great Lakes HPC.

## Pre-Deployment (Local)

- [ ] All code committed to git
- [ ] .gitignore includes data/, models/, logs/, results/
- [ ] No sensitive credentials in code
- [ ] README_SLURM.md reviewed

## Initial Setup (Great Lakes)

### 1. Upload Code
```bash
# From local machine (Git Bash on Windows)
cd /f/umich/26wn/researchInterview/experimentalCode/RVQExperiment
git bundle create rvq.bundle --all
scp rvq.bundle gecm@greatlakes.arc-ts.umich.edu:~/

# On Great Lakes
cd ~
git clone rvq.bundle RVQExperiment
cd RVQExperiment
```

- [ ] Code uploaded to Great Lakes
- [ ] Repository structure verified

### 2. Verify Modules
```bash
module avail conda  # Should show conda/miniconda
module avail cuda   # Should show cuda/12.1 or similar
```

- [ ] Conda module available
- [ ] CUDA 12.1+ module available

### 3. Setup Hugging Face Token
```bash
echo "hf_YourActualTokenHere" > ~/.hf_token
chmod 600 ~/.hf_token
cat ~/.hf_token  # Verify it's correct
```

- [ ] HF token created in ~/.hf_token
- [ ] File permissions set to 600

### 4. Run Environment Setup
```bash
cd ~/RVQExperiment
bash slurm/environment/setup_env.sh 2>&1 | tee setup.log
```

**This will take ~30-60 minutes**. Watch for errors.

- [ ] Conda environment created
- [ ] PyTorch + CUDA verified
- [ ] LIBERO cloned and installed
- [ ] torch.load fix applied
- [ ] No errors in setup.log

### 5. Verify Environment
```bash
source activate rvq_training
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from libero.libero import benchmark; print('LIBERO OK')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

Expected output:
- PyTorch 2.2.0 or later
- CUDA: True
- LIBERO OK
- Transformers 4.40.1 or later

- [ ] PyTorch CUDA available
- [ ] LIBERO imports successfully
- [ ] Transformers installed

### 6. Verify Directory Structure
```bash
cd ~/RVQExperiment
ls -la slurm/
ls -la slurm/scripts/
ls -la slurm/jobs/
ls -la slurm/utils/
```

- [ ] All directories present
- [ ] All .sbatch files present
- [ ] All .py scripts present

## Test Run (Optional but Recommended)

### Test Phase 0 with Minimal Data
```bash
# Edit job script to use 1 episode
cd ~/RVQExperiment
cp slurm/jobs/0_collect_libero_data.sbatch slurm/jobs/test_collect.sbatch

# Edit test_collect.sbatch:
# Change: --num-episodes 100 → --num-episodes 1
# Change: --time=02:00:00 → --time=00:30:00

sbatch slurm/jobs/test_collect.sbatch
```

Wait for completion (~5-10 min):
```bash
squeue -u gecm
tail -f logs/slurm_*.out
```

- [ ] Test job submitted successfully
- [ ] Job runs without errors
- [ ] data/libero_actions_normalized.pt created
- [ ] Log file shows completion

## Full Pipeline Deployment

### Option A: Full Pipeline (Recommended)
```bash
cd ~/RVQExperiment
sbatch slurm/jobs/full_pipeline.sbatch
```

- [ ] Full pipeline job submitted
- [ ] Job ID recorded: __________

### Option B: Individual Phases
```bash
# Submit phases one at a time, waiting for each to complete
sbatch slurm/jobs/0_collect_libero_data.sbatch  # Job ID: _____
# Wait for completion, then:
sbatch slurm/jobs/1_phase1_rfsq.sbatch          # Job ID: _____
# Continue...
```

- [ ] Phase 0 submitted, Job ID: __________
- [ ] Phase 1 submitted, Job ID: __________
- [ ] Phase 2a submitted, Job ID: __________
- [ ] Phase 2b submitted, Job ID: __________
- [ ] Phase 3 submitted, Job ID: __________
- [ ] Baseline submitted, Job ID: __________

## Monitoring

### Check Job Status
```bash
# Current queue
squeue -u gecm

# Recent jobs
sacct -u gecm --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Detailed job info
scontrol show job <JOB_ID>
```

### View Logs
```bash
# Live monitoring
tail -f logs/slurm_*.out

# Check for errors
grep -i "error\|fail\|exception" logs/slurm_*.err

# View specific job log
cat logs/slurm_rvq_full_pipeline_<JOB_ID>.out
```

### Check Disk Usage
```bash
du -sh ~/RVQExperiment
du -sh ~/RVQExperiment/data
du -sh ~/RVQExperiment/models
du -sh ~/RVQExperiment/results
```

## Troubleshooting

### Job Failed to Start
- [ ] Check SLURM allocation: `squeue -u gecm`
- [ ] Check account balance: `my_accounts`
- [ ] Check partition access: `sinfo -p spgpu`

### Job Crashed
```bash
# Check error log
cat logs/slurm_*_<JOB_ID>.err

# Check exit code
sacct -j <JOB_ID> --format=JobID,State,ExitCode
```

Common issues:
- **ExitCode 1**: Python error, check .err file
- **ExitCode 137**: OOM (out of memory)
- **ExitCode 143**: Job timeout

### Out of Memory (OOM)
Edit job script, reduce batch_size:
```bash
# In train_phase1_rfsq.py call
--batch_size 32  # Try 32, 16, or 8
```

### GPU Not Available
```bash
# Check CUDA in job
nvidia-smi  # Add this to sbatch script
echo $CUDA_VISIBLE_DEVICES
```

### Module Not Found
```bash
# Activate environment
source activate rvq_training
source slurm/environment/paths.env

# Verify imports
python -c "from utils.experiment_logger import ExperimentLogger"
python -c "from models.rfsq_models import ActionRFSQAE"
```

## Post-Completion

### Verify Results
```bash
cd ~/RVQExperiment

# Check models
ls -lh models/
# Expected: rfsq_robust_best.pt, best_draft_with_projection.pt

# Check results
cat results/rsd_evaluation_results.json
cat results/baseline_results.json

# Check videos
ls -lh results/baseline_failures/
```

- [ ] All models saved
- [ ] Results JSON files created
- [ ] Failure videos recorded (if any)
- [ ] Logs complete

### Download Results (to local machine)
```bash
# From local machine
scp -r gecm@greatlakes.arc-ts.umich.edu:~/RVQExperiment/results ./
scp -r gecm@greatlakes.arc-ts.umich.edu:~/RVQExperiment/models ./
scp -r gecm@greatlakes.arc-ts.umich.edu:~/RVQExperiment/logs ./
```

- [ ] Results downloaded to local machine
- [ ] Models downloaded
- [ ] Logs downloaded

### Cleanup (Optional)
```bash
# Remove large training data (keep models and results)
rm -rf ~/RVQExperiment/data/phase2_training_data.pt
rm -rf ~/RVQExperiment/.hf_cache/

# Or archive
tar -czf rvq_training_data.tar.gz data/
mv rvq_training_data.tar.gz ~/archive/
```

- [ ] Large files cleaned up or archived

## Success Criteria

All items should be ✅:
- [ ] Full pipeline completed without errors
- [ ] Phase 1 MSE < 0.01 for L0-L2
- [ ] Phase 2 token accuracy > 70%
- [ ] Phase 3 RSD evaluation completed
- [ ] Baseline evaluation completed
- [ ] Success rates computed
- [ ] Failure videos recorded
- [ ] All results downloaded

## Notes

**Expected Runtime**: ~24-30 hours for full pipeline

**Expected Storage**: ~5-10GB total

**Expected Cost**: ~24-30 GPU hours on Great Lakes

**Contact for Issues**:
- Great Lakes support: hpc-support@umich.edu
- Project questions: Check CLAUDE.md and README_SLURM.md

---

**Date Started**: __________

**Date Completed**: __________

**Total GPU Hours**: __________

**Notes**:
