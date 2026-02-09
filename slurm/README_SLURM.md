# RVQ Experiment on Great Lakes HPC - SLURM Guide

This guide explains how to run the RVQ (Residual Vector Quantization) experiment on the University of Michigan's Great Lakes HPC cluster using SLURM job scheduling.

## Quick Start

```bash
# 1. Setup environment (one-time)
cd /home/gecm/RVQExperiment
bash slurm/environment/setup_env.sh

# 2. Activate environment
source activate rvq_training
source slurm/environment/paths.env

# 3. Run full pipeline (or individual jobs)
sbatch slurm/jobs/full_pipeline.sbatch

# 4. Monitor jobs
squeue -u gecm
tail -f logs/slurm_*.out
```

## Environment Setup

### One-Time Setup

```bash
# Clone the repository to Great Lakes
cd ~
git clone <your-repo-url> RVQExperiment
cd RVQExperiment

# Run setup script (creates conda environment, installs LIBERO, etc.)
bash slurm/environment/setup_env.sh
```

This will:
- Create conda environment `rvq_training` with all dependencies
- Clone and install LIBERO
- Setup configuration files
- Verify PyTorch CUDA and LIBERO installations

### Hugging Face Token

The setup expects your Hugging Face token to be stored in `~/.hf_token`:

```bash
echo "hf_YourTokenHere" > ~/.hf_token
chmod 600 ~/.hf_token
```

### Environment Variables

Before running scripts, always source the environment:

```bash
source activate rvq_training
source slurm/environment/paths.env
```

This sets up:
- `RVQ_ROOT`: Project root directory
- `RVQ_DATA_DIR`: Data directory
- `RVQ_MODEL_DIR`: Model checkpoints directory
- `RVQ_RESULTS_DIR`: Evaluation results directory
- `RVQ_LOG_DIR`: Training logs directory
- `HF_TOKEN`: Hugging Face token (from ~/.hf_token)
- `MUJOCO_GL`: Rendering backend (egl for headless)

## Job Submission

### Individual Jobs

Run phases sequentially:

```bash
# Phase 0: Collect LIBERO action data (2 hours)
sbatch slurm/jobs/0_collect_libero_data.sbatch

# Phase 1: Train RFSQ AutoEncoder (4 hours)
sbatch slurm/jobs/1_phase1_rfsq.sbatch

# Phase 2a: Collect OpenVLA features + RFSQ labels (6 hours)
sbatch slurm/jobs/2_phase2_data_collection.sbatch

# Phase 2b: Train Draft + Conditional models (8 hours)
sbatch slurm/jobs/3_phase2_training.sbatch

# Phase 3: Evaluate RSD model (6 hours)
sbatch slurm/jobs/4_phase3_rsd_evaluation.sbatch

# Baseline: Evaluate OpenVLA-OFT only (4 hours)
sbatch slurm/jobs/baseline_openvla.sbatch
```

### Full Pipeline

Run all phases in one job (24 hours):

```bash
sbatch slurm/jobs/full_pipeline.sbatch
```

This automatically runs all phases sequentially with error checking. If any phase fails, the job stops and sends an email notification.

## Monitoring Jobs

### Check Job Status

```bash
# List your jobs
squeue -u gecm

# Check job details
scontrol show job <job_id>

# View job history
sacct -u gecm --starttime $(date -d '7 days ago' +%Y-%m-%d)
```

### View Logs

```bash
# Real-time monitoring
tail -f logs/slurm_<job_name>_<job_id>.out

# View errors
cat logs/slurm_<job_name>_<job_id>.err

# List all logs
ls -lht logs/
```

### Cancel Jobs

```bash
# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u gecm
```

## Output Structure

After running the pipeline, you'll have:

```
RVQExperiment/
├── data/
│   ├── libero_actions_normalized.pt       # Phase 0 output
│   └── phase2_training_data.pt            # Phase 2a output
├── models/
│   ├── rfsq_robust_best.pt                # Phase 1 output
│   ├── best_draft_with_projection.pt      # Phase 2b output (Draft)
│   └── openvla_rfsq_conditional/
│       └── best_rfsq_head.pt              # Phase 2b output (Conditional)
├── results/
│   ├── rsd_evaluation_results.json        # Phase 3 output
│   ├── baseline_results.json              # Baseline output
│   └── baseline_failures/                 # Failure videos (if enabled)
│       ├── failure_episode_0.mp4
│       └── ...
└── logs/
    ├── phase1_robust_rfsq_training_*/     # Phase 1 logs
    │   ├── metrics.jsonl
    │   ├── summary.json
    │   └── log.txt
    ├── phase2_draft_model_training_*/     # Phase 2 Draft logs
    ├── phase2_conditional_rfsq_head_*/    # Phase 2 Conditional logs
    └── slurm_*.out                        # SLURM job logs
```

## Resource Requirements

### Job Configuration

All jobs use the following Great Lakes resources:
- **Account**: `eecs545w26_class`
- **Partition**: `spgpu` (A40 48GB GPU)
- **Nodes**: 1
- **GPUs**: 1 (A40)
- **CPUs**: 4
- **Memory**: 48GB
- **Email**: `gecm@umich.edu` (BEGIN, END, FAIL notifications)

### Time Limits

| Job | Time Limit | Typical Runtime |
|-----|-----------|----------------|
| Data Collection | 2 hours | 1-1.5 hours |
| Phase 1 Training | 4 hours | 2-3 hours |
| Phase 2 Data | 6 hours | 3-5 hours |
| Phase 2 Training | 8 hours | 5-7 hours |
| Phase 3 Evaluation | 6 hours | 3-5 hours |
| Baseline Evaluation | 4 hours | 2-3 hours |
| **Full Pipeline** | **24 hours** | **18-22 hours** |

## Troubleshooting

### Common Issues

#### 1. Module Not Found

```
ModuleNotFoundError: No module named 'libero'
```

**Solution**: Activate conda environment
```bash
source activate rvq_training
```

#### 2. CUDA Not Available

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce batch size: Edit `.sbatch` file and add `--batch_size 16`
- Request more memory: Change `#SBATCH --mem=48GB` to `#SBATCH --mem=64GB`
- Check GPU usage: `nvidia-smi` in your job

#### 3. Data File Not Found

```
ERROR: Data file not found: /path/to/data.pt
```

**Solution**: Run previous phases first
```bash
# Check what exists
ls -l ${RVQ_DATA_DIR}/
ls -l ${RVQ_MODEL_DIR}/

# Run missing phases
sbatch slurm/jobs/0_collect_libero_data.sbatch
```

#### 4. Environment Not Setup

```
ERROR: RVQ_ROOT not set
```

**Solution**: Source environment variables
```bash
source slurm/environment/paths.env
```

#### 5. LIBERO Rendering Issues

```
ERROR: Could not initialize EGL
```

**Solution**: Already handled in `paths.env`:
```bash
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
```

If still failing, try:
```bash
export MUJOCO_EGL_DEVICE_ID="0"
```

#### 6. HuggingFace Authentication

```
ERROR: huggingface_hub.errors.HfPermissionError
```

**Solution**: Check token file
```bash
cat ~/.hf_token  # Should show your token
# If not, create it:
echo "hf_YourTokenHere" > ~/.hf_token
chmod 600 ~/.hf_token
```

### Debugging Tips

1. **Test with small dataset**: Add `--debug` flag to training scripts
   ```bash
   python train_phase1_rfsq.py --debug  # Runs only 10 epochs
   ```

2. **Interactive debugging**: Request interactive GPU session
   ```bash
   salloc --account=eecs545w26_class --partition=spgpu --gres=gpu:1 --mem=48GB --time=2:00:00
   source activate rvq_training
   python slurm/scripts/train_phase1_rfsq.py --debug
   ```

3. **Check logs**: Always check both `.out` and `.err` files
   ```bash
   tail -n 100 logs/slurm_rvq_phase1_training_*.err
   ```

4. **Verify GPU**: Add to job script before running Python
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Performance Tuning

### Faster Training

Reduce epochs for quick testing:
```bash
python train_phase1_rfsq.py --epochs 10 --batch_size 128
```

### Storage Management

Clean up old checkpoints:
```bash
# Remove all but best checkpoints
find ${RVQ_MODEL_DIR} -name "checkpoint_*.pt" -mtime +7 -delete

# Check disk usage
du -sh ${RVQ_ROOT}/*
```

### Parallel Jobs

Run baseline evaluation in parallel with Phase 3:
```bash
sbatch slurm/jobs/4_phase3_rsd_evaluation.sbatch
sbatch slurm/jobs/baseline_openvla.sbatch  # Runs in parallel
```

## Results Analysis

### View Training Logs

```bash
# Phase 1 final metrics
cat logs/phase1_robust_rfsq_training_*/summary.json | jq .

# Phase 2 accuracy per layer
cat logs/phase2_conditional_rfsq_head_*/summary.json | jq '.final_metrics'
```

### Compare Results

```bash
# RSD vs Baseline
jq '.success_rate' results/rsd_evaluation_results.json
jq '.success_rate' results/baseline_results.json
```

### View Failure Videos

```bash
# List all failure videos
ls -lh results/baseline_failures/*.mp4

# Play video (if Great Lakes has a viewer)
ffplay results/baseline_failures/failure_episode_0.mp4
```

## Advanced Usage

### Custom Arguments

All scripts support extensive CLI arguments. View help:

```bash
python slurm/scripts/train_phase1_rfsq.py --help
python slurm/scripts/train_phase2_conditional.py --help
python slurm/scripts/evaluate_phase3_rsd.py --help
```

### Train Only Specific Model

Phase 2 can train Draft or Conditional model separately:

```bash
# Train only Draft model
python train_phase2_conditional.py \
    --data_path $RVQ_DATA_DIR/phase2_training_data.pt \
    --train_draft_only \
    --epochs 50

# Train only Conditional model
python train_phase2_conditional.py \
    --data_path $RVQ_DATA_DIR/phase2_training_data.pt \
    --train_conditional_only \
    --epochs 50
```

### Modify SLURM Resources

Edit `.sbatch` files to request different resources:

```bash
# Example: Request 2 GPUs and more memory
#SBATCH --gres=gpu:2
#SBATCH --mem=96GB
```

## Getting Help

### Documentation

- Main README: `../CLAUDE.md`
- Phase 1 Guide: `../RVQ_PHASE1_GUIDE.md`
- Phase 2/3 Guide: `../PHASE2_3_GUIDE.md`

### Support

- Great Lakes Help: `hpc-support@umich.edu`
- SLURM Documentation: `https://arc.umich.edu/greatlakes/`

### Useful Commands

```bash
# Check account balance
my_accounts

# View partition info
sinfo -p spgpu

# See who's using GPUs
squeue -p spgpu

# Check your storage quota
quota
```

## Expected Results

After successful completion, you should see:

**Phase 1**: MSE < 0.01 for L0-L2, MSE < 0.001 for L0-L7
**Phase 2**: Token prediction accuracy > 70% for all layers
**Phase 3**:
- Success rate ≈ Baseline
- Latency ↓ 30-50%
- Adaptive rate: 20-30%

If results don't match expectations, check:
1. Training converged (check logs)
2. Models loaded correctly (no shape mismatches)
3. Sufficient training data (100+ episodes per phase)

## Next Steps

After getting results:
1. Analyze failure videos to understand error cases
2. Compare RSD speedup vs baseline
3. Tune adaptive thresholds for better performance
4. Experiment with different RFSQ layer counts
5. Test on other LIBERO task suites (libero_object, libero_goal)

Happy experimenting! 🚀
