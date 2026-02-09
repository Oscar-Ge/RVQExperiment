#!/bin/bash
#SBATCH --job-name=rvq_phase3_eval
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gecm@umich.edu

# Job: Evaluate Phase 3 RSD (Residual Speculative Decoding)

echo "========================================="
echo "Job: Phase 3 RSD Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Setup environment
module load conda/miniconda
module load cuda/12.1

# Activate conda environment
source activate rvq_training

# Set environment variables
source slurm/environment/paths.env

# Print environment info
echo "Python: $(which python)"
echo "CUDA: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Check all required models exist
RFSQ_MODEL="${RVQ_MODEL_DIR}/rfsq_robust_best.pt"
DRAFT_MODEL="${RVQ_MODEL_DIR}/best_draft_with_projection.pt"
COND_MODEL="${RVQ_MODEL_DIR}/openvla_rfsq_conditional/best_rfsq_head.pt"

if [ ! -f "$RFSQ_MODEL" ]; then
    echo "ERROR: RFSQ model not found: $RFSQ_MODEL"
    exit 1
fi

if [ ! -f "$DRAFT_MODEL" ]; then
    echo "ERROR: Draft model not found: $DRAFT_MODEL"
    exit 1
fi

if [ ! -f "$COND_MODEL" ]; then
    echo "ERROR: Conditional model not found: $COND_MODEL"
    exit 1
fi

# Run Phase 3 evaluation
cd slurm/scripts
python evaluate_phase3_rsd.py \
    --rfsq-model "$RFSQ_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --conditional-model "$COND_MODEL" \
    --num-episodes 50 \
    --monitor-types entropy variance \
    --entropy-threshold 1.5 \
    --variance-threshold 0.1 \
    --device cuda

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
