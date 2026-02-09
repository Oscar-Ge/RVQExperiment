#!/bin/bash
#SBATCH --job-name=rvq_phase1_training
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gecm@umich.edu

# Job: Train Phase 1 RFSQ AutoEncoder

echo "========================================="
echo "Job: Phase 1 RFSQ Training"
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

# Check data file exists
if [ ! -f "${RVQ_DATA_DIR}/libero_actions_normalized.pt" ]; then
    echo "ERROR: Data file not found: ${RVQ_DATA_DIR}/libero_actions_normalized.pt"
    echo "Please run job 0_collect_libero_data.sbatch first"
    exit 1
fi

# Run Phase 1 training
cd slurm/scripts
python train_phase1_rfsq.py \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --hidden_dim 16 \
    --num_layers 8 \
    --device cuda \
    --checkpoint_every 10

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
