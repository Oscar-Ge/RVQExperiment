#!/bin/bash
#SBATCH --job-name=rvq_phase2_training
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gecm@umich.edu

# Job: Train Phase 2 models (Draft + Conditional)

echo "========================================="
echo "Job: Phase 2 Model Training"
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

# Check Phase 2 data exists
if [ ! -f "${RVQ_DATA_DIR}/phase2_training_data.pt" ]; then
    echo "ERROR: Phase 2 data not found: ${RVQ_DATA_DIR}/phase2_training_data.pt"
    echo "Please run job 2_phase2_data_collection.sbatch first"
    exit 1
fi

# Run Phase 2 training (both models)
cd slurm/scripts
python train_phase2_conditional.py \
    --data_path "${RVQ_DATA_DIR}/phase2_training_data.pt" \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --draft_hidden_dim 512 \
    --cond_hidden_dim 1024 \
    --token_embed_dim 64 \
    --device cuda \
    --checkpoint_every 5

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
