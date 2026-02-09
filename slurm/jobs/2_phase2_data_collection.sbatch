#!/bin/bash
#SBATCH --job-name=rvq_phase2_data
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

# Job: Collect Phase 2 training data (OpenVLA features + RFSQ labels)

echo "========================================="
echo "Job: Phase 2 Data Collection"
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

# Check Phase 1 model exists
if [ ! -f "${RVQ_MODEL_DIR}/rfsq_robust_best.pt" ]; then
    echo "ERROR: Phase 1 model not found: ${RVQ_MODEL_DIR}/rfsq_robust_best.pt"
    echo "Please run job 1_phase1_rfsq.sbatch first"
    exit 1
fi

# Run Phase 2 data collection
cd slurm/scripts
python collect_phase2_data.py \
    --num-episodes 200 \
    --rfsq-model "${RVQ_MODEL_DIR}/rfsq_robust_best.pt" \
    --output-path "${RVQ_DATA_DIR}/phase2_training_data.pt" \
    --device cuda

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
