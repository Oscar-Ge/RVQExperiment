#!/bin/bash
#SBATCH --job-name=rvq_baseline_openvla
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

# Job: Evaluate OpenVLA-OFT baseline (no RSD)

echo "========================================="
echo "Job: OpenVLA-OFT Baseline Evaluation"
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

# Run baseline evaluation
cd slurm/scripts
python evaluate_openvla_baseline.py \
    --num-episodes 50 \
    --save-failure-videos \
    --device cuda

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
