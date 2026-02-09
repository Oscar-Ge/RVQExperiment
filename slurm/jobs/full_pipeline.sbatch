#!/bin/bash
#SBATCH --job-name=rvq_full_pipeline
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gecm@umich.edu

# Job: Run full RVQ experiment pipeline (all phases sequentially)

echo "========================================="
echo "Job: Full RVQ Experiment Pipeline"
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

cd slurm/scripts

# ============================================================
# Phase 0: Data Collection
# ============================================================
echo ""
echo "========================================="
echo "Phase 0: Collecting LIBERO Action Data"
echo "========================================="

python collect_libero_data.py \
    --task-suite libero_spatial \
    --num-episodes 100 \
    --chunk-size 8 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 0 (data collection) failed"
    exit 1
fi

# ============================================================
# Phase 1: RFSQ AutoEncoder Training
# ============================================================
echo ""
echo "========================================="
echo "Phase 1: Training RFSQ AutoEncoder"
echo "========================================="

python train_phase1_rfsq.py \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --hidden_dim 16 \
    --num_layers 8 \
    --device cuda \
    --checkpoint_every 10

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 1 (RFSQ training) failed"
    exit 1
fi

# ============================================================
# Phase 2a: Collect Training Data
# ============================================================
echo ""
echo "========================================="
echo "Phase 2a: Collecting Phase 2 Training Data"
echo "========================================="

python collect_phase2_data.py \
    --num-episodes 200 \
    --rfsq-model "${RVQ_MODEL_DIR}/rfsq_robust_best.pt" \
    --output-path "${RVQ_DATA_DIR}/phase2_training_data.pt" \
    --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 2a (data collection) failed"
    exit 1
fi

# ============================================================
# Phase 2b: Train Draft + Conditional Models
# ============================================================
echo ""
echo "========================================="
echo "Phase 2b: Training Draft + Conditional Models"
echo "========================================="

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

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 2b (model training) failed"
    exit 1
fi

# ============================================================
# Phase 3: RSD Evaluation
# ============================================================
echo ""
echo "========================================="
echo "Phase 3: Evaluating RSD Model"
echo "========================================="

python evaluate_phase3_rsd.py \
    --rfsq-model "${RVQ_MODEL_DIR}/rfsq_robust_best.pt" \
    --draft-model "${RVQ_MODEL_DIR}/best_draft_with_projection.pt" \
    --conditional-model "${RVQ_MODEL_DIR}/openvla_rfsq_conditional/best_rfsq_head.pt" \
    --num-episodes 50 \
    --monitor-types entropy variance \
    --entropy-threshold 1.5 \
    --variance-threshold 0.1 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 3 (evaluation) failed"
    exit 1
fi

# ============================================================
# Baseline: OpenVLA-OFT Evaluation
# ============================================================
echo ""
echo "========================================="
echo "Baseline: Evaluating OpenVLA-OFT"
echo "========================================="

python evaluate_openvla_baseline.py \
    --num-episodes 50 \
    --save-failure-videos \
    --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Baseline evaluation failed"
    exit 1
fi

# ============================================================
# Pipeline Complete
# ============================================================
echo ""
echo "========================================="
echo "FULL PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - RFSQ model: ${RVQ_MODEL_DIR}/rfsq_robust_best.pt"
echo "  - Draft model: ${RVQ_MODEL_DIR}/best_draft_with_projection.pt"
echo "  - Conditional model: ${RVQ_MODEL_DIR}/openvla_rfsq_conditional/best_rfsq_head.pt"
echo "  - RSD results: ${RVQ_RESULTS_DIR}/rsd_evaluation_results.json"
echo "  - Baseline results: ${RVQ_RESULTS_DIR}/baseline_results.json"
echo "  - Logs: ${RVQ_LOG_DIR}/"
echo "========================================="
