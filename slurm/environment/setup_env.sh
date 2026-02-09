#!/bin/bash
# One-time setup script for RVQ training environment on Great Lakes

set -e

echo "=== RVQ Training Environment Setup for Great Lakes ==="

# Get project root from environment or use default
if [ -z "${RVQ_ROOT}" ]; then
    export RVQ_ROOT="${HOME}/RVQExperiment"
    echo "Using default RVQ_ROOT: ${RVQ_ROOT}"
fi

# Load modules
echo "Loading required modules..."
module load conda/miniconda
module load cuda/12.1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Create conda environment
echo "Creating conda environment from conda_env.yml..."
conda env create -f "${SCRIPT_DIR}/conda_env.yml" || {
    echo "Environment already exists, updating instead..."
    conda env update -f "${SCRIPT_DIR}/conda_env.yml" --prune
}

# Activate environment
echo "Activating environment..."
source activate rvq_training

# Setup data directories
echo "Setting up data directories..."
source "${SCRIPT_DIR}/paths.env"

# Clone and install LIBERO
echo "Setting up LIBERO..."
cd "${RVQ_DATA_DIR}"
if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO

# Apply torch.load fix for newer PyTorch versions
echo "Applying torch.load compatibility fix..."
sed -i 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
    libero/libero/benchmark/__init__.py || echo "Fix already applied or file not found"

# Install LIBERO
echo "Installing LIBERO..."
pip install -e .

# Setup LIBERO config
echo "Setting up LIBERO configuration..."
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << EOF
benchmark_root: ${RVQ_DATA_DIR}/LIBERO/libero/libero
bddl_files: ${RVQ_DATA_DIR}/LIBERO/libero/libero/bddl_files
init_states: ${RVQ_DATA_DIR}/LIBERO/libero/libero/init_files
EOF

# Verify installations
echo ""
echo "=== Verifying Installation ==="

echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "Checking LIBERO..."
python -c "from libero.libero import benchmark; print('LIBERO: OK')"

echo "Checking OpenVLA..."
python -c "from openvla import OpenVLAModel; print('OpenVLA: OK')"

echo "Checking imageio (for video recording)..."
python -c "import imageio; print(f'imageio: {imageio.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo "Activate environment with: source activate rvq_training"
echo "Set environment variables with: source ${SCRIPT_DIR}/paths.env"
echo ""
echo "Next steps:"
echo "  1. Submit data collection job: sbatch slurm/jobs/0_collect_libero_data.sbatch"
echo "  2. Or run full pipeline: sbatch slurm/jobs/full_pipeline.sbatch"
