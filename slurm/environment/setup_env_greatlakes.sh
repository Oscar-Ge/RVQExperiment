#!/bin/bash
# Setup script specifically for Great Lakes HPC
# Uses module: python3.10-anaconda/2023.03

set -e

echo "=== RVQ Training Environment Setup for Great Lakes ==="
echo ""

# ============================================================
# Load Great Lakes modules
# ============================================================
echo "Step 1: Loading Great Lakes modules..."

module load python3.10-anaconda/2023.03
echo "  ✅ Loaded: python3.10-anaconda/2023.03"

# Try to load CUDA (optional)
if module load cuda/12.1 2>/dev/null; then
    echo "  ✅ Loaded: cuda/12.1"
elif module load cuda/12.0 2>/dev/null; then
    echo "  ✅ Loaded: cuda/12.0"
else
    echo "  ⚠️  No CUDA module found (PyTorch bundled CUDA will be used)"
fi

# ============================================================
# Detect project root
# ============================================================
if [ -z "$RVQ_ROOT" ]; then
    RVQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo ""
    echo "Using detected RVQ_ROOT: $RVQ_ROOT"
else
    echo ""
    echo "Using existing RVQ_ROOT: $RVQ_ROOT"
fi

# ============================================================
# Create conda environment
# ============================================================
echo ""
echo "Step 2: Creating conda environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/conda_env.yml"

if [ ! -f "$ENV_FILE" ]; then
    echo "  ❌ Error: conda_env.yml not found at $ENV_FILE"
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "^rvq_training "; then
    echo "  ⚠️  Environment 'rvq_training' already exists"
    read -p "  Update existing environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Updating environment..."
        conda env update -n rvq_training -f "$ENV_FILE" --prune
    else
        echo "  Skipping environment creation"
        SKIP_ENV=true
    fi
else
    echo "  Creating environment (this may take 10-30 minutes)..."
    conda env create -f "$ENV_FILE"
fi

# ============================================================
# Activate environment
# ============================================================
echo ""
echo "Step 3: Activating environment..."

conda activate rvq_training
echo "  ✅ Environment activated"

# ============================================================
# Install LIBERO
# ============================================================
echo ""
echo "Step 4: Setting up LIBERO..."

mkdir -p "${RVQ_ROOT}/data"
cd "${RVQ_ROOT}/data"

if [ ! -d "LIBERO" ]; then
    echo "  Cloning LIBERO repository..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
else
    echo "  ✅ LIBERO directory already exists"
fi

cd LIBERO

# Apply torch.load fix
BENCHMARK_INIT="libero/libero/benchmark/__init__.py"
if [ -f "$BENCHMARK_INIT" ]; then
    if grep -q "weights_only=False" "$BENCHMARK_INIT"; then
        echo "  ✅ torch.load fix already applied"
    else
        echo "  Applying torch.load fix..."
        sed -i 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' "$BENCHMARK_INIT"
        echo "  ✅ Fix applied"
    fi
else
    echo "  ⚠️  Warning: $BENCHMARK_INIT not found"
fi

# Install LIBERO
if [ "$SKIP_ENV" != true ]; then
    echo "  Installing LIBERO..."
    pip install -e .
else
    echo "  Skipping LIBERO installation"
fi

# ============================================================
# Setup LIBERO config
# ============================================================
echo ""
echo "Step 5: Configuring LIBERO..."

mkdir -p ~/.libero
LIBERO_CONFIG=~/.libero/config.yaml
LIBERO_ROOT="${RVQ_ROOT}/data/LIBERO/libero/libero"

cat > "$LIBERO_CONFIG" << EOF
benchmark_root: ${LIBERO_ROOT}
bddl_files: ${LIBERO_ROOT}/bddl_files
init_states: ${LIBERO_ROOT}/init_files
EOF

echo "  ✅ Config written to $LIBERO_CONFIG"

# ============================================================
# Verify installation
# ============================================================
echo ""
echo "Step 6: Verifying installation..."

echo ""
echo "  PyTorch:"
python -c "import torch; print(f'    Version: {torch.__version__}'); print(f'    CUDA: {torch.cuda.is_available()}')" || {
    echo "  ❌ PyTorch check failed"
    exit 1
}

echo ""
echo "  Transformers:"
python -c "import transformers; print(f'    Version: {transformers.__version__}')" || {
    echo "  ❌ Transformers check failed"
    exit 1
}

echo ""
echo "  LIBERO:"
python -c "from libero.libero import benchmark; print('    ✅ OK')" || {
    echo "  ❌ LIBERO check failed"
    exit 1
}

# ============================================================
# Create project directories
# ============================================================
echo ""
echo "Step 7: Creating project directories..."

cd "$RVQ_ROOT"
mkdir -p data models logs results .hf_cache
echo "  ✅ Directories created"

# ============================================================
# Complete
# ============================================================
echo ""
echo "========================================="
echo "✅ SETUP COMPLETE!"
echo "========================================="
echo ""
echo "Environment: rvq_training"
echo "Root: $RVQ_ROOT"
echo ""
echo "Next steps:"
echo ""
echo "  1. Setup Hugging Face token:"
echo "     echo 'hf_YourToken' > ~/.hf_token"
echo "     chmod 600 ~/.hf_token"
echo ""
echo "  2. Update job scripts:"
echo "     bash slurm/environment/update_job_scripts.sh"
echo ""
echo "  3. Submit jobs:"
echo "     cd $RVQ_ROOT"
echo "     sbatch slurm/jobs/full_pipeline.sbatch"
echo ""
echo "========================================="
