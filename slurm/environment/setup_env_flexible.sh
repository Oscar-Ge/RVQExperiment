#!/bin/bash
# Flexible setup script for RVQ training environment on Great Lakes
# Handles different module configurations

set -e

echo "=== RVQ Training Environment Setup for Great Lakes ==="
echo ""

# Detect RVQ_ROOT
if [ -z "$RVQ_ROOT" ]; then
    RVQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo "Using detected RVQ_ROOT: $RVQ_ROOT"
else
    echo "Using existing RVQ_ROOT: $RVQ_ROOT"
fi

# ============================================================
# Step 1: Load modules (with fallback)
# ============================================================
echo ""
echo "Step 1: Loading required modules..."

# Try to load conda module (try multiple names)
CONDA_LOADED=false
for module_name in "conda" "miniconda" "anaconda" "Anaconda3" "python/anaconda"; do
    echo "  Trying module: $module_name"
    if module load "$module_name" 2>/dev/null; then
        echo "  ✅ Successfully loaded: $module_name"
        CONDA_LOADED=true
        break
    fi
done

if [ "$CONDA_LOADED" = false ]; then
    echo "  ⚠️  No conda module found via 'module load'"
    echo "  Checking if conda is already available..."

    if command -v conda &> /dev/null; then
        echo "  ✅ Conda found in PATH: $(which conda)"
        CONDA_LOADED=true
    else
        echo ""
        echo "  ❌ Conda not found!"
        echo ""
        echo "  Please do ONE of the following:"
        echo ""
        echo "  Option A: Find the correct module name"
        echo "    1. Run: module spider conda"
        echo "    2. Load the module: module load <module_name>"
        echo "    3. Re-run this script"
        echo ""
        echo "  Option B: Install Miniconda to your home directory"
        echo "    1. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "    2. bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
        echo "    3. \$HOME/miniconda3/bin/conda init bash"
        echo "    4. source ~/.bashrc"
        echo "    5. Re-run this script"
        echo ""
        exit 1
    fi
fi

# Try to load CUDA module (optional, will be loaded in jobs)
echo ""
echo "  Trying to load CUDA module..."
for cuda_version in "cuda/12.1" "cuda/12.0" "cuda/11.8" "cuda"; do
    if module load "$cuda_version" 2>/dev/null; then
        echo "  ✅ Loaded CUDA: $cuda_version"
        break
    fi
done

# ============================================================
# Step 2: Create conda environment
# ============================================================
echo ""
echo "Step 2: Creating conda environment from conda_env.yml..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/conda_env.yml"

if [ ! -f "$ENV_FILE" ]; then
    echo "  ❌ Error: conda_env.yml not found at $ENV_FILE"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^rvq_training "; then
    echo "  ⚠️  Environment 'rvq_training' already exists"
    read -p "  Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        conda env remove -n rvq_training -y
    else
        echo "  Skipping environment creation"
        echo "  If you want to update packages, run:"
        echo "    conda env update -n rvq_training -f $ENV_FILE"
        # Skip to next step
        conda activate rvq_training
        SKIP_INSTALL=true
    fi
fi

if [ "$SKIP_INSTALL" != true ]; then
    echo "  Creating environment (this may take 10-30 minutes)..."
    conda env create -f "$ENV_FILE"
fi

# ============================================================
# Step 3: Activate environment
# ============================================================
echo ""
echo "Step 3: Activating environment..."

# Use conda activate (works in bash with conda init)
if ! conda activate rvq_training 2>/dev/null; then
    echo "  ⚠️  'conda activate' failed, trying 'source activate'..."
    if ! source activate rvq_training 2>/dev/null; then
        echo "  ❌ Failed to activate environment"
        echo "  Try manually: conda activate rvq_training"
        exit 1
    fi
fi

echo "  ✅ Environment activated"

# ============================================================
# Step 4: Install LIBERO
# ============================================================
echo ""
echo "Step 4: Setting up LIBERO..."

# Create data directory
mkdir -p "${RVQ_ROOT}/data"
cd "${RVQ_ROOT}/data"

# Clone LIBERO if not exists
if [ ! -d "LIBERO" ]; then
    echo "  Cloning LIBERO repository..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
else
    echo "  ✅ LIBERO directory already exists"
fi

cd LIBERO

# Apply torch.load fix
echo "  Applying torch.load compatibility fix..."
BENCHMARK_INIT="libero/libero/benchmark/__init__.py"

if [ -f "$BENCHMARK_INIT" ]; then
    # Check if fix is already applied
    if grep -q "weights_only=False" "$BENCHMARK_INIT"; then
        echo "  ✅ Fix already applied"
    else
        echo "  Applying fix..."
        sed -i 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' "$BENCHMARK_INIT"
        echo "  ✅ Fix applied"
    fi
else
    echo "  ⚠️  Warning: $BENCHMARK_INIT not found, skipping fix"
fi

# Install LIBERO
echo "  Installing LIBERO..."
pip install -e .

# ============================================================
# Step 5: Setup LIBERO config
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
# Step 6: Verify installation
# ============================================================
echo ""
echo "Step 6: Verifying installation..."

echo ""
echo "  Checking PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" || {
    echo "  ❌ PyTorch check failed"
    exit 1
}

echo ""
echo "  Checking Transformers..."
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" || {
    echo "  ❌ Transformers check failed"
    exit 1
}

echo ""
echo "  Checking LIBERO..."
python -c "from libero.libero import benchmark; print('  ✅ LIBERO OK')" || {
    echo "  ❌ LIBERO check failed"
    exit 1
}

# ============================================================
# Step 7: Create directories
# ============================================================
echo ""
echo "Step 7: Creating project directories..."

cd "$RVQ_ROOT"
mkdir -p data models logs results .hf_cache
echo "  ✅ Directories created"

# ============================================================
# Setup Complete
# ============================================================
echo ""
echo "========================================="
echo "✅ SETUP COMPLETE!"
echo "========================================="
echo ""
echo "Environment: rvq_training"
echo "Location: $RVQ_ROOT"
echo ""
echo "Next steps:"
echo "  1. Setup Hugging Face token:"
echo "     echo 'hf_YourTokenHere' > ~/.hf_token"
echo "     chmod 600 ~/.hf_token"
echo ""
echo "  2. Test the environment:"
echo "     source activate rvq_training"
echo "     source slurm/environment/paths.env"
echo "     python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "  3. Submit a test job:"
echo "     sbatch slurm/jobs/0_collect_libero_data.sbatch"
echo ""
echo "  4. Or run full pipeline:"
echo "     sbatch slurm/jobs/full_pipeline.sbatch"
echo ""
echo "========================================="
