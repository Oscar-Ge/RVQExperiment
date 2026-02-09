#!/bin/bash
# Script to update all SLURM job scripts for Great Lakes
# Updates module loading to use: python3.10-anaconda/2023.03

set -e

echo "=== Updating SLURM Job Scripts for Great Lakes ==="
echo ""

# Detect project root
if [ -z "$RVQ_ROOT" ]; then
    RVQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

JOBS_DIR="$RVQ_ROOT/slurm/jobs"

if [ ! -d "$JOBS_DIR" ]; then
    echo "❌ Error: Jobs directory not found at $JOBS_DIR"
    exit 1
fi

cd "$JOBS_DIR"

# Backup original files
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Creating backup in: $BACKUP_DIR"
cp *.sbatch "$BACKUP_DIR/"
echo "  ✅ Backup created"

echo ""
echo "Updating job scripts..."

# Update all .sbatch files
for file in *.sbatch; do
    echo "  Processing: $file"

    # Replace conda module
    sed -i 's|module load conda/miniconda|module load python3.10-anaconda/2023.03|g' "$file"

    # Make CUDA loading more robust
    sed -i 's|module load cuda/12.1$|module load cuda/12.1 2>/dev/null \|\| module load cuda/12.0 2>/dev/null \|\| echo "Using PyTorch bundled CUDA"|g' "$file"

    # Update activation command
    sed -i 's|source activate rvq_training|conda activate rvq_training|g' "$file"

    echo "    ✅ Updated"
done

echo ""
echo "========================================="
echo "✅ All job scripts updated!"
echo "========================================="
echo ""
echo "Changes made:"
echo "  - Module: conda/miniconda → python3.10-anaconda/2023.03"
echo "  - CUDA: Added fallback to cuda/12.0 and PyTorch CUDA"
echo "  - Activation: source activate → conda activate"
echo ""
echo "Backup location: $JOBS_DIR/$BACKUP_DIR"
echo ""
echo "Verify changes:"
echo "  head -30 $JOBS_DIR/0_collect_libero_data.sbatch"
echo ""
echo "Next step:"
echo "  sbatch $JOBS_DIR/full_pipeline.sbatch"
echo ""
