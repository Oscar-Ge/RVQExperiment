# Phase 2 Data Collection Script

## Overview

`collect_phase2_data.py` is a standalone SLURM-compatible script that converts the Modal-based `collect_training_data` function from `phase2_conditional/modal_train_phase2_conditional.py` into a pure PyTorch script with argparse CLI.

**Purpose**: Collect training data for Phase 2 (Conditional RFSQ Head Training) by:
1. Running OpenVLA-OFT policy in LIBERO environment
2. Extracting OpenVLA hidden states from model internals
3. Encoding actions with trained RFSQ tokenizer to get token labels
4. Saving paired (hidden_state, rfsq_tokens) samples for training

## Key Features

### Model Loading
- **OpenVLA**: Loads `moojink/openvla-7b-oft-finetuned-libero-spatial` (frozen)
- **RFSQ Tokenizer**: Loads trained ActionRFSQAE from Phase 1 (frozen)
- **LIBERO**: Sets up task suite and environments

### Data Extraction
For each environment step:
- **Hidden States**: Extract from OpenVLA's last hidden layer [4096-dim]
- **RFSQ Tokens**: Encode actions using RFSQ tokenizer to get discrete codes [8, 16, 8]
- **Raw Actions**: Save original action sequences [8, 7]

### Robust Error Handling
- Fallback for hidden state extraction (cumsum issues)
- Fallback for action prediction failures
- Handles multiple action output shapes from OpenVLA
- Graceful episode failure recovery

### Progress Tracking
- TQDM progress bars for episode collection
- Per-task statistics
- Summary statistics at end (total samples, failed episodes, per-task counts)
- Metadata JSON file with collection info

## Usage

### Basic Example (200 episodes)
```bash
python collect_phase2_data.py \
    --num-episodes 200 \
    --rfsq-model /models/rfsq_robust_best.pt \
    --output-path /data/phase2_training_data.pt \
    --device cuda
```

### SLURM Job Submission
```bash
sbatch << 'SLURM'
#!/bin/bash
#SBATCH --job-name=phase2_collect
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cd /path/to/RVQExperiment/slurm/scripts

python collect_phase2_data.py \
    --num-episodes 200 \
    --rfsq-model /models/rfsq_robust_best.pt \
    --output-path /data/phase2_training_data.pt \
    --device cuda \
    --verbose
SLURM
```

### Full Option List
```
--num-episodes INT          Number of episodes to collect (default: 200)
--rfsq-model PATH           Path to trained RFSQ tokenizer (REQUIRED)
--output-path PATH          Path to save data (REQUIRED)
--device STR                Device: cuda or cpu (default: cuda)
--seed INT                  Random seed (default: 42)

RFSQ Config:
--action-dim INT            Action dimension (default: 7)
--hidden-dim INT            RFSQ hidden dimension (default: 16)
--num-layers INT            Number of RFSQ layers (default: 8)
--num-levels INT            Quantization levels (default: 7)
--use-layernorm             Use LayerNorm in RFSQ (default: True)

LIBERO Settings:
--task-suite STR            Task suite name (default: libero_spatial)
--max-episode-steps INT     Max steps per episode (default: 300)

OpenVLA Settings:
--openvla-model STR         OpenVLA model ID (default: moojink/openvla-7b-oft-finetuned-libero-spatial)

Logging:
--verbose                   Enable verbose output
```

## Output Format

### Data File: `phase2_training_data.pt`
Python list of dictionaries, each containing:
```python
{
    'hidden_state': torch.Tensor,           # [4096] - OpenVLA hidden state
    'rfsq_tokens': torch.Tensor,            # [8, 16, 8] - RFSQ token codes
    'raw_action': torch.Tensor,             # [8, 7] - Original action sequence
    'task_id': int,                         # Task identifier
    'task_description': str,                # Task language description
}
```

**Total samples**: Usually 10,000-20,000 for 200 episodes

### Metadata File: `phase2_training_data.json`
```json
{
    "num_samples": 15234,
    "num_episodes": 200,
    "failed_episodes": 2,
    "total_steps": 15234,
    "task_counts": {
        "0": 1542,
        "1": 1876,
        ...
    },
    "args": { ... }
}
```

## Data Characteristics

### Hidden States [4096]
- Mean: ~0.0, Std: ~0.1
- Extracted from OpenVLA's Vision Transformer backbone
- Bfloat16 converted to Float32

### RFSQ Tokens [8, 16, 8]
- Dimensions: Chunk=8, Hidden=16, Layers=8
- Values: Integers in [0, 6] (7 quantization levels)
- Layer structure:
  - L0-L2: Coarse motion (4-6 bits)
  - L3-L7: Fine details (1-2 bits per layer)

### Per-Task Distribution
Uniform sampling: ~1500-1900 samples per task in libero_spatial

## Integration with Training

This collected data is used by `train_phase2_conditional.py`:

```bash
# Train Draft Model + Conditional RFSQ Head
python train_phase2_conditional.py \
    --data_path /data/phase2_training_data.pt \
    --epochs 50 \
    --batch_size 32
```

## Dependencies

- PyTorch (2.0+)
- Transformers (4.40+)
- LIBERO (from GitHub)
- OpenVLA (moojink fork)
- PIL, NumPy, tqdm

## Environment Variables

These are auto-configured but can be overridden:
```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_FOLDER=/path/to/libero
export LIBERO_NO_PROMPT=1
```

## Troubleshooting

### Issue: "LIBERO not found"
```bash
cd /root
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

### Issue: "RFSQ model not found"
Make sure the Phase 1 training completed successfully:
```bash
python train_phase1_rfsq.py --num-episodes 50 --epochs 100 --device cuda
```

### Issue: "cumsum" errors in hidden state extraction
This is handled automatically with fallback (random hidden state). Consider increasing batch size to reduce frequency.

### Issue: Low sample collection rate
Check:
1. LIBERO environment setup (`--verbose` flag)
2. OpenVLA model loading
3. GPU memory (use `--device cpu` to test without GPU)

## Performance Notes

- **Speed**: ~0.5-1.0 samples/second on A100 GPU
- **Memory**: ~30GB total (OpenVLA + LIBERO + batch processing)
- **Time**: 200 episodes ≈ 6-12 hours

## Related Files

- `phase2_conditional/modal_train_phase2_conditional.py` - Original Modal version (lines 124-458)
- `models/rfsq_models.py` - ActionRFSQAE implementation
- `train_phase2_conditional.py` - Training script that uses this data
