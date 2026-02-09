# Great Lakes Quick Start Guide

**遇到模块加载问题？这个指南帮你快速解决！**

## 🚨 问题：找不到 conda/miniconda 模块

如果你看到这个错误：
```
The following module(s) are unknown: "conda/miniconda"
```

**不要慌！** 按照下面的步骤操作。

---

## 📋 解决步骤

### 步骤 1: 找到正确的模块名称

```bash
# 在 Great Lakes 上运行
module spider conda
```

可能的输出示例：
- `python/anaconda`
- `anaconda3`
- `Anaconda3/2023.03`
- 或者其他类似名称

**记下找到的模块名称！**

### 步骤 2A: 如果找到了 Conda 模块

```bash
# 加载找到的模块（用实际名称替换）
module load anaconda3  # 或 python/anaconda, 或其他你找到的名称

# 验证
conda --version

# 运行灵活的安装脚本
cd ~/RVQExperiment
bash slurm/environment/setup_env_flexible.sh
```

### 步骤 2B: 如果没有找到任何 Conda 模块（推荐）

**直接安装 Miniconda 到你的 home 目录**（只需要 5-10 分钟）：

```bash
# 1. 下载 Miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 2. 安装（-b = batch mode，不需要手动确认）
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 3. 初始化
$HOME/miniconda3/bin/conda init bash

# 4. 重新加载 shell 配置
source ~/.bashrc

# 5. 验证安装
conda --version
which conda  # 应该显示 ~/miniconda3/bin/conda

# 6. 运行安装脚本
cd ~/RVQExperiment
bash slurm/environment/setup_env_flexible.sh
```

---

## ✅ 完整安装流程（从头开始）

假设你已经在 Great Lakes 上：

```bash
# 1. 克隆仓库（如果还没有）
cd ~
git clone https://github.com/Oscar-Ge/RVQExperiment.git
# 或者从 bundle 克隆（如果你之前创建了）

# 2. 进入项目目录
cd RVQExperiment

# 3. 安装 Miniconda（如果系统没有 conda）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# 4. 运行灵活的安装脚本
bash slurm/environment/setup_env_flexible.sh

# 5. 设置 Hugging Face token
echo "hf_你的token" > ~/.hf_token
chmod 600 ~/.hf_token

# 6. 验证环境
source activate rvq_training
source slurm/environment/paths.env
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from libero.libero import benchmark; print('LIBERO OK')"
```

**如果所有验证都通过，你就可以提交任务了！**

---

## 🎯 提交第一个测试任务

```bash
# 先测试一个小任务（只收集 1 个 episode）
cd ~/RVQExperiment

# 创建测试脚本
cat > slurm/jobs/test_minimal.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=rvq_test
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Load conda (use YOUR method)
# Option A: If you installed Miniconda to home
source ~/miniconda3/etc/profile.d/conda.sh

# Option B: If using module
# module load anaconda3  # or whatever module you found

# Activate environment
conda activate rvq_training
source slurm/environment/paths.env

# Print info
echo "Python: $(which python)"
nvidia-smi

# Simple test
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from libero.libero import benchmark; print('LIBERO OK')"

echo "✅ Test passed!"
EOF

# 提交测试任务
mkdir -p logs
sbatch slurm/jobs/test_minimal.sbatch

# 查看状态
squeue -u gecm

# 查看输出（等任务完成后）
cat logs/test_*.out
```

---

## 🔧 更新 SLURM 作业脚本

所有的 `.sbatch` 文件需要更新模块加载部分。

**如果你用 Miniconda（推荐）**，把所有 `.sbatch` 文件中的：

```bash
# 旧的（不工作）
module load conda/miniconda
module load cuda/12.1

source activate rvq_training
```

改成：

```bash
# 新的（灵活）
# Activate conda (installed in home directory)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rvq_training

# Try to load CUDA (optional, PyTorch has bundled CUDA)
module load cuda/12.1 2>/dev/null || echo "Using PyTorch bundled CUDA"
```

**如果你用系统模块**，改成：

```bash
# 新的（使用你找到的模块名称）
module load anaconda3  # 替换成你的模块名称
module load cuda/12.1 2>/dev/null || true

conda activate rvq_training
```

---

## 📝 快速批量更新所有作业脚本

```bash
cd ~/RVQExperiment/slurm/jobs

# 备份原文件
mkdir -p backup
cp *.sbatch backup/

# 更新所有脚本（如果你用 Miniconda）
for file in *.sbatch; do
    sed -i 's|module load conda/miniconda|source ~/miniconda3/etc/profile.d/conda.sh|g' "$file"
    sed -i 's|module load cuda/12.1|module load cuda/12.1 2>/dev/null \|\| echo "Using PyTorch bundled CUDA"|g' "$file"
    sed -i 's|source activate|conda activate|g' "$file"
done

echo "✅ All job scripts updated!"
```

---

## ❓ 常见问题

### Q: 安装 Miniconda 需要多长时间？
**A**: 5-10 分钟（下载 + 安装）

### Q: Miniconda 会占用多少空间？
**A**: 约 3-5 GB（包括 conda 环境）

### Q: 我能用系统的 Python 吗？
**A**: 不推荐。Great Lakes 的系统 Python 可能缺少必要的包，用 conda 环境更稳定。

### Q: 我需要在每个作业中都加载 conda 吗？
**A**: 是的，每个 `.sbatch` 脚本都需要激活 conda 环境。

### Q: CUDA 模块找不到怎么办？
**A**: 没关系！PyTorch 自带 CUDA，不需要系统 CUDA 模块也能用 GPU。

---

## 🆘 如果还有问题

1. **查看详细错误日志**:
   ```bash
   cat logs/slurm_*_<JOB_ID>.err
   ```

2. **检查环境**:
   ```bash
   conda activate rvq_training
   conda list | grep torch
   conda list | grep transformers
   ```

3. **手动测试脚本**:
   ```bash
   # 在登录节点上测试（不用 GPU）
   cd ~/RVQExperiment/slurm/scripts
   python -c "from models.rfsq_models import ActionRFSQAE; print('✅ Models load OK')"
   ```

4. **联系 Great Lakes 支持**:
   ```
   hpc-support@umich.edu
   ```

---

## ✅ 成功标志

当你看到这些输出时，说明一切正常：

```
✅ SETUP COMPLETE!
PyTorch: 2.2.0 (or later)
CUDA available: True
LIBERO OK
```

然后你就可以运行：

```bash
sbatch slurm/jobs/full_pipeline.sbatch
```

祝你好运！🚀
