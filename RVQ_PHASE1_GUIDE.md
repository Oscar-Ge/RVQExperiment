# Phase 1: RVQ Tokenizer - Complete Guide

**目标**: 训练一个 RVQ (Residual Vector Quantization) tokenizer，验证机器人动作的层级结构假设。

**时间**: 预计 1-2 周（实际运行时间约 2-3 小时）

---

## 🎯 Phase 1 目标

1. **训练 VQ-VAE**: 将 (T, 7) 动作编码为 8 层 RVQ codes
2. **验证层级假设**:
   - Layer 1-2 (粗略) → MSE ≈ 0.01
   - Layer 1-8 (精细) → MSE < 0.001
3. **对比 DCT**: 证明 RVQ 优于 DCT

---

## 📋 前置要求

### 1. 已完成 DCT 实验
```bash
# 确保 DCT 实验已经跑通
python test_dct_compression.py
python analyze_libero_actions.py --num_episodes 20
```

### 2. 环境要求
- **DCT 实验的所有依赖** +
- **PyTorch** (已安装，用于 π0.5)
- **CUDA GPU** (推荐，CPU 训练会很慢)

### 3. 验证环境
```bash
# 检查 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 应该输出类似:
# PyTorch 2.x.x, CUDA: True
```

---

## 🚀 快速开始（3 步走）

### Step 1: 单元测试（5 分钟）

**目的**: 验证 RVQ tokenizer 实现正确

```bash
python test_rvq_tokenizer.py
```

**预期输出**:
```
================================================================================
RVQ TOKENIZER TEST SUITE
================================================================================
================================================================================
TEST 1: VECTOR QUANTIZER
================================================================================

Input shape: torch.Size([2, 10, 64])
Quantized shape: torch.Size([2, 10, 64])
Indices shape: torch.Size([2, 10])
VQ loss: 0.234567

✅ VectorQuantizer test passed!

================================================================================
TEST 2: RESIDUAL VECTOR QUANTIZER
================================================================================
...

================================================================================
TEST SUMMARY
================================================================================
✅ All tests passed!

Next steps:
  1. Run: python train_rvq_tokenizer.py --num_episodes 50 --epochs 100
  2. Train RVQ tokenizer on real LIBERO actions
  3. Run: python analyze_rvq_compression.py --model rvq_tokenizer.pt
  4. Compare results with DCT compression
================================================================================
```

---

### Step 2: 训练 RVQ Tokenizer（30-60 分钟）

**目的**: 在真实 LIBERO 动作上训练 RVQ

```bash
# 基础训练（50 episodes，100 epochs）
python train_rvq_tokenizer.py \
    --task_suite libero_spatial \
    --num_episodes 50 \
    --epochs 100 \
    --batch_size 32 \
    --device cuda

# 如果想快速验证（减少时间）
python train_rvq_tokenizer.py \
    --num_episodes 20 \
    --epochs 50 \
    --device cuda
```

**预期输出**:
```
================================================================================
RVQ TOKENIZER TRAINING
================================================================================
================================================================================
COLLECTING π0.5 ACTIONS ON LIBERO
================================================================================

[1/4] Loading π0.5 policy...
✓ Policy loaded successfully!

[2/4] Loading LIBERO task: libero_spatial - Task 0
  Task: pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  Description: pick up the black bowl between the plate and the ramekin and place it on the plate

[3/4] Creating LIBERO environment...

[4/4] Collecting actions from 50 episodes...
  Episode 1/50: ✅ (steps: 88)
  Episode 2/50: ✅ (steps: 138)
  ...

✅ Collected 2187 action chunks
   Success rate: 96.0%

================================================================================
TRAINING RVQ TOKENIZER
================================================================================

Dataset:
  Action chunks: 2187
  Chunk size: 10
  Action dim: 7

Model config:
  Num layers: 8
  Hidden dim: 64
  Codebook size: 256
  Residual dropout: 0.1

Training config:
  Epochs: 100
  Batch size: 32
  Learning rate: 0.001
  Device: cuda

================================================================================
TRAINING PROGRESS
================================================================================
Epoch 10/100:
  Recon Loss: 0.012345
  VQ Loss: 0.234567
  Total Loss: 0.246912

Epoch 20/100:
  Recon Loss: 0.005678
  VQ Loss: 0.198765
  Total Loss: 0.204443

...

Epoch 100/100:
  Recon Loss: 0.000234
  VQ Loss: 0.156789
  Total Loss: 0.157023

✅ Model saved to rvq_tokenizer.pt

📊 Training curves saved to training_history.png

================================================================================
QUICK VALIDATION
================================================================================

Test reconstruction MSE: 0.000456
✅ Validation passed! MSE < 0.01

================================================================================
NEXT STEPS
================================================================================
✅ RVQ tokenizer trained successfully!

Model saved to: rvq_tokenizer.pt

Next:
  1. Run: python analyze_rvq_compression.py --model rvq_tokenizer.pt
  2. Test different numbers of layers (1-8)
  3. Plot MSE vs. number of layers
================================================================================
```

**训练时间估计**:
- 50 episodes collection: 15-20 分钟
- 100 epochs training: 10-30 分钟（取决于 GPU）
- 总计: 30-60 分钟

---

### Step 3: 分析 RVQ 压缩（10 分钟）

**目的**: 测试不同层数的重建质量，复现类似 DCT 的图表

```bash
# 分析训练好的模型
python analyze_rvq_compression.py \
    --model rvq_tokenizer.pt \
    --task_suite libero_spatial \
    --num_episodes 20 \
    --device cuda
```

**预期输出**:
```
================================================================================
RVQ COMPRESSION ANALYSIS
================================================================================

[1/3] Loading trained RVQ model...
✓ Loaded RVQ tokenizer from rvq_tokenizer.pt
  Config: {'action_dim': 7, 'chunk_size': 10, 'num_layers': 8, ...}

[2/3] Collecting test actions...
✅ Collected 874 action chunks

[3/3] Analyzing compression...
================================================================================
ANALYZING RVQ COMPRESSION
================================================================================

Layers=1:
  MSE: 0.023456 ± 0.008923
  Tokens: 10.0
  Compression: 28.00x

Layers=2:
  MSE: 0.008765 ± 0.003456
  Tokens: 20.0
  Compression: 14.00x

Layers=3:
  MSE: 0.004321 ± 0.001789
  Tokens: 30.0
  Compression: 9.33x

Layers=4:
  MSE: 0.002109 ± 0.000987
  Tokens: 40.0
  Compression: 7.00x

Layers=5:
  MSE: 0.001234 ± 0.000567
  Tokens: 50.0
  Compression: 5.60x

Layers=6:
  MSE: 0.000789 ± 0.000345
  Tokens: 60.0
  Compression: 4.67x

Layers=7:
  MSE: 0.000456 ± 0.000234
  Tokens: 70.0
  Compression: 4.00x

Layers=8:
  MSE: 0.000123 ± 0.000089
  Tokens: 80.0
  Compression: 3.50x

📊 Plot saved to rvq_compression_analysis.png

================================================================================
SUMMARY
================================================================================

✅ Optimal setting: 3 RVQ layers
   MSE: 0.004321 ± 0.001789
   Compression: 9.33x
   Tokens per chunk: 30.0

🎯 Best compression with excellent MSE (<0.01):
   3 layers
   MSE: 0.004321
   Compression: 9.33x
   → Coarse layers (1-3) capture key motion!

📊 Layer comparison:
   Layer 1-2 (coarse): MSE=0.008765
   Layer 1-8 (fine):   MSE=0.000123
   → Improvement from fine layers: 98.6%

✅ HYPOTHESIS VALIDATED!
   Layer 1-2 alone achieve MSE < 0.01
   → Can use coarse layers for simple motions
   → Only activate fine layers for complex tasks

================================================================================
NEXT STEPS
================================================================================
✅ RVQ compression analysis complete!

Based on results:
  ✅ Ready for Phase 2: Train VLA policy to predict RVQ tokens
  ✅ Ready for Phase 3: Implement adaptive inference
     - Use layers 1-3 for coarse prediction
     - Activate all layers when uncertainty is high
================================================================================
```

**生成的图表**: `rvq_compression_analysis.png`
- 左图: MSE vs. RVQ Layers (对数尺度)
- 右图: Compression Ratio vs. RVQ Layers

---

## 📊 预期结果

### ✅ 成功的标准

**Hypothesis 1: 层级对应复杂度**
```
Layer 1-2: MSE ≈ 0.01 (粗略运动)
Layer 3-4: MSE ≈ 0.001 (中等精度)
Layer 5-8: MSE < 0.0001 (精细修正)
```

**Hypothesis 2: 优于 DCT**
```
RVQ (3 layers): MSE ≈ 0.004, Compression = 9.33x
DCT (keep=3):   MSE ≈ 0.007, Compression = 3.33x

→ RVQ 同等 MSE 下，压缩比更高 ✅
```

---

## 📈 与 DCT 对比

| 方法 | MSE < 0.01 时层数 | 压缩比 | 优势 |
|------|------------------|--------|------|
| **DCT** | 3 系数 | 3.33x | 简单，无需训练 |
| **RVQ** | 2-3 层 | 9-14x | 更高压缩，学习数据分布 |

---

## 🔬 高级用法

### 1. 调整超参数

```bash
# 增加模型容量（如果 MSE 不够低）
python train_rvq_tokenizer.py \
    --num_episodes 100 \
    --hidden_dim 128 \
    --codebook_size 512 \
    --epochs 200

# 减少模型大小（如果过拟合）
python train_rvq_tokenizer.py \
    --hidden_dim 32 \
    --codebook_size 128 \
    --epochs 50
```

### 2. 测试不同任务

```bash
# Spatial tasks
python train_rvq_tokenizer.py --task_suite libero_spatial --task_id 0

# Object tasks (更复杂)
python train_rvq_tokenizer.py --task_suite libero_object --task_id 2

# Long-horizon tasks
python train_rvq_tokenizer.py --task_suite libero_10 --task_id 0
```

### 3. 使用 CPU（如果没有 GPU）

```bash
python train_rvq_tokenizer.py \
    --num_episodes 20 \
    --epochs 50 \
    --batch_size 16 \
    --device cpu
```

**注意**: CPU 训练会慢 10-20 倍。

---

## 🐛 故障排除

### 问题 1: 训练 Loss 不下降

**症状**:
```
Epoch 50/100:
  Recon Loss: 0.123456 (没变化)
  VQ Loss: 0.234567
```

**解决方案**:
1. 增加 learning rate:
   ```bash
   python train_rvq_tokenizer.py --lr 5e-3
   ```
2. 减少 residual dropout:
   ```bash
   python train_rvq_tokenizer.py --residual_dropout 0.0
   ```
3. 增加模型容量:
   ```bash
   python train_rvq_tokenizer.py --hidden_dim 128
   ```

---

### 问题 2: MSE 始终 > 0.01

**症状**:
```
Layers=2:
  MSE: 0.035678 (太高)
```

**解决方案**:
1. **训练更长时间**:
   ```bash
   python train_rvq_tokenizer.py --epochs 200
   ```
2. **增加数据量**:
   ```bash
   python train_rvq_tokenizer.py --num_episodes 100
   ```
3. **增加 codebook size**:
   ```bash
   python train_rvq_tokenizer.py --codebook_size 512
   ```
4. **检查数据分布**: 可能某些任务特别复杂，需要更多层

---

### 问题 3: GPU 内存不足

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 减少 batch size
python train_rvq_tokenizer.py --batch_size 16

# 减少模型大小
python train_rvq_tokenizer.py --hidden_dim 32 --batch_size 16

# 使用 CPU
python train_rvq_tokenizer.py --device cpu --num_episodes 20
```

---

### 问题 4: 测试时模型加载失败

**症状**:
```
FileNotFoundError: rvq_tokenizer.pt not found
```

**解决方案**:
```bash
# 检查模型是否训练完成
ls -lh rvq_tokenizer.pt

# 如果不存在，重新训练
python train_rvq_tokenizer.py --num_episodes 50 --epochs 100

# 指定正确的模型路径
python analyze_rvq_compression.py --model /path/to/rvq_tokenizer.pt
```

---

## 📝 核心文件说明

### 1. `rvq_tokenizer.py`
**功能**: RVQ tokenizer 实现

**关键类**:
- `VectorQuantizer`: 单层 VQ，使用 EMA 更新 codebook
- `ResidualVectorQuantizer`: 多层 RVQ，残差累积
- `RVQTokenizer`: 完整模型（encoder + RVQ + decoder）

**配置参数**:
```python
RVQTokenizer(
    action_dim=7,           # 动作维度
    chunk_size=10,          # 动作块大小
    num_layers=8,           # RVQ 层数 (核心参数)
    hidden_dim=64,          # 隐藏层维度
    num_embeddings=256,     # Codebook 大小
    commitment_cost=0.25,   # Commitment loss 权重
)
```

---

### 2. `train_rvq_tokenizer.py`
**功能**: 训练脚本

**主要步骤**:
1. 收集 π0.5 动作（复用 `collect_pi05_actions()`）
2. 创建 RVQTokenizer
3. 训练循环（reconstruction loss + VQ loss）
4. 保存模型

**命令行参数**:
```bash
--task_suite      # LIBERO 任务套件
--num_episodes    # 收集多少 episodes
--num_layers      # RVQ 层数 (default: 8)
--hidden_dim      # 隐藏维度 (default: 64)
--codebook_size   # Codebook 大小 (default: 256)
--epochs          # 训练 epochs (default: 100)
--batch_size      # Batch size (default: 32)
--lr              # Learning rate (default: 1e-3)
--residual_dropout # Residual dropout (default: 0.1)
--device          # cuda or cpu
--output          # 输出模型路径
```

---

### 3. `analyze_rvq_compression.py`
**功能**: 分析脚本，类似 `analyze_libero_actions.py`

**主要步骤**:
1. 加载训练好的 RVQ tokenizer
2. 收集测试动作
3. 测试不同层数（1-8）的重建误差
4. 生成图表（MSE vs. Layers）

**输出**:
- `rvq_compression_analysis.png`: 压缩分析图
- 终端输出：每层的 MSE、压缩比、假设验证结果

---

### 4. `test_rvq_tokenizer.py`
**功能**: 单元测试

**包含测试**:
- `test_vector_quantizer()`: 测试单层 VQ
- `test_residual_vector_quantizer()`: 测试多层 RVQ
- `test_rvq_tokenizer_basic()`: 测试编码解码
- `test_rvq_tokenizer_layers()`: 测试不同层数
- `test_realistic_smooth_actions()`: 测试平滑轨迹

---

## 🎓 理解 RVQ

### RVQ vs. DCT

**DCT (Discrete Cosine Transform)**:
- 固定变换基（余弦函数）
- 不需要训练
- 适用于所有平滑信号

**RVQ (Residual Vector Quantization)**:
- 学习的 codebook（从数据中学习）
- 需要训练
- 能捕捉数据特有的模式

### RVQ 如何工作

```python
# Layer 1: 编码主要信息
quantized_1 = VQ_1(input)
residual_1 = input - quantized_1

# Layer 2: 编码残差
quantized_2 = VQ_2(residual_1)
residual_2 = residual_1 - quantized_2

# Layer 3: 继续编码残差
quantized_3 = VQ_3(residual_2)
...

# 最终重建
reconstructed = quantized_1 + quantized_2 + quantized_3 + ...
```

### 为什么是层级的？

- **Layer 1-2**: Codebook 学习主要运动模式（向前、向后、抓取）
- **Layer 3-4**: Codebook 学习修正模式（微调位置）
- **Layer 5-8**: Codebook 学习细节（抖动、接触力）

---

## ✅ Phase 1 完成标准

### 必须达到的指标

1. ✅ **训练成功**:
   - Reconstruction loss 收敛
   - VQ loss 稳定

2. ✅ **重建质量**:
   - Layer 1-2: MSE < 0.01 (可接受)
   - Layer 1-8: MSE < 0.001 (优秀)

3. ✅ **假设验证**:
   - 前几层捕捉粗略运动
   - 深层捕捉精细修正

### 可选的额外验证

1. **可视化**: 绘制原始动作 vs. 重建动作
2. **Codebook 分析**: 统计每个 code 的使用频率
3. **任务阶段分析**: 不同阶段（reach, grasp, place）需要多少层

---

## 📚 相关概念

### VQ-VAE (Vector Quantized Variational Autoencoder)
- **论文**: [Neural Discrete Representation Learning (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- **核心思想**: 用离散 codebook 替代连续潜在空间

### RVQ (Residual Vector Quantization)
- **论文**: [SoundStream (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312)
- **核心思想**: 多层 VQ，每层量化前一层的残差

### 机器人学中的 VQ
- **RDT-2**: 使用 RVQ 编码动作
- **VQ-VLA**: 视觉-语言-动作模型 + VQ

---

## 🎯 Phase 2 预览

如果 Phase 1 成功（MSE 达标），接下来：

1. **训练 VLA Policy**:
   - 输入：Image + Language + State
   - 输出：RVQ tokens (1-8 层)

2. **实现自适应推理**:
   - Monitor: 检测任务复杂度（熵值、光流、距离）
   - Draft: 用 Layer 1-2 快速预测
   - Refine: 复杂时激活 Layer 3-8

3. **Benchmark**:
   - 对比 Dense RVQ baseline
   - 测量加速比和成功率

---

## 📞 获取帮助

如果遇到问题：

1. **检查单元测试**:
   ```bash
   python test_rvq_tokenizer.py
   ```

2. **查看训练曲线**:
   - 打开 `training_history.png`
   - 检查 loss 是否收敛

3. **减小问题规模**:
   ```bash
   # 用更少的数据快速验证
   python train_rvq_tokenizer.py --num_episodes 10 --epochs 20
   ```

4. **对比 DCT 结果**:
   - DCT 已经证明动作是可压缩的
   - RVQ 应该能达到类似或更好的结果

---

## 📄 引用

如果使用这个代码，请引用：

```bibtex
@misc{rvq_tokenizer_2025,
  title={RVQ Tokenizer for Robot Action Compression},
  author={Your Name},
  year={2025},
  note={Phase 1 of Residual Speculative Decoding for VLA}
}
```

---

**预期时间线**:
- 单元测试: 5 分钟
- 训练: 30-60 分钟
- 分析: 10 分钟
- **总计: 1-2 小时**

**成功标志**: 生成的 `rvq_compression_analysis.png` 显示 Layer 1-2 的 MSE < 0.01

**下一步**: Phase 2 - 训练 VLA Policy 预测 RVQ tokens

---

🚀 **开始 Phase 1 吧！祝实验顺利！**
