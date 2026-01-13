# DCT Tokenizer 运行指南

π0-FAST DCT压缩概念验证 - 完整运行指南

## 📋 概述

这是一个**2小时概念验证实验**，用于测试DCT压缩是否能应用于π0.5的动作序列。

**目标**: 验证π0.5生成的动作是否可以用DCT+量化压缩，而不损失任务关键信息。

**不做什么**: 不训练模型，不实现完整BPE，不集成到π0.5策略中。

---

## 🔧 前置要求

### 1. 环境要求
- Python 3.8+
- CUDA GPU（推荐，用于π0.5推理）
- 已经配置好的 `basic-run` 环境

### 2. 必需依赖
```bash
# 基础科学计算
numpy
scipy>=1.2.0  # 需要 scipy.fftpack（DCT/IDCT）
matplotlib    # 用于可视化

# π0.5 相关（应该已经安装）
torch
openpi
openpi_client
libero
robosuite==1.4.0
```

### 3. 验证环境
```bash
# 检查 scipy 是否支持 DCT
python -c "from scipy.fftpack import dct, idct; print('✅ scipy DCT available')"

# 检查 π0.5 是否可用
python basic-run/run_pi05_libero_benchmark_pytorch.py --check-env
```

---

## 🚀 快速开始（推荐顺序）

### Step 1: 单元测试（5秒）

**目的**: 验证DCT tokenizer基本功能

```bash
python test_dct_compression.py
```

**预期输出**:
```
================================================================================
π0-FAST DCT TOKENIZER TEST SUITE
================================================================================
================================================================================
TESTING DCT TOKENIZER - ROUND-TRIP ENCODING
================================================================================

Tokenizer: MinimalDCTTokenizer(action_dim=7, chunk_size=16, num_dct_keep=4, num_bins=256, compression_ratio=4.00x)

Generating dummy dataset for fitting...
Fitted tokenizer on 160 actions
  Action range: [-2.4490, 2.5149]

Testing round-trip encoding...

  Original shape: (16, 7) (112 values)
  Token length: 28 tokens
  Compression ratio: 4.00x
  Reconstructed shape: (16, 7)

  Reconstruction Metrics:
    MSE (Mean Squared Error): 0.001234
    MAE (Mean Absolute Error): 0.028456
    Max Absolute Error: 0.089234

  ✅ PASSED! MSE (0.001234) < threshold (0.1)

...
================================================================================
TEST SUMMARY
================================================================================
✅ All critical tests passed!

Next steps:
  1. Run analyze_libero_actions.py to test on real π0.5 actions
  2. Check if compression works well on actual robot tasks
  3. Analyze MSE across different task phases (reach, grasp, etc.)
================================================================================
```

---

### Step 2: 真实数据分析（10-15分钟）

**目的**: 在真实π0.5动作上测试DCT压缩

```bash
# 基础运行（20 episodes，libero_spatial task 0）
python analyze_libero_actions.py --task_suite libero_spatial --num_episodes 20

# 更多episodes获得更稳定结果
python analyze_libero_actions.py --task_suite libero_spatial --num_episodes 50

# 测试不同任务
python analyze_libero_actions.py --task_suite libero_object --task_id 3 --num_episodes 30
```

**预期输出**:
```
================================================================================
π0-FAST DCT COMPRESSION ANALYSIS
================================================================================
================================================================================
COLLECTING π0.5 ACTIONS ON LIBERO
================================================================================

[1/4] Loading π0.5 policy...
Using pre-converted PyTorch checkpoint: ~/.cache/openpi/converted_checkpoints/pi05_libero_pytorch

Loading pi05-libero policy from: ~/.cache/openpi/...
✓ Policy loaded successfully!

GPU Memory Usage:
  Allocated: 2.45 GB
  Reserved: 2.68 GB

[2/4] Loading LIBERO task: libero_spatial - Task 0
  Task: LIVING_ROOM_SCENE0_put_the_black_bowl_on_top_of_the_cabinet
  Description: put the black bowl on top of the cabinet
  Initial states: 50

[3/4] Creating LIBERO environment...

[4/4] Collecting actions from 20 episodes...
  Episode 1/20: ✅ (steps: 45)
  Episode 2/20: ✅ (steps: 52)
  Episode 3/20: ❌ (steps: 220)
  ...

✅ Collected 1250 action chunks
   Success rate: 65.0%
   Avg episode length: 98.3 steps

================================================================================
ANALYZING COMPRESSION WITH DIFFERENT SETTINGS
================================================================================

DCT keep= 2:
  MSE: 0.023456 ± 0.008923
  Tokens: 14.0
  Compression: 8.00x

DCT keep= 4:
  MSE: 0.004567 ± 0.001234
  Tokens: 28.0
  Compression: 4.00x

DCT keep= 6:
  MSE: 0.001234 ± 0.000456
  Tokens: 42.0
  Compression: 2.67x

...

📊 Plot saved to dct_compression_analysis.png

================================================================================
SUMMARY
================================================================================

✅ Optimal setting: Keep 4 DCT coefficients
   MSE: 0.004567 ± 0.001234
   Compression: 4.00x
   Tokens per chunk: 28.0

🎯 Best compression with excellent MSE (<0.01):
   Keep 4 coefficients
   MSE: 0.004567
   Compression: 4.00x
   → Autoregressive decoding could be 4.0x faster!

================================================================================
NEXT STEPS
================================================================================
✅ DCT compression proof of concept complete!

Potential next steps:
  1. Add BPE (Byte-Pair Encoding) to further compress tokens
  2. Train an autoregressive model to predict these tokens
  3. Implement Residual Speculative Decoding
  4. Analyze task-aware compression (different settings for reach vs grasp)
================================================================================
```

---

## 📊 输出文件

### 1. `dct_compression_analysis.png`
可视化图表，包含：
- **左图**: 重建误差 vs DCT系数数量（对数尺度）
  - 红色虚线: 目标MSE=0.01
  - 绿色虚线: 优秀MSE=0.001
- **右图**: 压缩比 vs DCT系数数量
- 底部: 任务元数据（任务名、episodes数量、成功率等）

---

## 🎯 如何解读结果

### ✅ 成功的标准
1. **MSE < 0.01** 表示重建质量足够好
2. **压缩比 ≥ 4x** 表示显著加速潜力
3. **低标准差** 表示压缩稳定

### 示例场景

#### Scenario A: 理想结果
```
DCT keep=4: MSE=0.003, Compression=4.00x
```
**结论**:
- ✅ 4个DCT系数足够
- ✅ 可以实现4x压缩
- ✅ π0-FAST理论上可行！

#### Scenario B: 需要更多系数
```
DCT keep=4: MSE=0.025 (太高)
DCT keep=6: MSE=0.008, Compression=2.67x
```
**结论**:
- ⚠️ 需要6个系数才能达到目标精度
- ✅ 仍可实现2.67x压缩
- ✅ π0-FAST可行，但加速比稍低

#### Scenario C: 压缩效果不佳
```
DCT keep=12: MSE=0.015 (仍然偏高)
```
**结论**:
- ❌ 动作序列可能不够平滑
- 需要分析具体任务特性
- 可能需要针对不同任务阶段使用不同压缩率

---

## 🔬 高级用法

### 1. 测试多个任务
```bash
# 创建批量测试脚本
for task_id in 0 1 2 3 4; do
    python analyze_libero_actions.py \
        --task_suite libero_spatial \
        --task_id $task_id \
        --num_episodes 20 \
        --output "results/task_${task_id}_compression.png"
done
```

### 2. 使用CPU（如果没有GPU）
```bash
python analyze_libero_actions.py \
    --task_suite libero_spatial \
    --num_episodes 10 \
    --device cpu
```
**注意**: CPU推理会很慢（约10-20倍），建议减少episodes数量。

### 3. 分析特定任务套件
```bash
# Spatial reasoning tasks（推荐开始）
python analyze_libero_actions.py --task_suite libero_spatial --num_episodes 30

# Object manipulation tasks（更复杂）
python analyze_libero_actions.py --task_suite libero_object --num_episodes 30

# Goal-oriented tasks
python analyze_libero_actions.py --task_suite libero_goal --num_episodes 30

# Long-horizon tasks（LIBERO-10，更复杂）
python analyze_libero_actions.py --task_suite libero_10 --task_id 0 --num_episodes 20
```

---

## 🐛 故障排除

### 问题1: `ModuleNotFoundError: No module named 'scipy.fftpack'`
**解决方案**:
```bash
pip install scipy>=1.2.0
```

### 问题2: LIBERO导入失败
**解决方案**:
```bash
# 检查LIBERO安装
pip install libero

# 检查robosuite版本
pip install 'robosuite==1.4.0'
```

### 问题3: π0.5 checkpoint未找到
**解决方案**:
```bash
# 确保已经运行过basic-run中的脚本
cd basic-run
python run_pi05_libero_benchmark_pytorch.py --task_suite libero_spatial --num_episodes 1

# 这会自动下载并转换checkpoint
```

### 问题4: GPU内存不足
**解决方案**:
```bash
# 1. 减少episodes数量
python analyze_libero_actions.py --num_episodes 10

# 2. 使用CPU（慢）
python analyze_libero_actions.py --device cpu --num_episodes 5
```

### 问题5: 结果图表不显示中文
**解决方案**:
图表标题和标签都使用英文，应该没有中文显示问题。如果有问题，检查matplotlib配置。

---

## 📈 预期运行时间

| 步骤 | Episodes | 预期时间 | 输出 |
|------|----------|----------|------|
| `test_dct_compression.py` | N/A | 5秒 | 终端输出 |
| `analyze_libero_actions.py` | 10 | 5-8分钟 | PNG图表 |
| `analyze_libero_actions.py` | 20 | 10-15分钟 | PNG图表 |
| `analyze_libero_actions.py` | 50 | 25-35分钟 | PNG图表 |

**注意**:
- 首次运行需要下载π0.5 checkpoint（约2-3 GB），额外增加5-10分钟
- GPU环境下的时间估计，CPU会慢10-20倍

---

## 📝 代码文件说明

### 1. `minimal_dct_tokenizer.py`
**核心类**: `MinimalDCTTokenizer`

**主要方法**:
- `fit(actions_dataset)`: 从数据集计算归一化统计量
- `encode(actions)`: 将 [16, 7] 动作编码为离散tokens
- `decode(tokens)`: 将tokens解码回 [16, 7] 动作
- `get_compression_ratio()`: 返回压缩比

**配置参数**:
- `action_dim=7`: 动作维度（LIBERO固定为7）
- `chunk_size=16`: 动作块大小（π0.5固定为16）
- `num_dct_keep=4`: 保留的DCT系数数量（**核心参数**）
- `num_bins=256`: 量化bins数量（默认uint8）

### 2. `test_dct_compression.py`
**功能**: 单元测试和基础验证

**包含测试**:
- `test_roundtrip()`: 基础编码-解码测试
- `test_different_compression_ratios()`: 测试不同压缩设置
- `test_with_realistic_actions()`: 用模拟的平滑轨迹测试

### 3. `analyze_libero_actions.py`
**功能**: 主分析脚本，在真实π0.5动作上测试

**主要函数**:
- `collect_pi05_actions()`: 运行π0.5收集动作
- `analyze_compression()`: 测试不同DCT设置
- `plot_results()`: 生成可视化图表

**命令行参数**:
```bash
--task_suite    # LIBERO任务套件（default: libero_spatial）
--task_id       # 任务ID（default: 0）
--num_episodes  # Episodes数量（default: 20）
--device        # PyTorch设备（default: cuda）
--output        # 输出图表路径（default: dct_compression_analysis.png）
```

---

## 🎓 理解DCT压缩

### 为什么DCT能压缩机器人动作？

1. **机器人动作是平滑的**: 相邻时间步的动作变化小
2. **DCT提取低频成分**: 平滑信号的能量集中在低频
3. **高频系数≈0**: 可以丢弃，只保留低频系数
4. **量化为离散值**: 进一步压缩存储

### 类比
- 类似JPEG压缩图像
- 类比MP3压缩音频
- 机器人动作就是"时间域上的平滑信号"

---

## ✅ 实验成功后的下一步

如果MSE < 0.01 且压缩比 ≥ 4x，则证明π0-FAST可行！

### 后续研究方向

1. **添加BPE**: 进一步压缩DCT tokens
2. **训练Token预测器**: 用小型自回归模型预测tokens
3. **实现Residual Speculative Decoding**:
   - Draft model: Token predictor
   - Verification: π0.5 full model
4. **任务感知压缩**:
   - 抓取阶段用更多系数（高精度）
   - 移动阶段用更少系数（高压缩）

---

## 📞 问题反馈

如果遇到问题：
1. 检查本文档的"故障排除"部分
2. 确认环境配置（运行 `test_dct_compression.py`）
3. 查看 `basic-run/QUICK_START.md` 确认π0.5环境正常

---

## 📄 引用

本实验基于以下工作：
- **π0.5**: Physical Intelligence的视觉-语言-动作模型
- **LIBERO**: 长时域机器人操作benchmark
- **DCT**: 离散余弦变换（广泛用于信号压缩）

---

**祝实验顺利！** 🚀

预期结果：证明DCT可以4x压缩π0.5动作，为π0-FAST奠定理论基础。
