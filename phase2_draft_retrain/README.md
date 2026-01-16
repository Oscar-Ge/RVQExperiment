# Draft Model Retraining with Projection Layer

## 📋 任务概述

**目标**：重新训练Draft Model，使其能够正确处理OpenVLA的4096维输出

**问题诊断**：
- ❌ 当前Draft Model基于`hidden_dim=512`训练
- ❌ 训练时使用**模拟的Embedding Extractor**，而非真实OpenVLA
- ❌ 无法直接处理OpenVLA的4096维输出
- ❌ 在Phase 3中会收到随机噪声（因为projection是随机初始化的）

**解决方案**：Plan B (Retrain with Projection)
1. ✅ 在Draft Model入口增加`Linear(4096, 512)` projection层
2. ✅ 使用**冻结的真实OpenVLA**提取特征
3. ✅ 重新进行Phase 2 (Day 5-6) 的训练
4. ✅ 训练projection层 + Draft Model的所有参数

**预期收益**：
- ✅ 维度匹配：能处理OpenVLA的4096维输出
- ✅ 语义正确：学到真实OpenVLA的特征分布
- ✅ 保持轻量：Draft仍然是512维（4.7M参数）
- ✅ 推理加速：45-55ms（相比baseline的70ms）

---

## 📁 文件结构

```
phase2_draft_retrain/
├── README.md                          # 本文件：任务概述
├── TRAINING_PLAN.md                   # 详细训练方案
├── modal_train_draft_with_projection.py  # 训练脚本（Modal版本）
├── INTEGRATION_GUIDE.md               # 与Phase 3的集成指南
├── DATA_FLOW.md                       # 数据流说明
└── TESTING_CHECKLIST.md               # 测试清单
```

---

## 🎯 快速开始（For Agent）

### Step 1: 理解问题

阅读`TRAINING_PLAN.md`了解：
- 为什么需要重新训练
- 新架构的数据流
- 训练配置

### Step 2: 准备环境

确保可以访问：
- ✅ Modal GPU资源（A100）
- ✅ OpenVLA fine-tuned模型（`moojink/openvla-7b-oft-finetuned-libero-spatial`）
- ✅ Phase 1的RFSQ Decoder（`/models/rfsq_best.pt`）
- ✅ LIBERO数据集

### Step 3: 运行训练

```bash
# 启动训练
modal run modal_train_draft_with_projection.py \
    --num-episodes 200 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4

# 预计时间：3-4小时（A100）
```

### Step 4: 验证训练结果

检查：
- ✅ Coarse layer accuracy > 85%
- ✅ Checkpoint包含projection weights
- ✅ Draft Model可以接受4096维输入

### Step 5: 集成到Phase 3

参考`INTEGRATION_GUIDE.md`：
- 加载新的Draft Model checkpoint
- 验证projection layer正确工作
- 测试Speculative Decoding加速效果

---

## 📊 预期结果

### 训练指标

| Metric | Target | 说明 |
|--------|--------|------|
| Coarse Layer Accuracy (L0-L2) | >85% | 前3层预测准确率 |
| Training Time | 3-4 hours | A100 GPU |
| Model Size | 4.7M params | Draft + Projection |
| Checkpoint Size | ~20MB | 完整模型 |

### Phase 3集成后

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Success Rate | 0% | 85-95% | ✅ 可用 |
| Inference Time | N/A | 45-55ms | ✅ 1.3-1.6x faster |
| Draft Acceptance | N/A | 60-80% | ✅ 有效加速 |

---

## 🚨 关键差异（vs 原始Phase 2训练）

| 方面 | 原始Phase 2 | 新训练 |
|------|------------|--------|
| **Feature Extractor** | 模拟的random embeddings | **真实OpenVLA（frozen）** |
| **Hidden Dim** | 直接512 | **4096 → projection → 512** |
| **Projection Layer** | ❌ 无 | ✅ **训练Linear(4096, 512)** |
| **训练数据** | 可能质量不高 | 真实OpenVLA特征 |
| **可用性** | ❌ 无法用于Phase 3 | ✅ 直接集成 |

---

## 🔗 相关文档

- **Phase 3问题诊断**：`../phase3/CRITICAL_FIX.md`
- **Phase 3核心实现**：`../phase3/rsd_engine_core.py`
- **原始Phase 2训练**：查看之前的训练logs和代码

---

## ✅ 成功标准

训练完成后，你应该能够：

1. **加载模型**：
   ```python
   draft_model = RFSQDraftModelWithProjection(...)
   checkpoint = torch.load('best_draft_with_projection.pt')
   draft_model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **前向传播**：
   ```python
   # 输入：真实OpenVLA的4096维hidden state
   hidden_4096 = torch.randn(1, 4096)

   # 输出：前3层的RFSQ token预测
   logits = draft_model(hidden_4096)  # [1, 3, 8*16, 7]
   ```

3. **在Phase 3中使用**：
   ```python
   engine = RSDInferenceEngine(
       main_model=main_model,
       draft_model=draft_model,  # ✅ 新训练的
       rfsq_head=rfsq_head,
       rfsq_decoder=rfsq_decoder,
       processor=processor,
       device=device,
   )

   # 不再需要随机初始化的projection！
   ```

---

## 📝 下一步

1. **立即**：阅读`TRAINING_PLAN.md`了解详细方案
2. **然后**：查看`modal_train_draft_with_projection.py`了解实现
3. **准备好后**：运行训练
4. **训练完成**：参考`INTEGRATION_GUIDE.md`集成到Phase 3

---

## 🎉 为什么这样做

**短期**：
- 解决Phase 3的projection随机初始化问题
- 使Speculative Decoding真正工作

**长期**：
- 建立正确的训练流程
- 为未来的改进奠定基础
- 确保实验的可复现性

**替代方案对比**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| A. 禁用Draft | 简单安全 | 无加速 | 🟡 临时方案 |
| **B. 重新训练** | **正确且完整** | **需要3-4小时** | ✅ **推荐** |
| C. 修改架构为4096 | 避免projection | Draft太大，慢 | ❌ 不推荐 |

---

**准备好了吗？开始重新训练吧！🚀**
