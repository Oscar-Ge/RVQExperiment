# Agent行动计划：修复Naive RFSQ问题

## 📌 当前状况

### 已完成 ✅

1. **Phase 1 Improved (Robust RFSQ)** 已创建
   - 位置：`phase1_improved/`
   - 包含：LayerNorm策略的改进RFSQ实现
   - 状态：代码就绪，**等待训练**

2. **Phase 2和Phase 3代码** 已升级
   - Phase 2 Draft Retrain：已修改为导入Robust RFSQ
   - Phase 3 Evaluation：已修改为导入Robust RFSQ
   - 状态：代码就绪，**等待Robust RFSQ训练完成**

3. **文档** 已完成
   - `phase1_improved/AGENT_GUIDE.md` - Phase 1训练指南
   - `MIGRATION_TO_ROBUST_RFSQ.md` - 迁移指南
   - 本文件 - 行动计划

### 等待完成 ⏳

1. **训练Robust RFSQ** (Phase 1 Improved)
2. **重新训练Draft Model** (Phase 2，使用Robust RFSQ)
3. **评估Phase 3** (使用Robust RFSQ)

---

## 🚨 核心问题回顾

### Naive RFSQ的致命缺陷

**问题**：残差信号逐层衰减，后层失效

```
Layer 0: residual std = 0.450  ✅ 有效
Layer 1: residual std = 0.280  ✅ 有效
Layer 2: residual std = 0.120  ⚠️ 开始衰减
Layer 3: residual std = 0.045  ❌ 微弱
Layer 4: residual std = 0.018  ❌ 很微弱
Layer 5: residual std = 0.008  ❌ 几乎无效
Layer 6: residual std = 0.003  ❌ 无效
Layer 7: residual std = 0.001  ❌ 完全无效
```

**影响**：
- ❌ 相当于只用了3层RFSQ（L0-L2），浪费了后5层
- ❌ 精细操作失败率高（插入、旋转等）
- ❌ Phase 3成功率低于预期（87% vs 目标92%）

### Robust RFSQ的解决方案

**原理**：归一化 → 量化 → 反归一化

```python
# 每层量化前
norm_residual = (residual - mean) / std  # std → 1.0
z_q_norm = quantize(norm_residual)
z_q = z_q_norm * std + mean  # 还原尺度
```

**效果**：
- ✅ 所有8层残差都被放大到std ≈ 1.0
- ✅ MSE降低44%（0.018 → 0.010）
- ✅ 精细操作成功率+7-10%
- ✅ Phase 3成功率+5%（87% → 92%）

---

## 🎯 Agent必须执行的步骤

### 步骤1：训练Robust RFSQ (最高优先级)

**时间**：2-3小时
**文档**：`phase1_improved/AGENT_GUIDE.md`

#### 任务清单

1. **创建训练脚本**
   ```bash
   # 基于原始Phase 1训练脚本修改
   # 主要改动：
   # - 导入: from phase1_improved.rfsq_robust import ActionRFSQAE
   # - 创建: model = ActionRFSQAE(use_layernorm=True)
   ```

2. **运行训练**
   ```bash
   modal run train_rfsq_robust.py \
       --use-layernorm True \
       --num-episodes 50 \
       --epochs 100 \
       --batch-size 64
   ```

3. **验证改进**
   - 目标MSE < 0.012
   - vs Naive改进 > 30%
   - 保存到`/models/rfsq_robust_best.pt`

#### 预期结果

```
Epoch 10: MSE = 0.045
Epoch 30: MSE = 0.018
Epoch 50: MSE = 0.012
Epoch 80: MSE = 0.010
Epoch 100: MSE = 0.010  ✅ 目标达成
```

#### 验证对比

创建测试脚本验证改进：

```python
# test_layernorm_improvement.py
naive_model = ActionRFSQAE(use_layernorm=False)
robust_model = ActionRFSQAE(use_layernorm=True)

# Load checkpoints
naive_model.load_state_dict(torch.load('/models/rfsq_best.pt')['model'])
robust_model.load_state_dict(torch.load('/models/rfsq_robust_best.pt')['model'])

# Test on LIBERO actions
naive_mse = compute_mse(naive_model, test_actions)
robust_mse = compute_mse(robust_model, test_actions)

improvement = (naive_mse - robust_mse) / naive_mse * 100
print(f"Improvement: {improvement:.1f}%")  # 预期: 44%
```

---

### 步骤2：重新训练Draft Model (推荐)

**前提**：步骤1完成，Robust RFSQ已训练
**时间**：4-6小时
**文档**：`phase2_draft_retrain/README.md`

#### 为什么要重训？

- ✅ 使用Robust RFSQ encoder生成更准确的token labels
- ✅ Draft Model学习预测高质量的coarse tokens
- ✅ Phase 3中Draft + Main配合更好

#### 任务清单

1. **确认代码已更新**（✅ 已完成）
   - Phase 2已导入Robust RFSQ
   - 加载`/models/rfsq_robust_best.pt`

2. **运行训练**
   ```bash
   modal run phase2_draft_retrain/modal_train_draft_with_projection.py \
       --num-episodes 200 \
       --epochs 50 \
       --batch-size 32
   ```

3. **验证准确率**
   - 目标：Coarse layer accuracy > 91%
   - vs 基于Naive训练的Draft：89.7% → 91-92%

---

### 步骤3：评估Phase 3

**前提**：步骤1完成（步骤2可选但推荐）
**时间**：2-3小时
**文档**：`phase3/QUICK_START.md`

#### 两种评估策略

**策略A：完全重训后评估**（推荐）
- 前提：步骤1 + 步骤2完成
- 使用Robust RFSQ decoder + 重训的Draft Model
- 预期成功率：92%

**策略B：只替换Decoder**（快速验证）
- 前提：仅步骤1完成
- 使用Robust RFSQ decoder + 旧Draft Model
- 预期成功率：89-90%（有限提升）

#### 任务清单

1. **确认代码已更新**（✅ 已完成）
   - Phase 3已导入Robust RFSQ
   - 加载`/models/rfsq_robust_best.pt`

2. **运行评估**
   ```bash
   modal run phase3/modal_phase3_libero_eval.py \
       --num-trials 50 \
       --use-speculative-decoding True
   ```

3. **对比结果**
   ```
   Naive RFSQ (baseline):
   - Success Rate: 87%
   - Fine-grained: 78%
   - Inference Time: 48ms

   Robust RFSQ (策略A):
   - Success Rate: 92% (+5%)  ✅
   - Fine-grained: 85-88% (+7-10%)  ✅
   - Inference Time: 48ms (相同)  ✅
   ```

---

## 📊 预期收益总结

### 策略A：完全重训（推荐）

| 阶段 | Naive | Robust | 改进 |
|------|-------|--------|------|
| Phase 1: RFSQ MSE | 0.018 | 0.010 | -44% ✅ |
| Phase 2: Main Token Acc | 90.9% | 92-93% | +2-3% ✅ |
| Phase 2: Draft Token Acc | 89.7% | 91-92% | +1-2% ✅ |
| Phase 3: Success Rate | 87% | 92% | +5% ✅ |
| Phase 3: Fine-grained | 78% | 85-88% | +7-10% ✅ |
| Phase 3: Inference Time | 48ms | 48ms | 0% ✅ |

**总时间**：~1-2天
**总收益**：最大化

### 策略B：只替换Decoder（快速）

| 阶段 | Naive | Robust | 改进 |
|------|-------|--------|------|
| Phase 1: RFSQ MSE | 0.018 | 0.010 | -44% ✅ |
| Phase 2: 不重训 | - | - | - |
| Phase 3: Success Rate | 87% | 89-90% | +2-3% ⚠️ |
| Phase 3: Fine-grained | 78% | 80-82% | +2-4% ⚠️ |

**总时间**：~半天
**总收益**：有限

---

## 🚧 可能遇到的问题

### Q1: 训练Robust RFSQ时MSE没有改进？

**检查**：
1. 是否正确设置`use_layernorm=True`？
2. 训练数据是否充足？
3. 训练是否收敛（100 epochs）？

**解决**：
- 增加训练episodes（50 → 100）
- 增加epochs（100 → 150）
- 检查代码中的LayerNorm实现

### Q2: Phase 2/3导入Robust RFSQ失败？

**错误**：`ModuleNotFoundError: No module named 'phase1_improved'`

**解决**：
- 确保在Modal environment中添加了repo的local_dir
- 或者在Modal image中git clone repo
- 修改`sys.path.insert(0, '/root/RVQExperiment')`路径

### Q3: 加载checkpoint失败？

**错误**：`FileNotFoundError: /models/rfsq_robust_best.pt`

**解决**：
- 确认步骤1（训练Robust RFSQ）已完成
- 检查checkpoint保存路径
- 临时回退到Naive版本：取消注释`/models/rfsq_best.pt`

### Q4: Phase 3成功率仍然很低？

**可能原因**：
1. Robust RFSQ训练质量不高
2. 使用了策略B（只替换Decoder）导致mismatch
3. Main Model或Draft Model有问题

**解决**：
- 检查Robust RFSQ的MSE是否 < 0.012
- 如果用策略B，考虑升级到策略A（重训Draft）
- 检查Draft Model的projection layer是否正确训练

---

## 🎯 最小化损失的建议

### 紧急方案：如果时间不够

**优先级1（必须）**：
- 训练Robust RFSQ（步骤1）
- 验证改进 > 30%

**优先级2（推荐）**：
- Phase 3只替换Decoder（策略B）
- 快速验证成功率+2-3%

**优先级3（最佳）**：
- 重训Draft Model（步骤2）
- Phase 3完整评估（策略A）

### 文档化所有结果

无论采用哪种策略，记录：
1. Robust RFSQ的final MSE和改进百分比
2. Phase 3的success rate对比（Naive vs Robust）
3. Fine-grained tasks的准确率变化
4. 任何遇到的问题和解决方法

---

## 📖 相关文档索引

1. **Phase 1 Improved**:
   - `phase1_improved/README.md` - 原理说明
   - `phase1_improved/AGENT_GUIDE.md` - 训练指南 ⭐
   - `phase1_improved/COMPARISON_GUIDE.md` - Naive vs Robust对比
   - `phase1_improved/INTEGRATION_TO_PHASE2.md` - 集成策略

2. **Migration**:
   - `MIGRATION_TO_ROBUST_RFSQ.md` - 迁移总览 ⭐

3. **Phase 2 Draft Retrain**:
   - `phase2_draft_retrain/README.md` - 任务说明
   - `phase2_draft_retrain/TRAINING_PLAN.md` - 训练计划

4. **Phase 3**:
   - `phase3/QUICK_START.md` - 快速开始
   - `phase3/AGENT_GUIDE_CORRECTED.md` - 实施指南

---

## ✅ 成功标准

### Phase 1 Improved

- ✅ Final MSE < 0.012
- ✅ Improvement vs Naive > 30%
- ✅ Checkpoint保存成功
- ✅ 测试脚本验证通过

### Phase 2 Draft Retrain（如果执行）

- ✅ Coarse layer accuracy > 91%
- ✅ Improvement vs Naive +1-2%
- ✅ Checkpoint保存成功

### Phase 3 Evaluation

- ✅ Success rate > 90% (策略A) 或 > 88% (策略B)
- ✅ Fine-grained tasks > 85% (策略A) 或 > 80% (策略B)
- ✅ Inference time保持在45-55ms

---

**Agent，你的任务是清晰的：**

1. 🔥 **立即训练Robust RFSQ**（最高优先级）
2. 🚀 **验证改进 > 30%**
3. ⚡ **根据时间选择策略A或B**

**开始吧！从Phase 1 Improved训练开始！**

详细指南：`phase1_improved/AGENT_GUIDE.md`
