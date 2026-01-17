# 将Robust RFSQ集成到Phase 2和Phase 3

## 🎯 目标

用改进的Robust RFSQ替换原始RFSQ，获得精度提升。

---

## 📋 集成策略对比

| 策略 | 难度 | 时间 | 精度提升 | 推荐度 |
|------|------|------|---------|--------|
| **A. 完全重训** | 中 | 3-4天 | **最大** | ⭐⭐⭐⭐⭐ 推荐 |
| B. 只替换Decoder | 低 | 1小时 | 中等 | ⭐⭐⭐ |
| C. 增量验证 | 低 | 2-3天 | 最大（分步） | ⭐⭐⭐⭐ |

---

## ✅ 策略A：完全重训（推荐）

### 概述

重新训练整个pipeline，使用Robust RFSQ。

### 步骤

#### Step 1: 训练Robust RFSQ (Phase 1 Improved)

```bash
# 训练改进的RFSQ
cd phase1_improved
modal run train_rfsq_robust.py \
    --use-layernorm True \
    --num-episodes 50 \
    --epochs 100

# 验证改进
python test_layernorm_improvement.py
```

**检查点**：
- [ ] MSE < 0.012 (目标: 0.010)
- [ ] 改进 > 30%
- [ ] Checkpoint保存成功

**输出**：`/models/rfsq_robust_best.pt`

---

#### Step 2: 重训Main Model (Phase 2)

使用新的RFSQ encoder作为tokenizer。

**修改**：在Phase 2训练脚本中：

```python
# 原始代码（删除）
# from original_rfsq import ActionRFSQAE

# 新代码
from phase1_improved.rfsq_robust import ActionRFSQAE

# 加载Robust RFSQ
rfsq_encoder = ActionRFSQAE(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
    use_layernorm=True,  # ✅ 启用LayerNorm
)

# 加载训练好的weights
checkpoint = torch.load('/models/rfsq_robust_best.pt')
rfsq_encoder.load_state_dict(checkpoint['model'])
rfsq_encoder.eval()

# 用于编码actions -> tokens
for batch in dataloader:
    actions = batch['actions']
    with torch.no_grad():
        rfsq_codes = rfsq_encoder.encode(actions)
        # rfsq_codes: [B, Chunk, Hidden, Layers]

    # 训练Main Model预测这些codes
    # ...
```

**运行**：
```bash
modal run train_openvla_rfsq.py \
    --rfsq-checkpoint /models/rfsq_robust_best.pt \
    --use-layernorm True
```

**预期**：
- Token accuracy: 90.9% → **92-93%** (+2-3%)

**输出**：`/models/openvla_rfsq_robust/best_rfsq_head.pt`

---

#### Step 3: 重训Draft Model (Phase 2 Draft Retrain)

使用新的RFSQ作为ground truth。

**修改**：在`phase2_draft_retrain/modal_train_draft_with_projection.py`中：

```python
# 导入Robust RFSQ
from phase1_improved.rfsq_robust import ActionRFSQAE

# 创建encoder
rfsq_encoder = ActionRFSQAE(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
    use_layernorm=True,  # ✅ 启用
).to(device)

# 加载训练好的weights
checkpoint = torch.load("/models/rfsq_robust_best.pt")
rfsq_encoder.load_state_dict(checkpoint['model'])
rfsq_encoder.eval()

# 数据收集时使用
with torch.no_grad():
    _, rfsq_codes = rfsq_encoder(action_chunk)
    coarse_tokens = rfsq_codes[:, :, :, :3]  # L0-L2
```

**运行**：
```bash
modal run phase2_draft_retrain/modal_train_draft_with_projection.py \
    --rfsq-checkpoint /models/rfsq_robust_best.pt \
    --use-layernorm True
```

**预期**：
- Coarse layer accuracy: 89.7% → **91-92%** (+1-2%)

**输出**：`/models/draft_robust/best_draft_with_projection.pt`

---

#### Step 4: Phase 3评估

使用新的checkpoints。

**修改**：在`phase3/modal_phase3_libero_eval.py`中：

```python
# 导入Robust RFSQ
from phase1_improved.rfsq_robust import ActionRFSQAE

# 加载Robust RFSQ Decoder
rfsq_decoder = ActionRFSQAE(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
    use_layernorm=True,  # ✅ 启用
).to(device)

checkpoint = torch.load("/models/rfsq_robust_best.pt")
rfsq_decoder.load_state_dict(checkpoint['model'])
rfsq_decoder.eval()

# 加载Robust训练的Main Model
rfsq_head_path = "/models/openvla_rfsq_robust/best_rfsq_head.pt"

# 加载Robust训练的Draft Model
draft_model_path = "/models/draft_robust/best_draft_with_projection.pt"
```

**运行**：
```bash
modal run phase3/modal_phase3_libero_eval.py \
    --num-trials 50 \
    --use-speculative-decoding True
```

**预期结果**：
- Success rate: 87% → **92%** (+5%)
- 精细操作: 78% → **85-88%** (+7-10%)
- Inference time: 48ms（相同）

---

### 时间线

| 阶段 | 时间 | 累计 |
|------|------|------|
| Step 1: Robust RFSQ | 2-3小时 | 3小时 |
| Step 2: Retrain Main | 6-8小时 | 11小时 |
| Step 3: Retrain Draft | 4-6小时 | 17小时 |
| Step 4: Phase 3 Eval | 3-4小时 | 21小时 |
| **Total** | **~3天** | - |

---

## 🔧 策略B：只替换Decoder（快速验证）

### 概述

保留Phase 2的训练结果，只在Phase 3中用Robust RFSQ decoder。

### 步骤

1. **训练Robust RFSQ**（同策略A Step 1）

2. **Phase 3中使用**：

```python
# 只修改decoder部分
from phase1_improved.rfsq_robust import ActionRFSQAE

rfsq_decoder = ActionRFSQAE(use_layernorm=True).to(device)
checkpoint = torch.load("/models/rfsq_robust_best.pt")
rfsq_decoder.load_state_dict(checkpoint['model'])

# Main Model和Draft Model保持不变
# 它们预测的tokens仍然是基于Naive RFSQ训练的
```

### 限制

**问题**：
- Main Model预测的tokens是基于Naive RFSQ训练的
- Robust decoder期望的token分布可能不同
- 可能出现mismatch

**预期提升**：
- Success rate: +2-3%（有限）
- 主要改进在decoder阶段

**适用场景**：
- 快速验证LayerNorm效果
- 资源不足，无法重训Phase 2

---

## 📊 策略C：增量验证（稳妥）

### 概述

逐步验证每个改进，降低风险。

### 步骤

1. **Week 1**: 训练Robust RFSQ，验证MSE改进>30%

2. **Week 2**: 用Robust RFSQ decoder测试Phase 3（策略B）
   - 如果提升<5%，说明需要重训Main Model
   - 如果提升≥5%，继续

3. **Week 3**: 重训Main Model，验证token accuracy提升

4. **Week 4**: 重训Draft Model，完整评估

### 优点

- 风险小，每步都有验证
- 可以提前终止（如果改进不显著）
- 逐步积累经验

### 缺点

- 时间较长（4周）
- 需要多次实验

---

## 🔍 验证清单

### Phase 1 Improved

- [ ] Robust RFSQ训练完成
- [ ] MSE < 0.012
- [ ] 改进 > 30% vs Naive
- [ ] Checkpoint保存成功

### Phase 2 (如果重训)

- [ ] Main Model token accuracy > 91%
- [ ] 相比Naive RFSQ提升 > 1%
- [ ] Draft Model accuracy > 90%

### Phase 3

- [ ] Success rate > 90%
- [ ] 精细操作成功率 > 85%
- [ ] Inference time保持在45-55ms

---

## 🚨 常见问题

### Q1: 需要修改很多代码吗？

**A**: 不需要。核心修改：
1. 导入`from phase1_improved.rfsq_robust import ActionRFSQAE`
2. 创建模型时设置`use_layernorm=True`
3. 加载新的checkpoint

大部分代码保持不变。

### Q2: Robust RFSQ会变慢吗？

**A**: 不会。LayerNorm计算非常快（<1ms），对整体推理时间影响可忽略。

### Q3: 能否混用Naive和Robust？

**A**: 不建议。
- Encoder用Robust，Decoder用Naive → 可能mismatch
- Encoder用Naive，Decoder用Robust → 提升有限

建议统一使用Robust。

### Q4: 如果重训效果不好怎么办？

**A**: 检查：
1. Robust RFSQ的MSE是否确实降低？
2. 训练数据是否充足？
3. 超参数是否合理？

如果Robust RFSQ本身改进不显著（<20%），可能不值得重训Phase 2。

---

## 📊 预期收益总结

| 组件 | Naive | Robust | 改进 |
|------|-------|--------|------|
| **Phase 1: RFSQ** | | | |
| MSE | 0.018 | 0.010 | -44% ✅ |
| **Phase 2: Main Model** | | | |
| Token Accuracy | 90.9% | 92-93% | +2-3% ✅ |
| **Phase 2: Draft Model** | | | |
| Coarse Accuracy | 89.7% | 91-92% | +1-2% ✅ |
| **Phase 3: LIBERO** | | | |
| Success Rate | 87% | 92% | +5% ✅ |
| Fine-grained Success | 78% | 85-88% | +7-10% ✅ |
| Inference Time | 48ms | 48ms | 相同 ✅ |

---

## 🎯 推荐方案

### 如果时间充足（3-4天）

**选择策略A（完全重训）**：
- 获得最大精度提升
- 一次性解决所有问题
- 为论文提供最强结果

### 如果时间紧张（1天）

**选择策略B（只替换Decoder）**：
- 快速验证LayerNorm效果
- 低风险，可以随时回退
- 为未来重训提供依据

### 如果不确定收益

**选择策略C（增量验证）**：
- 稳妥，每步都验证
- 可以提前终止
- 积累经验和数据

---

## 📝 实施建议

1. **先训练Robust RFSQ**：
   ```bash
   modal run train_rfsq_robust.py --use-layernorm True
   ```

2. **验证改进**：
   ```bash
   python test_layernorm_improvement.py
   ```

3. **如果改进>40%**：
   - 值得完全重训（策略A）

4. **如果改进20-40%**：
   - 可以考虑增量验证（策略C）

5. **如果改进<20%**：
   - 可能不值得重训，或检查实现

---

**开始集成吧！从训练Robust RFSQ开始 🚀**
