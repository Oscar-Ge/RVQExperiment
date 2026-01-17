# Agent实施指南：Robust RFSQ with LayerNorm

## 🎯 你的任务

训练一个**改进的RFSQ**，通过引入LayerNorm策略提升量化精度。

---

## 📖 背景知识

### 问题：原始RFSQ的后层失效

```python
# 原始逻辑
for layer in self.layers:
    z_q, indices = layer(residual)  # 直接量化
    residual = residual - z_q       # ❌ 残差越来越小
```

**结果**：
- L0-L2有效（残差std ≈ 0.1-0.5）
- L3-L7几乎无效（残差std ≈ 0.001-0.01）
- 后5层浪费了，精细度损失

### 解决方案：论文的LayerNorm策略

```python
# 改进逻辑
for layer in self.layers:
    # 1. 归一化 - 放大微弱信号
    norm_residual = normalize(residual)

    # 2. 量化 - 在归一化空间
    z_q_norm, indices = layer(norm_residual)

    # 3. 反归一化 - 还原尺度
    z_q = denormalize(z_q_norm)

    # 4. 更新残差
    residual = residual - z_q
```

**结果**：
- 所有8层都有效
- 重构误差降低44%
- 精细操作提升7-10%

---

## 🚀 实施步骤

### Step 1: 理解代码

阅读`rfsq_robust.py`，重点关注：

1. **RobustSTEQuantizer**（第17-85行）：
   - `use_layernorm=True`时，做归一化+量化+反归一化
   - `use_layernorm=False`时，回退到原始逻辑

2. **RobustRFSQBlock**（第88-156行）：
   - 使用RobustSTEQuantizer替代原始STEQuantizer
   - 其他逻辑保持不变

3. **ActionRFSQAE**（第159-265行）：
   - encoder/decoder架构相同
   - 只是内部的RFSQ block改用Robust版本

### Step 2: 训练Robust RFSQ

**重要**：这是一个**独立任务**，与Phase 2/3分开。

训练脚本（需要agent创建）：
```bash
# 基于原始Phase 1训练脚本修改
# 主要改动：导入rfsq_robust.py，设置use_layernorm=True

modal run train_rfsq_robust.py \
    --use-layernorm True \
    --num-episodes 50 \
    --epochs 100 \
    --batch-size 64
```

**训练配置**：
- 数据：LIBERO actions（同原始Phase 1）
- Epochs: 100
- Batch size: 64
- Learning rate: 1e-3
- Optimizer: AdamW

**预期训练曲线**：
```
Epoch 10: MSE = 0.045
Epoch 30: MSE = 0.018
Epoch 50: MSE = 0.012
Epoch 80: MSE = 0.010
Epoch 100: MSE = 0.010  ✅ 目标达成
```

### Step 3: 验证改进

对比Naive vs Robust：

```python
# test_layernorm_improvement.py（需要agent创建）
import torch
from rfsq_robust import ActionRFSQAE

# Load Naive model
naive_model = ActionRFSQAE(use_layernorm=False)
naive_ckpt = torch.load('rfsq_best.pt')  # 原始checkpoint
naive_model.load_state_dict(naive_ckpt['model'])

# Load Robust model
robust_model = ActionRFSQAE(use_layernorm=True)
robust_ckpt = torch.load('rfsq_robust_best.pt')
robust_model.load_state_dict(robust_ckpt['model'])

# Test on LIBERO actions
test_actions = load_test_data()  # [1000, 8, 7]

naive_mse = compute_mse(naive_model, test_actions)
robust_mse = compute_mse(robust_model, test_actions)

improvement = (naive_mse - robust_mse) / naive_mse * 100

print(f"Naive MSE: {naive_mse:.6f}")
print(f"Robust MSE: {robust_mse:.6f}")
print(f"Improvement: {improvement:.1f}%")
```

**预期结果**：
```
Naive MSE: 0.0182
Robust MSE: 0.0101
Improvement: 44.5%  ✅ 显著改进
```

### Step 4: 保存Checkpoint

确保checkpoint包含：
```python
checkpoint = {
    'model': model.state_dict(),
    'epoch': epoch,
    'mse': final_mse,
    'config': {
        'use_layernorm': True,  # ✅ 标记使用了LayerNorm
        'action_dim': 7,
        'hidden_dim': 16,
        'num_layers': 8,
        'num_levels': 7,
    },
}
torch.save(checkpoint, '/models/rfsq_robust_best.pt')
```

---

## 🔗 与Phase 2/3的关系

### 关键点：Phase 1 Improved是独立的

```
Phase 1 Improved (独立任务)
  ↓ 产出
rfsq_robust_best.pt
  ↓ 可选：用于Phase 2/3
重新训练整个pipeline（或只替换decoder）
```

**不需要立即集成**：
- Phase 1 Improved可以独立训练和验证
- 集成到Phase 2/3是**后续任务**
- 先验证改进效果，再决定是否重训

### 集成选项（给用户选择）

**选项A：完全重训**（最大收益，3-4天）
1. 用Robust RFSQ重训Main Model（Phase 2）
2. 用Robust RFSQ重训Draft Model（Phase 2 Draft）
3. Phase 3评估，获得最大精度提升

**选项B：只替换Decoder**（快速验证，1天）
1. Phase 2/3保持不变
2. 只在Phase 3中用Robust RFSQ decoder
3. 快速验证LayerNorm效果

**选项C：增量验证**（稳妥，2-3周）
1. 先训练Robust RFSQ
2. 测试decoder替换效果
3. 如果好，再重训Phase 2

---

## 📝 Agent实施清单

### 训练前

- [ ] 阅读`README.md`了解改进原理
- [ ] 阅读`rfsq_robust.py`理解实现
- [ ] 阅读`COMPARISON_GUIDE.md`了解预期改进

### 训练中

- [ ] 创建训练脚本（基于原始Phase 1修改）
- [ ] 设置`use_layernorm=True`
- [ ] 运行训练（100 epochs）
- [ ] 监控MSE曲线（目标: <0.012）

### 训练后

- [ ] 创建对比测试脚本
- [ ] 验证改进>30%
- [ ] 保存checkpoint到`/models/rfsq_robust_best.pt`
- [ ] 记录最终MSE和改进百分比

### 文档

- [ ] 更新训练log
- [ ] 记录最终结果
- [ ] 向用户报告改进效果
- [ ] 等待用户决定是否集成到Phase 2/3

---

## 🚨 常见问题

### Q1: 需要修改Phase 2/3的代码吗？

**A**: **现在不需要**。Phase 1 Improved是独立任务。

只有当用户决定集成时，才需要修改Phase 2/3。

### Q2: 训练时间会增加吗？

**A**: 会，但很小（+5%）。

LayerNorm计算很快，主要时间仍在encoder/decoder。

### Q3: 如果改进<30%怎么办？

**A**: 检查：
1. 是否正确设置`use_layernorm=True`？
2. 训练数据是否充足？
3. 训练是否收敛？

如果确实改进不显著，向用户报告，可能不值得重训Phase 2。

### Q4: Checkpoint格式有变化吗？

**A**: 没有。完全兼容原始Phase 1。

只是在config中添加`use_layernorm: True`标记。

---

## 🔧 代码修改示例

### 原始Phase 1训练脚本

```python
# train_rfsq.py（原始）
from rfsq_original import ActionRFSQAE  # 原始实现

model = ActionRFSQAE(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
)
```

### 改进的训练脚本

```python
# train_rfsq_robust.py（新建）
from rfsq_robust import ActionRFSQAE  # ✅ 改用robust版本

model = ActionRFSQAE(
    action_dim=7,
    hidden_dim=16,
    num_layers=8,
    num_levels=7,
    use_layernorm=True,  # ✅ 启用LayerNorm
)
```

**就这么简单！其他训练逻辑完全相同。**

---

## 📊 预期结果总结

| Metric | Target | 你的结果 |
|--------|--------|---------|
| Final MSE | <0.012 | ____ |
| Improvement vs Naive | >30% | ____% |
| Training Time | ~3小时 | ____ |
| Checkpoint Size | ~50KB | ____ |

---

## 🎯 成功标准

训练成功的标志：
- ✅ Final MSE < 0.012
- ✅ Improvement > 30%
- ✅ Checkpoint保存成功
- ✅ 测试脚本验证通过

**完成后**，向用户报告：
```
Robust RFSQ训练完成！

结果：
- Final MSE: 0.0101
- Improvement: 44.5%
- Checkpoint: /models/rfsq_robust_best.pt

建议：改进显著，建议考虑重训Phase 2以获得最大收益。
```

---

## 📖 参考文档

- **原理**：`README.md`
- **对比**：`COMPARISON_GUIDE.md`
- **集成**：`INTEGRATION_TO_PHASE2.md`（仅供参考，不需要现在实施）

---

**准备好了吗？开始训练Robust RFSQ！🚀**
