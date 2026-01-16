# Draft Model Training & Integration - Testing Checklist

## 📋 训练前检查

- [ ] Modal环境可用
- [ ] GPU资源充足（A100，预计4-6小时）
- [ ] OpenVLA fine-tuned模型可访问（`moojink/openvla-7b-oft-finetuned-libero-spatial`）
- [ ] Phase 1 RFSQ decoder可用（`/models/rfsq_best.pt`）
- [ ] LIBERO环境配置正确

---

## 📦 数据收集阶段

### 运行命令
```bash
modal run modal_train_draft_with_projection.py \
    --num-episodes 200 \
    --skip-training True  # 只收集数据
```

### 验证清单

- [ ] **数据收集开始**
  ```
  ✅ OpenVLA loaded
  ✅ RFSQ Encoder loaded
  ✅ 10 tasks in libero_spatial
  ```

- [ ] **数据收集进度**
  ```
  Task 1/10: pick up the red block
    Episode 1: 1250 samples
    Episode 2: 2430 samples
  ...
  ```

- [ ] **数据保存成功**
  ```
  ✅ Saved 60,000+ samples to /data/draft_training_data.pt
  ```

- [ ] **验证数据**
  ```bash
  modal run -c "
  import torch
  data = torch.load('/data/draft_training_data.pt')
  print(f'Total samples: {len(data)}')
  print(f'Sample keys: {data[0].keys()}')
  print(f'Hidden shape: {data[0][\"hidden_state\"].shape}')  # [4096]
  print(f'Tokens shape: {data[0][\"coarse_tokens\"].shape}')  # [8, 16, 3]
  "
  ```

**预期**：
- 样本数：60,000-150,000
- Hidden shape: `[4096]`
- Tokens shape: `[8, 16, 3]`

---

## 🚀 训练阶段

### 运行命令
```bash
modal run modal_train_draft_with_projection.py \
    --num-episodes 200 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4
```

### 验证清单

- [ ] **模型创建**
  ```
  ✅ Model created: 4.7M parameters
  Train samples: 54,000
  Val samples: 6,000
  ```

- [ ] **训练开始**
  ```
  Epoch 1/50
    Train Loss: 1.850 | Train Acc: L0=0.452 L1=0.428 L2=0.401
    Val Loss: 1.780 | Val Acc: L0=0.465 L1=0.441 L2=0.415 | Avg=0.440
  ```

- [ ] **训练进度（每5个epochs检查）**

  **Epoch 5-10**:
  - [ ] Train loss < 1.0
  - [ ] Val accuracy (avg) > 60%

  **Epoch 15-20**:
  - [ ] Train loss < 0.4
  - [ ] Val accuracy (avg) > 80%

  **Epoch 25-30**:
  - [ ] Train loss < 0.25
  - [ ] Val accuracy (avg) > 85%

  **Epoch 40-50**:
  - [ ] Train loss < 0.18
  - [ ] Val accuracy (avg) > 87%

- [ ] **最佳模型保存**
  ```
  ✅ Best model saved: 0.892
  ```

- [ ] **训练完成**
  ```
  🎉 Training Complete!
     Best Val Accuracy: 0.892
  ```

**目标**：Val accuracy > 85%

---

## ✅ 训练后验证

### 检查Checkpoint

```bash
# 1. Checkpoint存在
modal volume ls rsd-models | grep best_draft_with_projection.pt

# 2. 下载并检查
modal volume get rsd-models best_draft_with_projection.pt ./

# 3. 检查内容
python -c "
import torch
ckpt = torch.load('best_draft_with_projection.pt', weights_only=False)
print('=' * 60)
print('Checkpoint Validation')
print('=' * 60)
print(f'Keys: {list(ckpt.keys())}')
print(f'Val Accuracy: {ckpt[\"val_accuracy\"]:.3f}')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Per-layer acc: {ckpt[\"val_accuracies_per_layer\"]}')
print()
print('Model State Dict Keys (first 10):')
for i, key in enumerate(list(ckpt['model_state_dict'].keys())[:10]):
    print(f'  {i+1}. {key}')
print()
has_proj = any('projection' in k for k in ckpt['model_state_dict'].keys())
print(f'Has projection: {has_proj}')
print('=' * 60)
"
```

**预期输出**：
```
============================================================
Checkpoint Validation
============================================================
Keys: ['model_state_dict', 'optimizer_state_dict', 'epoch', 'val_accuracy', ...]
Val Accuracy: 0.892
Epoch: 47
Per-layer acc: [0.919, 0.898, 0.873]

Model State Dict Keys (first 10):
  1. input_projection.weight
  2. input_projection.bias
  3. decoder.decoder_layer.self_attn.in_proj_weight
  ...

Has projection: True
============================================================
```

验证清单：
- [ ] Val accuracy > 0.85
- [ ] Has `input_projection.weight` and `input_projection.bias`
- [ ] Checkpoint包含所有必要keys

### 测试模型加载

```python
# test_model_loading.py
import torch
import torch.nn as nn

# 复制模型定义（从modal_train_draft_with_projection.py）
class DraftTransformerDecoder(nn.Module):
    # ... (完整定义)
    pass

class RFSQDraftModelWithProjection(nn.Module):
    # ... (完整定义)
    pass

# 测试
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RFSQDraftModelWithProjection(
    input_dim=4096,
    hidden_dim=512,
    num_coarse_layers=3,
)

checkpoint = torch.load('best_draft_with_projection.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 测试推理
test_input = torch.randn(1, 4096).to(device)
with torch.no_grad():
    output = model(test_input)

print(f"✅ Input shape: {test_input.shape}")
print(f"✅ Output shape: {output.shape}")
print(f"✅ Expected: [1, 3, 128, 7]")
assert output.shape == (1, 3, 128, 7)
print("✅ All tests passed!")
```

- [ ] 模型可以加载
- [ ] Forward pass不报错
- [ ] 输出shape正确：`[1, 3, 128, 7]`

---

## 🔗 集成到Phase 3

### 更新代码

参考`INTEGRATION_GUIDE.md`：

- [ ] 更新`rsd_engine_core.py`：
  - [ ] 添加Draft Model定义
  - [ ] 删除随机初始化的projection
  - [ ] 更新Draft预测逻辑

- [ ] 更新`modal_phase3_libero_eval.py`：
  - [ ] 导入新的Draft Model类
  - [ ] 更新Draft Model加载代码

### 测试集成

```bash
# Test 1: 单次推理
modal run modal_phase3_libero_eval.py \
    --num-trials 1 \
    --use-speculative-decoding True
```

验证logs：
- [ ] Draft Model加载成功
  ```
  ✅ Draft Model loaded (val_acc: 0.892)
  ✅ Projection layer included: True
  ```

- [ ] RSD Engine创建成功
  ```
  ✅ RSD Inference Engine created
     Hidden size: 4096
     Draft hidden size: 512
  ```

- [ ] 第一次推理成功
  ```
  Draft time: 12.3ms
  Draft tokens shape: torch.Size([1, 3, 128])
  Main time: 45.7ms
  Acceptance rate: 72.5%
  Actions shape: (8, 7)
  ```

- [ ] Episode完成
  ```
  Trial 1/1: ✓ (steps: 127, inf: 285.3ms)
  ```

```bash
# Test 2: 小规模测试
modal run modal_phase3_libero_eval.py \
    --num-trials 5 \
    --use-speculative-decoding True
```

验证结果：
- [ ] Success rate > 70%
- [ ] Inference time < 60ms
- [ ] Draft acceptance rate > 50%

---

## 🎯 完整评估

```bash
# 运行完整评估
modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True
```

### 预期结果

- [ ] **成功率**
  ```
  Total Successes: 44/50 (88%)
  Success Rate: 85-95%
  ```

- [ ] **推理时间**
  ```
  Avg Inference Time: 45-55ms
  (Baseline without Draft: ~70ms)
  Speedup: 1.3-1.6x ✅
  ```

- [ ] **Draft统计**
  ```
  Draft acceptance rate: 60-80%
  Avg draft time: 10-15ms
  Avg main time: 25-35ms
  ```

- [ ] **任务分布**
  ```
  Task 1: 88% (44/50)
  Task 2: 92% (46/50)
  ...
  All tasks > 70%
  ```

---

## 📊 性能对比表

填写实际测试结果：

| Metric | Phase 3 (无Draft) | Phase 3 (新Draft) | 目标 | 达成 |
|--------|------------------|------------------|------|------|
| Success Rate | ____% | ____% | 85-95% | [ ] |
| Avg Inference Time | ____ms | ____ms | 45-55ms | [ ] |
| Speedup | 1.0x | ____x | 1.3-1.6x | [ ] |
| Draft Acceptance | N/A | ____% | 60-80% | [ ] |
| GPU Memory | ____GB | ____GB | <16GB | [ ] |

---

## 🐛 故障排查

如果任何检查项失败，参考：

- **训练问题** → `TRAINING_PLAN.md` 的"常见问题"部分
- **集成问题** → `INTEGRATION_GUIDE.md` 的"故障排查"部分
- **性能问题** → 检查：
  - [ ] Draft Model accuracy是否>85%
  - [ ] Main Model是否用了fine-tuned版本
  - [ ] RFSQ decoder是否正确工作
  - [ ] Token comparison逻辑是否实现

---

## ✅ 最终签收

所有检查项通过后：

- [ ] 训练数据收集成功（60k+ samples）
- [ ] 模型训练完成（val acc > 85%）
- [ ] Checkpoint验证通过
- [ ] 集成到Phase 3成功
- [ ] 单次推理测试通过
- [ ] 小规模测试通过（5 trials）
- [ ] 完整评估通过（50 trials）
- [ ] 性能达标（success rate 85-95%, speedup 1.3-1.6x）

**签收人**：__________________  **日期**：__________

---

## 📝 报告模板

测试完成后，填写以下报告：

```
# Draft Model Training Report

## Training Summary
- Date: ___________
- Episodes: 200
- Epochs: 50
- Training Time: ___ hours
- Best Val Accuracy: ____%

## Integration Results
- Phase 3 Success Rate: ____%
- Avg Inference Time: ___ms
- Speedup vs Baseline: ___x
- Draft Acceptance Rate: ____%

## Conclusion
- [ ] Training successful (acc > 85%)
- [ ] Integration successful
- [ ] Performance meets target
- [ ] Ready for production use

## Next Steps
- [ ] Run full benchmark (500 trials)
- [ ] Document results
- [ ] Archive checkpoints
```

---

**Good luck with testing! 🚀**
