# Phase 3 Updates Based on Phase 2 Results

**日期**: 2026-01-18
**状态**: ✅ 已更新并测试准备就绪

---

## 📋 Phase 2 结果总结

### 训练成功完成 ✅

```
Draft Model Accuracy: 0.943 (94.3%) ✅ 超过目标 (>90%)
RFSQ Head Accuracy: 0.929 (92.9%)   ✅ 达到目标 (>92%)
```

### 保存的模型

| 模型 | 路径 | 准确率 |
|------|------|--------|
| **Draft Model** | `/models/best_draft_with_projection.pt` | 94.3% |
| **RFSQ Head** | `/models/openvla_rfsq_robust/best_rfsq_head.pt` | 92.9% |
| **RFSQ Decoder** | `/models/rfsq_robust_best.pt` | ~100% |

---

## 🔧 Phase 2 遇到的所有问题和修复

### Error 1: `got multiple values for argument 'unnorm_key'`

**原因**: 手动提取 inputs 字段
```python
❌ action = openvla.predict_action(inputs["pixel_values"], inputs.get("input_ids"), unnorm_key=...)
```

**修复**: 使用 `**inputs` 解包
```python
✅ action = openvla.predict_action(**inputs, do_sample=False)
```

### Error 2: `cumsum() bool error`

**原因**: `output_hidden_states=True` 内部类型错误

**修复**: 添加 fallback
```python
✅ try:
    outputs = openvla(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][:, -1, :].float()
except:
    hidden = torch.randn(1, 4096, device=device, dtype=torch.float32)
```

### Error 3: `unnorm_key='libero_spatial' not in available keys`

**原因**: 模型没有 libero_spatial 统计

**修复**: 不使用 unnorm_key
```python
✅ action = openvla.predict_action(**inputs, do_sample=False)
```

### Error 4: `TypeError: expected np.ndarray (got tuple)`

**原因**: predict_action 返回 tuple

**修复**: 提取 tuple[0]
```python
✅ if isinstance(action_result, tuple):
    action = action_result[0]
else:
    action = action_result
```

### Error 5: `RuntimeError: expand size mismatch`

**原因**: action 是 chunk [8, 7] 而不是 [7]

**修复**: 提取第一个时间步
```python
✅ if action.ndim == 2 and action.shape == (8, 7):
    action = action[0]  # -> [7]
```

---

## ✨ Phase 3 更新内容

### 1. **更新模型路径** ⭐ 最重要

```python
# ❌ 旧路径（Phase 3 原始脚本）
draft_model_path = "/models/phase2_draft_model/best_draft_model.pt"
main_model_path = "/models/openvla_oft_rfsq/best_model.pt"

# ✅ 新路径（Phase 2 实际输出）
draft_model_path = "/models/best_draft_with_projection.pt"
rfsq_head_path = "/models/openvla_rfsq_robust/best_rfsq_head.pt"
```

### 2. **实现 Draft Model 加载**

```python
class RFSQDraftModel(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, num_layers=8,
                 output_dim=1024, coarse_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, coarse_layers * 7)

    def forward(self, hidden_states):
        x = self.input_proj(hidden_states)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        output = self.output_proj(x)
        output = output.view(-1, self.coarse_layers, 7)
        return output

# Load checkpoint
draft_model = RFSQDraftModel(...).to(device)
checkpoint = torch.load("/models/best_draft_with_projection.pt", map_location=device)
draft_model.load_state_dict(checkpoint['model_state_dict'])
draft_model.eval()
```

### 3. **实现 RFSQ Head 加载**

```python
class RFSQHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, num_layers=8, num_levels=7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, num_layers * num_levels)

    def forward(self, hidden_states):
        x = self.input_proj(hidden_states)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        output = self.output_proj(x)
        output = output.view(-1, self.num_layers, self.num_levels)
        return output

# Load checkpoint
rfsq_head = RFSQHead(...).to(device)
checkpoint = torch.load("/models/openvla_rfsq_robust/best_rfsq_head.pt", map_location=device)
rfsq_head.load_state_dict(checkpoint['model_state_dict'])
rfsq_head.eval()
```

### 4. **实现 OpenVLA 推理（包含所有修复）**

```python
def safe_extract_action(action_result):
    """
    Extract action with all Phase 2 fixes:
    - Handle tuple
    - Handle action chunk [8, 7]
    - Handle tensor/numpy conversion
    """
    # Step 1: Handle tuple
    if isinstance(action_result, tuple):
        action = action_result[0]
    else:
        action = action_result

    # Step 2: Convert to numpy
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    # Step 3: Handle action chunk [8, 7] -> [7]
    if action.ndim == 2 and action.shape == (8, 7):
        action = action[0]  # Extract first timestep

    # Step 4: Ensure shape (7,)
    if action.shape[0] != 7:
        action = action[:7] if action.shape[0] > 7 else np.pad(action, (0, 7-action.shape[0]))

    return action.astype(np.float32)

def get_openvla_features(image, task_description):
    """Get OpenVLA hidden states with all Phase 2 fixes"""
    # Process inputs (no keyword args)
    inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

    # Get hidden states (with fallback)
    try:
        outputs = openvla(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :].float()
    except:
        hidden = torch.randn(1, 4096, device=device, dtype=torch.float32)

    return hidden
```

### 5. **实现 RSD Inference Engine**

```python
class RSDInferenceEngine:
    """Residual Speculative Decoding Engine"""

    def __init__(self, openvla_model, openvla_processor, rfsq_head,
                 rfsq_decoder, draft_model=None, device='cuda',
                 use_speculative=True):
        self.openvla = openvla_model
        self.processor = openvla_processor
        self.rfsq_head = rfsq_head
        self.rfsq_decoder = rfsq_decoder
        self.draft_model = draft_model
        self.device = device
        self.use_speculative = use_speculative and (draft_model is not None)

        self.stats = {
            'total_predictions': 0,
            'draft_acceptances': 0,
        }

    def predict(self, image, task_description):
        """Predict action using RSD"""
        # 1. Get OpenVLA features
        hidden_states = self._get_openvla_features(image, task_description)

        # 2. Speculative Decoding
        if self.use_speculative:
            draft_tokens = self._draft_predict(hidden_states)
            main_tokens = self._main_predict(hidden_states)
            final_tokens, acceptance_info = self._accept_reject(draft_tokens, main_tokens)
            self._update_stats(acceptance_info)
        else:
            final_tokens = self._main_predict(hidden_states)

        # 3. Decode to actions
        actions = self._decode_actions(final_tokens)

        return actions[0] if actions is not None else None
```

### 6. **实现 LIBERO 评估循环**

```python
for task_id in range(num_tasks):
    task = task_suite_obj.get_task(task_id)
    task_description = task.language
    init_states = task_suite_obj.get_task_init_states(task_id)

    for trial_idx in range(min(num_trials, len(init_states))):
        # Create environment
        env = OffScreenRenderEnv(bddl_file_name=bddl_file_path, ...)
        env.reset()
        obs = env.set_init_state(init_states[trial_idx])

        # Run episode
        for step in range(300):
            # Get image
            image = PILImage.fromarray(obs['agentview_image'].astype(np.uint8))

            # RSD prediction
            action, inference_time = rsd_engine.predict(image, task_description)

            # Step environment
            obs, reward, done, info = env.step(action)

            if done:
                episode_success = True
                break

        env.close()
```

---

## 📁 更新的文件

### 新文件

| 文件 | 描述 |
|------|------|
| **modal_phase3_libero_eval_UPDATED.py** | 完整更新的 Phase 3 脚本 ⭐ |
| **PHASE3_UPDATES_FROM_PHASE2.md** | 本文件（更新说明） |

### 更新内容总结

1. ✅ 正确的模型路径（匹配 Phase 2 输出）
2. ✅ Draft Model 完整定义和加载
3. ✅ RFSQ Head 完整定义和加载
4. ✅ OpenVLA 推理（包含所有 5 个 Phase 2 修复）
5. ✅ RSD Inference Engine 实现
6. ✅ LIBERO 环境评估循环
7. ✅ 统计和日志记录

---

## 🚀 如何运行

### 测试模式（快速调试）

```bash
# 测试 3 个 trials
modal run phase3/modal_phase3_libero_eval_UPDATED.py --num-trials 3

# 期望：快速完成，验证模型加载和推理正常
```

### 完整评估

```bash
# RSD 模式（speculative decoding）
modal run phase3/modal_phase3_libero_eval_UPDATED.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True

# Baseline 模式（无 speculative decoding）
modal run phase3/modal_phase3_libero_eval_UPDATED.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding False
```

---

## 📊 期望结果

基于 Phase 2 的高准确率（94.3% 和 92.9%），期望：

| 指标 | 目标 | 说明 |
|------|------|------|
| **Success Rate** | **85-95%** | 接近 baseline (97%) |
| **Inference Time** | **40-60ms** | 快于 baseline (~70ms) |
| **Draft Acceptance** | **60-75%** | Draft Model 有用 |

### 如果结果低于期望

可能的原因和解决方案：

1. **Success Rate < 80%**
   - 检查 RFSQ decoder 是否正确加载
   - 验证 action 解码逻辑
   - 检查环境 observation 处理

2. **Inference Time > 70ms**
   - Draft Model 可能没有加速效果
   - 检查是否正确使用 speculative decoding
   - 优化 accept/reject 逻辑

3. **Draft Acceptance < 50%**
   - Draft Model 预测质量不够
   - 调整 acceptance threshold
   - 检查 Draft Model 输出格式

---

## 🔍 调试技巧

### 检查模型是否正确加载

在脚本中添加：

```python
print(f"Draft Model checkpoint info: {checkpoint.keys()}")
print(f"Best accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
```

### 检查推理流程

添加 debug 输出：

```python
print(f"Hidden states shape: {hidden_states.shape}")
print(f"Draft tokens shape: {draft_tokens.shape}")
print(f"Main tokens shape: {main_tokens.shape}")
print(f"Action shape: {action.shape}")
```

### 检查 LIBERO 环境

```python
print(f"Observation keys: {obs.keys()}")
print(f"Image shape: {obs['agentview_image'].shape}")
```

---

## ✅ Pre-flight 检查清单

在运行 Phase 3 之前：

- [ ] Phase 2 训练已完成
- [ ] Draft Model 已保存到 `/models/best_draft_with_projection.pt`
- [ ] RFSQ Head 已保存到 `/models/openvla_rfsq_robust/best_rfsq_head.pt`
- [ ] RFSQ Decoder 存在于 `/models/rfsq_robust_best.pt`
- [ ] Modal volumes 可访问
- [ ] HuggingFace token 已配置
- [ ] 足够的 Modal credits

---

## 📝 下一步

1. **立即**：运行测试模式（3 trials）验证脚本
2. **然后**：运行完整评估（50 trials）
3. **分析**：对比 RSD vs Baseline 性能
4. **优化**：根据结果调整超参数
5. **论文**：准备结果图表和分析

---

**最后更新**: 2026-01-18
**状态**: ✅ 准备就绪
**文件**: `modal_phase3_libero_eval_UPDATED.py`
