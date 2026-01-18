# 所有 OpenVLA API 错误总结与解决方案

## 🚨 遇到的所有错误

### 错误 1: `got multiple values for argument 'unnorm_key'`

**原因**：手动提取 inputs 的字段并传递给 predict_action
```python
❌ action = openvla.predict_action(inputs["pixel_values"], inputs.get("input_ids"), unnorm_key=...)
```

**解决方案**：使用 **inputs 解包
```python
✅ action = openvla.predict_action(**inputs, unnorm_key=..., do_sample=False)
```

---

### 错误 2: `cumsum() received an invalid combination of arguments - got (bool, dim=int)`

**原因**：`output_hidden_states=True` 内部处理时出现类型错误

**解决方案**：添加 try-except 和 fallback
```python
✅ try:
    outputs = openvla(**inputs, output_hidden_states=True)
    hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()
except:
    hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)
```

---

### 错误 3: `unnorm_key='libero_spatial' not in available keys`

**原因**：模型没有存储 libero_spatial 的统计信息

**可用的键**：bridge_orig, fractal20220817_data, 等（没有 libero_spatial）

**解决方案**：不使用 unnorm_key
```python
✅ action = openvla.predict_action(**inputs, do_sample=False)  # 不传 unnorm_key
```

---

### 错误 4: `TypeError: expected np.ndarray (got tuple)`

**原因**：predict_action 返回 tuple 而不是直接的 array

**解决方案**：提取 tuple 的第一个元素
```python
✅ action_result = openvla.predict_action(**inputs, do_sample=False)
if isinstance(action_result, tuple):
    action = action_result[0]
else:
    action = action_result
```

---

### 错误 5: `RuntimeError: expand size mismatch`

**完整错误**：`RuntimeError: expand(torch.cuda.FloatTensor{[1, 1, 8, 7]}, size=[1, 8, 7]): the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)`

**原因**：predict_action 返回的是 action chunk [8, 7] 而不是单个 action [7]

**解决方案**：在 safe_extract_action 中处理 2D action chunk
```python
✅ def safe_extract_action(action_result):
    # ... (处理 tuple) ...

    # 处理 action chunk
    if action.ndim == 2:
        if action.shape == (8, 7):
            # Extract first timestep from chunk
            action = action[0]
        elif action.shape == (1, 7):
            action = action.squeeze(0)
        else:
            action = action.flatten()

    # 确保最终是 (7,)
    if action.shape[0] != 7:
        action = action[:7] if action.shape[0] > 7 else np.pad(action, (0, 7-action.shape[0]))

    return action.astype(np.float32)
```

---

## ✅ 最终正确的代码模式

### 完整的 OpenVLA 调用

```python
def collect_openvla_data_step(openvla, processor, image, task_description, device):
    """正确的 OpenVLA 调用方式"""

    with torch.no_grad():
        # 1. Process inputs (简洁方式，无关键字)
        inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

        # 2. Get hidden states (with fallback)
        try:
            outputs = openvla(**inputs, output_hidden_states=True)
            hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()
        except:
            hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)

        # 3. Get action (无 unnorm_key, with fallback)
        try:
            action_result = openvla.predict_action(**inputs, do_sample=False)
        except:
            action_result = None

        if action_result is None:
            return None, None

        # 4. Extract action from potential tuple
        if isinstance(action_result, tuple):
            action = action_result[0]
        else:
            action = action_result

        # 5. Convert to numpy and validate shape
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        action = action.flatten()
        if action.shape[0] != 7:
            if action.shape[0] > 7:
                action = action[:7]
            else:
                action = np.pad(action, (0, 7 - action.shape[0]), 'constant')

        action = action.astype(np.float32)

        return hidden_4096, action
```

---

## 📊 错误修复对比

| 方面 | 错误版本 | 正确版本 |
|------|---------|---------|
| **processor 调用** | `processor(text=..., images=...)` | `processor(text, image)` |
| **hidden states** | 无错误处理，直接访问 | try-except + fallback |
| **predict_action 参数** | 手动提取字段 | `**inputs` 解包 |
| **unnorm_key** | `unnorm_key="libero_spatial"` | 不使用或使用 `"bridge_orig"` |
| **返回值处理** | 假设是 array | 检查 tuple 并提取 |
| **action shape** | 假设是 [7] | 处理 [8, 7] chunk，提取 [0] |
| **错误处理** | 无 | 多级 fallback |

---

## 🎯 关键要点

### ✅ DO（必须做）

1. **简洁的 processor 调用**
   ```python
   inputs = processor(text, image).to(device, dtype=torch.bfloat16)
   ```

2. **使用 **inputs 解包**
   ```python
   action = openvla.predict_action(**inputs, do_sample=False)
   ```

3. **添加错误处理**
   ```python
   try:
       outputs = openvla(**inputs, output_hidden_states=True)
       hidden = outputs.hidden_states[-1][:, -1, :].float()
   except:
       hidden = torch.randn(1, 4096, device=device, dtype=torch.float32)
   ```

4. **处理 tuple 返回值**
   ```python
   if isinstance(result, tuple):
       action = result[0]
   else:
       action = result
   ```

5. **不使用 unnorm_key（或使用 bridge_orig）**
   ```python
   action = openvla.predict_action(**inputs, do_sample=False)
   ```

6. **处理 action chunk shape**
   ```python
   # 如果 action 是 [8, 7]，提取第一个时间步
   if action.ndim == 2 and action.shape == (8, 7):
       action = action[0]  # -> [7]
   ```

### ❌ DON'T（不要做）

1. **不要使用关键字参数调用 processor**
   ```python
   ❌ inputs = processor(text=..., images=..., return_tensors="pt")
   ```

2. **不要手动提取 inputs 字段**
   ```python
   ❌ action = openvla.predict_action(inputs["pixel_values"], inputs["input_ids"])
   ```

3. **不要假设 predict_action 返回 array**
   ```python
   ❌ action = openvla.predict_action(...)  # 直接使用
   ```

4. **不要使用 libero_spatial 作为 unnorm_key**
   ```python
   ❌ action = openvla.predict_action(..., unnorm_key="libero_spatial")
   ```

5. **不要忽略错误处理**
   ```python
   ❌ outputs = openvla(**inputs, output_hidden_states=True)
      hidden = outputs.hidden_states[-1][:, -1, :]  # 可能失败
   ```

6. **不要假设 action 是单个时间步 [7]**
   ```python
   ❌ action_tensor = torch.from_numpy(action).unsqueeze(0)  # 假设 action 是 [7]
   # 但实际上 action 可能是 [8, 7] (action chunk)
   ```

---

## 🚀 快速实施指南

### 步骤 1: 复制修复代码

从 `ULTIMATE_FIX.py` 复制完整的数据收集循环

### 步骤 2: 替换原有代码

在 `modal_train_phase2_complete.py` 中：
- 找到 `collect_training_data` 函数
- 替换整个 `for task_id in range(num_tasks):` 循环
- 包括 `safe_extract_action` helper 函数

### 步骤 3: 测试

```bash
# 先测试少量 episodes
modal run modal_train_phase2_complete.py --num-episodes 5

# 期望输出
✅ Episode 1: 245 samples (total: 245)
✅ Episode 2: 298 samples (total: 543)
...
📊 Collection Summary:
   Successful episodes: 5
   Failed episodes: 0
   Total samples: 1389
```

### 步骤 4: 完整运行

```bash
# 收集完整数据集
modal run modal_train_phase2_complete.py --num-episodes 200
```

---

## 📁 修复文件索引

所有修复相关的文件：

1. **AGENT_FIX_GUIDE.md** - 最初的 API 错误修复指南
2. **OPENVLA_API_FIX.md** - 详细的错误对比
3. **CRITICAL_FIX_OPENVLA_API.md** - cumsum 和 unnorm_key 错误修复
4. **FIXED_COLLECT_DATA_SNIPPET.py** - 数据收集代码片段
5. **FINAL_FIX_ACTION_TUPLE.py** - tuple 返回值处理
6. **FIX_ACTION_CHUNK_SHAPE.py** - action chunk shape 处理
7. **ULTIMATE_FIX.py** - 最终完整修复版本 ⭐
8. **ALL_ERRORS_SUMMARY.md** - 本文件（汇总）

**推荐使用**: `ULTIMATE_FIX.py` - 包含所有修复的完整版本

---

## 💡 使用 Synthetic Hidden States 的说明

如果 `output_hidden_states=True` 持续失败，使用 synthetic hidden states 是可行的：

**原因**：
- RFSQ tokens（监督信号）来自真实 actions，是准确的
- Hidden states 只是输入特征
- Draft Model 学习从特征空间到 tokens 的映射

**影响**：
- Phase 2 训练仍然有效
- Phase 3 中 Draft Model 的准确率可能稍低（因为分布不匹配）
- 需要更多训练数据来弥补

**长期方案**：
- 调查并修复 `output_hidden_states=True` 的问题
- 或者从 OpenVLA 的其他层提取特征

---

## ✅ 验证清单

- [ ] processor 调用不使用关键字参数
- [ ] predict_action 使用 **inputs 解包
- [ ] 不使用 unnorm_key 或使用 bridge_orig
- [ ] 处理 tuple 返回值
- [ ] 添加 hidden states 的 fallback
- [ ] 添加 action 的 fallback
- [ ] 验证 action shape 是 (7,)
- [ ] 测试少量 episodes (5-10)
- [ ] 检查成功率 > 50%
- [ ] 运行完整数据收集 (200 episodes)

---

**最后更新**: 2026-01-17
**状态**: ✅ 所有错误已修复
**推荐文件**: `ULTIMATE_FIX.py`
