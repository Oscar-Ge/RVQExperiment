# 🔧 Phase 3 成功率为0的修复指南

## 问题诊断总结

当前成功率为0的**根本原因**：

1. ❌ **使用了未训练的OpenVLA模型** (`openvla/openvla-7b`)
   - 这个模型没有在LIBERO上训练，不知道如何完成LIBERO任务
   - 应该使用：`moojink/openvla-7b-oft-finetuned-libero-spatial`

2. ❌ **没有使用训练好的RFSQ head**
   - 虽然加载了`best_rfsq_head.pt`，但generate_action中完全没用它
   - 当前只调用了OpenVLA的`predict_action`（L1回归），没有用RFSQ token预测

3. ❌ **动作生成逻辑错误**
   - 只重复单个动作8次，而不是预测8个不同的动作
   - 没有使用RFSQ decoder解码token sequences

4. ❌ **错误处理太宽松**
   - 失败时默默使用随机动作，导致所有episode失败但看不到真正的错误

---

## 修复步骤

### Step 1: 修改模型加载（第302行）

```python
# ❌ 错误：使用原版OpenVLA
base_model_name = "openvla/openvla-7b"

# ✅ 正确：使用LIBERO fine-tuned版本
base_model_name = "moojink/openvla-7b-oft-finetuned-libero-spatial"
```

### Step 2: 重写generate_action方法（第559-598行）

**当前代码（错误）**：
```python
@torch.no_grad()
def generate_action(self, observation, task_description, chunk_len=8, action_dim=7):
    # ... prepare image ...

    # ❌ 只用OpenVLA的L1回归head
    action = self.main_model.predict_action(
        image,
        task_description,
        unnorm_key="libero_spatial",
        do_sample=False,
    )

    # ❌ 只是重复单个动作
    actions = np.tile(action, (chunk_len, 1))
    return actions, info
```

**正确实现**：
```python
@torch.no_grad()
def generate_action(self, observation, task_description, chunk_len=8, action_dim=7):
    """Generate action using RFSQ token prediction."""
    start_time = time.time()

    # Prepare image
    if isinstance(observation['full_image'], np.ndarray):
        image = Image.fromarray(observation['full_image'].astype(np.uint8))
    else:
        image = observation['full_image']

    try:
        # Step 1: Get hidden states from OpenVLA backbone
        inputs = self.processor(
            text=task_description,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Get model outputs (hidden states)
            outputs = self.main_model(**inputs, output_hidden_states=True)

            # Extract final hidden state
            # Shape: [Batch=1, Seq_Len, Hidden_Dim=4096]
            hidden_states = outputs.hidden_states[-1]

            # Take last token's hidden state
            # Shape: [Batch=1, Hidden_Dim=4096]
            final_hidden = hidden_states[:, -1, :]

            # Step 2: Use RFSQ head to predict token indices
            # Shape: [Batch=1, Num_Layers=8, Chunk=8, Hidden=16, Grid=7]
            logits = self.main_model.rfsq_head(final_hidden)

            # Get predicted indices (greedy decoding)
            # Shape: [Batch=1, Num_Layers=8, Chunk=8, Hidden=16]
            predicted_indices = torch.argmax(logits, dim=-1)

            # Step 3: Decode RFSQ indices to continuous actions
            # Reshape: [B=1, L=8, C=8, H=16] -> [B=1, C=8, H=16, L=8]
            indices_reshaped = predicted_indices.permute(0, 2, 3, 1)

            # Decode using RFSQ decoder
            # Output shape: [Batch=1, Chunk=8, Action_Dim=7]
            actions_tensor = self.rfsq_decoder.decode_from_indices(indices_reshaped)

            # Convert to numpy
            actions = actions_tensor.squeeze(0).cpu().numpy()  # [8, 7]

            # Clip to valid range
            actions = np.clip(actions, -1.0, 1.0)

        inference_time = time.time() - start_time
        self.stats['total_inferences'] += 1
        self.stats['total_time'] += inference_time

        info = {
            'total_time': inference_time,
            'used_rfsq': True,
        }

        return actions, info

    except Exception as e:
        print(f"      ❌ RFSQ prediction failed: {e}")
        import traceback
        traceback.print_exc()

        # 不要使用随机动作，直接raise让错误暴露
        raise RuntimeError(f"Action generation failed: {e}")
```

### Step 3: 处理RFSQ head缺失的情况（第405-407行）

**当前代码**：
```python
else:
    print(f"   ⚠️  RFSQ head not found at {rfsq_head_path}")
    main_model.rfsq_head = None  # ❌ 设置为None但后续会crash
```

**修复**：
```python
else:
    print(f"   ❌ RFSQ head not found at {rfsq_head_path}")
    print(f"   Cannot run evaluation without RFSQ head!")
    raise FileNotFoundError(f"RFSQ head checkpoint not found: {rfsq_head_path}")
```

### Step 4: 添加调试输出

在generate_action中添加：
```python
# 在Step 1后添加
print(f"      Hidden states shape: {hidden_states.shape}")

# 在Step 2后添加
print(f"      Logits shape: {logits.shape}")
print(f"      Predicted indices shape: {predicted_indices.shape}")
print(f"      Sample indices [0,0,0,:5]: {predicted_indices[0,0,0,:5]}")

# 在Step 3后添加
print(f"      Actions shape: {actions.shape}")
print(f"      Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
print(f"      Sample actions[0]: {actions[0]}")
```

---

## 预期改进

### 修复前
```
Task 1: ✗✗✗✗✗ (0/5) - 0% success
Task 2: ✗✗✗✗✗ (0/5) - 0% success
...
Overall: 0% (0/50)
```

### 修复后（使用正确模型）
```
Task 1: ✓✓✓✗✓ (4/5) - 80% success
Task 2: ✓✓✓✓✓ (5/5) - 100% success
...
Overall: 85-95% (43-47/50)
```

---

## Baseline对比验证

为了确认RFSQ没有明显降低性能，应该对比：

### Baseline (OpenVLA-OFT with L1 regression)
- 使用`moojink/openvla-7b-oft-finetuned-libero-spatial`
- 用原生`predict_action`（L1回归head）
- 预期：~97% success rate

### RSD (OpenVLA-OFT-RFSQ with RFSQ tokens)
- 使用同样的base model
- 用训练好的RFSQ head + decoder
- 预期：85-95% success rate（略微下降是正常的）

**如果RSD成功率太低（<80%）**，可能是：
1. RFSQ head训练不充分（检查Phase 2的90.9% token accuracy是否真实）
2. RFSQ decoder重构误差太大（检查Phase 1的MSE）
3. 解码逻辑有bug（检查indices shape和decode_from_indices实现）

---

## 快速测试

修复后，先运行单个trial测试：

```bash
modal run modal_phase3_libero_eval.py --num-trials 1 --use-speculative-decoding False
```

**期望输出**：
```
   Hidden states shape: torch.Size([1, 256, 4096])
   Logits shape: torch.Size([1, 8, 8, 16, 7])
   Predicted indices shape: torch.Size([1, 8, 8, 16])
   Sample indices [0,0,0,:5]: tensor([3, 4, 3, 2, 4])
   Actions shape: (8, 7)
   Actions range: [-0.856, 0.923]
   Sample actions[0]: [ 0.234 -0.123  0.456  0.012 -0.234  0.567 -0.890]

Trial 1/1: ✓ (time: 45.2s, inf: 89.3ms)
```

如果仍然失败，检查：
1. `hidden_states`是否全0？（vision encoder问题）
2. `predicted_indices`是否都一样？（RFSQ head问题）
3. `actions`是否范围正常？（RFSQ decoder问题）

---

## 总结

**核心修复**：
1. ✅ 使用正确的base model（OFT fine-tuned版本）
2. ✅ 通过RFSQ head预测tokens，而不是直接predict_action
3. ✅ 用RFSQ decoder解码tokens成actions
4. ✅ 暴露错误而不是默默使用随机动作

**为什么之前是0%**：
- 原版OpenVLA不知道如何做LIBERO任务
- 即使有fine-tuned model，也没有使用训练好的RFSQ pipeline
- 错误被隐藏了（随机动作导致失败但看不到root cause）

**预期结果**：
- 修复后应该达到85-95% success rate
- 如果还是<80%，需要检查Phase 1/2的训练质量

---

**现在的action plan**：

1. **立即修复**：按Step 1-4修改代码
2. **测试**：`--num-trials 1`验证不crash
3. **验证**：`--num-trials 5`确认有成功的episodes
4. **完整评估**：`--num-trials 50`得到最终指标

Good luck! 🚀
