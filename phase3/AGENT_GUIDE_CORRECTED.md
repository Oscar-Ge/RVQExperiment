# 🎯 Phase 3 正确实现指南 (For Experiment Agent)

## 📋 发现的问题总结

原始`modal_phase3_libero_eval.py`中发现了**12个致命问题**：

### 你朋友指出的5个问题：
1. ❌ **完全缺失Speculative Decoding逻辑** - 加载了Draft Model但从未使用
2. ❌ **数据流断裂** - `predict_action()`是黑盒，拿不到hidden states
3. ❌ **随机初始化action_head** - 临时创建的head输出随机噪声
4. ❌ **图像预处理缺失** - 切换到手动前向传播后会缺少预处理
5. ❌ **RFSQ解码逻辑未连接** - 定义了decoder但从未调用

### 额外发现的7个问题：
6. ❌ **Draft Model维度不匹配** - OpenVLA输出4096维，Draft期望512维
7. ❌ **RFSQ Head shape理解错误** - 注释误导，维度转换错误
8. ❌ **Chunk执行逻辑不完整** - 预测8步但只执行第1步
9. ❌ **成功判定有bug** - timeout会被误判为成功
10. ❌ **缺少projection layer** - 无法连接OpenVLA和Draft Model
11. ⚠️ **OpenVLA API使用不明确** - 需要调研如何获取hidden states
12. ⚠️ **Action denormalization缺失** - 可能影响性能

**结果**：成功率为0%，因为使用了未训练的模型 + 逻辑完全错误。

---

## ✅ 修复方案

我们提供了**完全重写**的正确实现：

### 新增文件

1. **`rsd_engine_core.py`** - 核心RSD Engine（纯Python，Modal-agnostic）
   - 完整的Speculative Decoding逻辑
   - 正确的数据流：输入 → Hidden States → Token Prediction → RFSQ Decoding
   - Draft projection layer (4096→512)
   - 正确的shape转换
   - Chunk执行辅助函数

2. **`CORRECTED_ENGINE_TEMPLATE.py`** - 详细的实现模板（含注释）

3. **本文件** - Agent实现指南

---

## 🚀 你的任务

### Step 1: 理解核心实现

阅读`rsd_engine_core.py`，了解正确的数据流：

```
Image + Text
  ↓ [processor预处理]
Inputs (pixel_values, input_ids)
  ↓ [OpenVLA forward]
Hidden States [Batch=1, Seq, Hidden=4096]
  ↓ [取最后一个token]
Last Hidden State [1, 4096]
  ↓
┌─────────────────────┬─────────────────────┐
│ Draft Path (加速)    │ Main Path (准确)     │
├─────────────────────┼─────────────────────┤
│ Projection (4096→512)│ RFSQ Head (4096→...)│
│ Draft Model         │ 8 Linear Heads      │
│ Coarse 3 Layers     │ All 8 Layers        │
└─────────────────────┴─────────────────────┘
  ↓ [Comparison & Fusion]
Final Tokens [1, 8, 8, 16]
  ↓ [Permute: Layers维度移到最后]
Tokens [1, 8, 16, 8]
  ↓ [RFSQ Decoder]
Actions [1, 8, 7]
```

### Step 2: 在Modal环境中使用

你需要在Modal脚本中：

#### A. 导入核心engine

```python
# 在modal_phase3_libero_eval.py开头
# 确保rsd_engine_core.py在同一目录或Python path中

from rsd_engine_core import RSDInferenceEngine, run_episode_with_chunks
```

#### B. 修复模型加载（第302行）

```python
# ❌ 错误：
base_model_name = "openvla/openvla-7b"

# ✅ 正确：
base_model_name = "moojink/openvla-7b-oft-finetuned-libero-spatial"
```

#### C. 替换SimpleRSDEngine（第530-611行）

删除整个`SimpleRSDEngine`类，使用：

```python
# 创建Engine
from rsd_engine_core import create_rsd_engine

engine = create_rsd_engine(
    main_model=main_model,
    draft_model=draft_model if use_speculative_decoding else None,
    rfsq_head=main_model.rfsq_head,  # 已加载的RFSQ head
    rfsq_decoder=rfsq_model,          # 已加载的RFSQ decoder
    processor=processor,
    device=device,
    chunk_len=8,
    action_dim=7,
)
```

#### D. 修复Episode Loop（第683-743行）

使用`run_episode_with_chunks`替代手动循环：

```python
# 在Task循环中
for trial_idx in range(min(num_trials, len(init_states))):
    try:
        # 创建环境
        env = OffScreenRenderEnv(...)
        env.reset()
        obs = env.set_init_state(init_states[trial_idx])

        # 运行episode
        result = run_episode_with_chunks(
            env=env,
            engine=engine,
            task_description=task_description,
            max_steps=300,
            use_speculative_decoding=use_speculative_decoding,
            verbose=(trial_idx == 0),  # 第一个trial打印详细信息
        )

        # 记录结果
        success = result['success']
        episode_time = result['steps'] * 0.1  # 估算
        episode_inference_time = result['inference_time_ms'] / 1000

        if success:
            task_successes += 1
            total_successes += 1

        total_inference_time += episode_inference_time

        print(f"   Trial {trial_idx + 1}/{num_trials}: "
              f"{'✓' if success else '✗'} "
              f"(steps: {result['steps']}, inf: {result['inference_time_ms']:.1f}ms)")

        env.close()

    except Exception as e:
        print(f"   ❌ Trial {trial_idx + 1} failed: {e}")
        continue
```

#### E. 添加最终统计

```python
# 在evaluation结束后
engine_stats = engine.get_stats()
print(f"\n📊 RSD Engine Statistics:")
print(f"   Total inferences: {engine_stats['total_inferences']}")
print(f"   Avg inference time: {engine_stats['avg_inference_time_ms']:.1f}ms")
if use_speculative_decoding:
    print(f"   Avg draft time: {engine_stats['avg_draft_time_ms']:.1f}ms")
    print(f"   Avg main time: {engine_stats['avg_main_time_ms']:.1f}ms")
    print(f"   Draft acceptance rate: {engine_stats['avg_acceptance_rate']:.1%}")
```

---

## 🔍 关键检查点

### 测试1: 模型加载（Modal image build后）

运行`--num-trials 1`，检查logs：

```
✓ Base OpenVLA-OFT model loaded
✓ RFSQ Decoder loaded (epoch X)
✓ RFSQ head loaded (val_acc: 0.909)
✓ Draft Model loaded (4.7M params)
✓ RSD Inference Engine created
   Hidden size: 4096
   Draft hidden size: 512
   Chunk length: 8
   Action dim: 7
```

### 测试2: 第一次推理（verbose=True）

应该看到：

```
      Input keys: dict_keys(['pixel_values', 'input_ids', ...])
      Pixel values shape: torch.Size([1, 3, 224, 224])
      Hidden state shape: torch.Size([1, 4096])
      Hidden state range: [-2.345, 3.210]
      Draft time: 15.3ms
      Draft tokens shape: torch.Size([1, 3, 1])
      Sample draft tokens: tensor([3, 4, 2])
      Main time: 45.7ms
      Main tokens shape: torch.Size([1, 8, 8, 16])
      Sample main tokens [0,0,0,:5]: tensor([3, 4, 3, 2, 5])
      Acceptance rate: 66.7%
      Decode time: 2.1ms
      Tokens for decoder: torch.Size([1, 8, 16, 8])
      Actions shape: (8, 7)
      Action range: [-0.856, 0.923]
      Sample action[0]: [ 0.234 -0.123  0.456  0.012 -0.234  0.567 -0.890]
```

### 测试3: Episode完成

```
   Trial 1/1: ✓ (steps: 127, inf: 285.3ms)
```

**如果成功**：继续运行更多trials

**如果失败**：根据错误信息调试（参考下面的故障排查）

---

## 🐛 故障排查

### 错误1: "Hidden state extraction failed"

**原因**：OpenVLA的API调用方式不对

**检查**：
```python
# 尝试不同的参数组合
outputs = self.main_model(
    input_ids=inputs['input_ids'],
    pixel_values=inputs['pixel_values'],
    attention_mask=inputs.get('attention_mask'),
    output_hidden_states=True,
)
```

### 错误2: "Draft prediction failed: dimension mismatch"

**原因**：Draft Model训练时的输入格式与推理不一致

**修复**：检查Phase 2训练代码，确认Draft Model期望的输入shape

可能需要调整：
```python
# 如果Draft Model期望flattened input
draft_input = draft_input.view(1, -1)  # Flatten
```

### 错误3: "RFSQ decoding failed: shape mismatch"

**原因**：Token shape转换错误

**检查**：
```python
print(f"Main tokens: {main_tokens.shape}")  # 应该是 [1, 8, 8, 16]
print(f"After permute: {final_tokens_reshaped.shape}")  # 应该是 [1, 8, 16, 8]
```

### 错误4: Actions全是0或NaN

**原因**：RFSQ decoder没有正确加载

**检查**：
```python
# 测试decoder
test_indices = torch.randint(0, 7, (1, 8, 16, 8)).to(device)
test_actions = rfsq_decoder.decode_from_indices(test_indices)
print(f"Test actions range: [{test_actions.min():.3f}, {test_actions.max():.3f}]")
```

### 错误5: 成功率仍然很低（<50%）

**可能原因**：
1. RFSQ head训练质量不好（检查Phase 2的90.9% accuracy是否真实）
2. RFSQ decoder重构误差太大（检查Phase 1的MSE）
3. Action normalization不一致（检查训练时的normalization方式）

**调试**：运行baseline（不使用RFSQ）对比：
```python
# 临时修改：直接用OpenVLA的predict_action
action = main_model.predict_action(
    image, task_description, unnorm_key="libero_spatial"
)
```

如果baseline成功率>90%，说明是RFSQ pipeline的问题。

---

## 📊 预期结果

### 修复后的性能指标

| Metric | Baseline (L1 Regression) | RSD (RFSQ Tokens) | Target |
|--------|-------------------------|-------------------|--------|
| Success Rate | 97.1% | 85-95% | ✅ |
| Inference Time | ~70ms | 45-55ms | ✅ 1.3-1.6x faster |
| Draft Acceptance | N/A | 60-80% | ✅ |
| Memory Usage | High (variable padding) | Low (fixed size) | ✅ |

**关键点**：
- 成功率轻微下降（2-12%）是**正常的**，因为RFSQ量化会有噪声
- 推理速度应该显著提升（如果Draft Model工作正常）
- Acceptance rate越高，加速效果越好

---

## 🎯 测试流程

### 阶段1: 单次测试（验证基本功能）

```bash
modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 1 \
    --use-speculative-decoding False  # 先不用Draft Model
```

**期望**：至少1个trial成功

### 阶段2: 小规模测试（验证稳定性）

```bash
modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 5 \
    --use-speculative-decoding False
```

**期望**：成功率 > 70%

### 阶段3: 启用Speculative Decoding

```bash
modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 5 \
    --use-speculative-decoding True
```

**期望**：
- 成功率与阶段2接近
- 推理时间减少30-50%
- Draft acceptance rate > 50%

### 阶段4: 完整评估

```bash
modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True
```

**期望**：
- 总体成功率 85-95%
- 平均推理时间 45-55ms
- 稳定的acceptance rate

---

## 📝 提交Checklist

修复完成后，确认以下内容：

- [ ] 使用正确的base model (`moojink/openvla-7b-oft-finetuned-libero-spatial`)
- [ ] 导入并使用`rsd_engine_core.py`
- [ ] 删除了错误的`SimpleRSDEngine`实现
- [ ] 使用`run_episode_with_chunks`执行episodes
- [ ] 修复了成功判定逻辑（`info_env.get('success', False)`）
- [ ] 运行`--num-trials 1`验证不crash
- [ ] 运行`--num-trials 5`验证成功率>70%
- [ ] 启用speculative decoding验证加速效果
- [ ] 记录最终的success rate和timing statistics

---

## 🎉 总结

**关键修复**：
1. ✅ 使用fine-tuned模型而不是原版OpenVLA
2. ✅ 实现完整的Speculative Decoding逻辑
3. ✅ 正确的数据流：Hidden States → Token Prediction → RFSQ Decoding
4. ✅ 添加Draft projection layer
5. ✅ 正确的shape转换和chunk执行

**为什么之前失败**：
- 原版OpenVLA不知道如何做LIBERO任务 → 0% success
- 没有使用训练好的RFSQ pipeline → 无法评估RSD
- 错误被隐藏（随机动作） → 看不到真正的问题

**现在应该达到**：
- 85-95% success rate
- 45-55ms inference time
- 1.3-1.6x speedup with speculative decoding

---

**Good luck! 🚀**

如有问题，检查：
1. 模型加载logs
2. 第一次推理的详细输出（verbose=True）
3. Shape mismatch错误
4. RFSQ decoder是否正确工作
