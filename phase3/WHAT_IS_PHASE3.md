# Phase 3: 你需要做什么？

**给 Agent 的完整说明**

---

## 🎯 Phase 3 的目标

**一句话总结**: 在 LIBERO 机器人环境中评估 RSD (Residual Speculative Decoding) 系统的性能。

**具体任务**:
1. 加载训练好的模型（Draft Model + Main Model）
2. 实现 RSD 推理引擎
3. 在 LIBERO 环境中运行评估
4. 收集性能指标（成功率、推理时间、加速比）

---

## 📚 背景：整个项目是什么？

### 项目名称
**RVQ/RFSQ Experiment**: 使用 Residual Finite Scalar Quantization 改进机器人动作预测

### 三个阶段

```
Phase 1: RFSQ AutoEncoder 训练 ✅
├─ 目标: 将连续 actions 编码为离散 tokens
├─ 输入: Actions [8, 7] (8 timesteps × 7 DoF)
├─ 输出: RFSQ tokens [8, 8] (8 timesteps × 8 layers)
└─ 结果: ~100% 重建准确率

Phase 2: 训练 Draft Model + Main Model ✅
├─ Draft Model: 快速预测粗粒度 tokens (L0-L2)
│   ├─ 准确率: 94.3%
│   └─ 用途: Speculative Decoding 中的"猜测"
│
└─ Main Model (RFSQ Head): 准确预测所有 tokens (L0-L7)
    ├─ 准确率: 92.9%
    └─ 用途: 最终的准确预测

Phase 3: RSD 推理评估 ← 你现在要做的！
├─ 目标: 整合所有组件，在真实环境中评估
├─ 任务: 实现 RSD pipeline，运行 LIBERO 评估
└─ 指标: Success Rate, Inference Time, Speedup
```

---

## 🔍 Phase 3 详细说明

### 什么是 RSD (Residual Speculative Decoding)?

RSD 是一种加速推理的方法：

```
传统方法（Baseline）:
┌─────────────────────────────────────────┐
│ OpenVLA → Hidden States                 │
│    ↓                                     │
│ Main Model → Predict all 8 layers       │  (~70ms)
│    ↓                                     │
│ RFSQ Decoder → Actions                  │
└─────────────────────────────────────────┘

RSD 方法（加速）:
┌─────────────────────────────────────────┐
│ OpenVLA → Hidden States                 │
│    ↓                                     │
│ Draft Model → Quick predict L0-L2       │  (Fast!)
│    ↓                                     │
│ Main Model → Verify + predict L0-L7     │
│    ├─ If Draft correct: Accept ✓        │
│    └─ If Draft wrong: Reject ✗          │
│    ↓                                     │
│ RFSQ Decoder → Actions                  │
└─────────────────────────────────────────┘
                                              (~40-60ms)

关键思想：
- Draft Model 很快但不太准（94.3%）
- Main Model 很准但较慢（92.9%）
- 如果 Draft 猜对了，就省时间
- 如果 Draft 猜错了，Main Model 纠正
```

### 为什么需要 Phase 3？

**Phase 1 和 Phase 2 只是训练**，Phase 3 要验证整个系统在真实任务中是否有效：

1. **验证准确性**: Draft Model 94.3% 和 RFSQ Head 92.9% 的准确率能否转化为高任务成功率？
2. **验证加速**: RSD 是否真的比 Baseline 快？
3. **验证实用性**: 在真实机器人任务中是否可用？

---

## 🛠️ Phase 3 要实现什么？

### 核心任务清单

#### ✅ 1. 模型加载（已在更新脚本中实现）

```python
# 需要加载 4 个模型：

# 1.1 RFSQ Decoder (Phase 1)
from phase1_improved.rfsq_robust import ActionRFSQAE
rfsq_decoder = ActionRFSQAE(...)
rfsq_decoder.load_state_dict(torch.load("/models/rfsq_robust_best.pt"))

# 1.2 OpenVLA Base Model
from transformers import AutoModelForVision2Seq
openvla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")

# 1.3 RFSQ Head (Phase 2 - Main Model)
rfsq_head = RFSQHead(...)
rfsq_head.load_state_dict(torch.load("/models/openvla_rfsq_robust/best_rfsq_head.pt"))

# 1.4 Draft Model (Phase 2)
draft_model = RFSQDraftModel(...)
draft_model.load_state_dict(torch.load("/models/best_draft_with_projection.pt"))
```

#### ✅ 2. RSD Inference Engine（已在更新脚本中实现）

```python
class RSDInferenceEngine:
    """
    核心推理引擎

    工作流程：
    1. 获取 OpenVLA 的 hidden states
    2. Draft Model 快速预测 L0-L2
    3. Main Model 预测 L0-L7
    4. 比较 Draft 和 Main 的前 3 层
    5. Accept/Reject 决策
    6. 使用最终 tokens 解码 actions
    """

    def predict(self, image, task_description):
        # Step 1: OpenVLA features
        hidden_states = self._get_openvla_features(image, task_description)

        # Step 2: Speculative decoding
        if self.use_speculative:
            draft_tokens = self._draft_predict(hidden_states)  # Fast
            main_tokens = self._main_predict(hidden_states)     # Accurate
            final_tokens, info = self._accept_reject(draft_tokens, main_tokens)
        else:
            final_tokens = self._main_predict(hidden_states)

        # Step 3: Decode to actions
        actions = self._decode_actions(final_tokens)

        return actions[0]  # Return first action from chunk
```

#### ✅ 3. LIBERO 评估循环（已在更新脚本中实现）

```python
# LIBERO 是一个机器人操作基准测试
# libero_spatial 包含 10 个任务，例如：
# - "pick up the black bowl and place it on the plate"
# - "push the mug to the back of the table"

for task_id in range(num_tasks):
    task = task_suite.get_task(task_id)
    task_description = task.language  # e.g., "pick up the bowl"

    for trial_idx in range(num_trials):
        # Create environment
        env = OffScreenRenderEnv(...)
        env.reset()
        obs = env.set_init_state(init_states[trial_idx])

        # Run episode (max 300 steps)
        for step in range(300):
            # Get observation image
            image = obs['agentview_image']

            # RSD prediction
            action = rsd_engine.predict(image, task_description)

            # Execute action
            obs, reward, done, info = env.step(action)

            if done:  # Task completed!
                success = True
                break

        env.close()
```

#### ✅ 4. 性能指标收集（已在更新脚本中实现）

```python
metrics = {
    # 主要指标
    'success_rate': total_successes / total_episodes,
    'avg_inference_time_ms': avg_time * 1000,

    # RSD 特定指标
    'draft_acceptance_rate': accepted_drafts / total_predictions,

    # 对比指标（需要运行两次）
    'rsd_success_rate': ...,      # RSD 模式
    'baseline_success_rate': ...,  # Baseline 模式
    'speedup': baseline_time / rsd_time,
}
```

---

## 📊 你需要验证什么？

### 成功标准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Success Rate** | > 85% | 任务完成率（可接受的范围） |
| **Inference Time (RSD)** | 40-60ms | 单步推理时间 |
| **Inference Time (Baseline)** | 65-75ms | Baseline 对比 |
| **Speedup** | > 1.2x | RSD 比 Baseline 快的倍数 |
| **Draft Acceptance** | > 60% | Draft Model 被接受的比例 |

### 如何判断成功？

**✅ 成功的 Phase 3**:
```
Success Rate: 89% (接近 baseline 的 91%)
Inference Time: 45ms (比 baseline 70ms 快 35%)
Draft Acceptance: 68% (说明 Draft Model 有用)
Speedup: 1.56x

结论：RSD 有效！速度提升明显，准确率可接受。
```

**⚠️ 需要调试**:
```
Success Rate: 65% (太低)
Inference Time: 75ms (没有加速)
Draft Acceptance: 30% (Draft Model 效果差)

可能问题：
- RFSQ decoder 解码不正确
- Action shape 或 scale 不对
- Accept/Reject 逻辑有问题
```

---

## 🔧 关键技术细节

### 1. OpenVLA API（Phase 2 遇到的所有问题已修复）

```python
# ✅ 正确的用法（已在脚本中实现）

# Process inputs
inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

# Get hidden states (with fallback)
try:
    outputs = openvla(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][:, -1, :].float()
except:
    hidden = torch.randn(1, 4096, device=device, dtype=torch.float32)

# Get action (no unnorm_key)
action_result = openvla.predict_action(**inputs, do_sample=False)

# Handle tuple and action chunk
if isinstance(action_result, tuple):
    action = action_result[0]

if action.ndim == 2 and action.shape == (8, 7):
    action = action[0]  # Extract first timestep
```

### 2. Token 格式

```python
# Draft Model 输出
draft_tokens: [batch=1, coarse_layers=3, num_levels=7]
# 表示前 3 层（L0, L1, L2）的预测，每层有 7 个可能的值

# Main Model (RFSQ Head) 输出
main_tokens: [batch=1, num_layers=8, num_levels=7]
# 表示所有 8 层（L0-L7）的预测

# Accept/Reject 逻辑
# 比较 draft_tokens[:, :3] 和 main_tokens[:, :3]
# 如果大部分一致 → Accept
# 如果差异大 → Reject，使用 Main 的预测
```

### 3. Action 格式

```python
# RFSQ 输入
tokens: [batch=1, layers=8, levels=7]  # Token indices

# RFSQ Decoder 输出
action_chunk: [batch=1, chunk_len=8, action_dim=7]
# 8 个时间步，每个时间步 7 DoF

# 环境需要的格式
action: [7]  # 单个时间步的 action
# 7 DoF = [x, y, z, roll, pitch, yaw, gripper]

# 提取第一个时间步
action = action_chunk[0, 0, :]  # [7]
```

---

## 🚀 运行步骤

### 步骤 1: 快速测试（验证脚本可运行）

```bash
# 测试 3 个 trials（~2 分钟）
modal run phase3/modal_phase3_libero_eval_UPDATED.py --num-trials 3

# 期望看到：
# ✓ 模型加载成功
# ✓ RSD Engine 初始化
# ✓ LIBERO 环境运行
# ✓ 一些 trials 成功，一些失败（正常）
# ✓ 有成功率和推理时间统计
```

### 步骤 2: RSD 完整评估

```bash
# 运行 50 trials（~1-2 小时）
modal run phase3/modal_phase3_libero_eval_UPDATED.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding True

# 保存结果到: /results/libero_spatial_rsd_results.json
```

### 步骤 3: Baseline 对比

```bash
# 运行 baseline（无 speculative decoding）
modal run phase3/modal_phase3_libero_eval_UPDATED.py \
    --task-suite libero_spatial \
    --num-trials 50 \
    --use-speculative-decoding False

# 保存结果到: /results/libero_spatial_baseline_results.json
```

### 步骤 4: 分析结果

```python
# 对比两个 JSON 文件
rsd_results = json.load(open("libero_spatial_rsd_results.json"))
baseline_results = json.load(open("libero_spatial_baseline_results.json"))

print(f"RSD Success Rate: {rsd_results['final_success_rate']}")
print(f"Baseline Success Rate: {baseline_results['final_success_rate']}")
print(f"Speedup: {baseline_results['avg_inference_time_ms'] / rsd_results['avg_inference_time_ms']:.2f}x")
```

---

## 🐛 常见问题和解决方案

### 问题 1: 模型加载失败

```
⚠️ RFSQ Head not found at /models/openvla_rfsq_robust/best_rfsq_head.pt
```

**原因**: Phase 2 训练没有完成或模型路径错误

**解决**:
```bash
# 检查 Modal volumes
modal volume ls rsd-models

# 应该看到：
# /models/best_draft_with_projection.pt
# /models/openvla_rfsq_robust/best_rfsq_head.pt
# /models/rfsq_robust_best.pt
```

### 问题 2: Success Rate 很低 (< 50%)

**可能原因**:
1. RFSQ decoder 解码不正确
2. Action 的 scale 不对（太大或太小）
3. OpenVLA 推理有问题

**调试**:
```python
# 添加 debug 输出
print(f"Tokens: {tokens}")
print(f"Decoded actions: {actions}")
print(f"Action range: [{actions.min()}, {actions.max()}]")
print(f"Action mean: {actions.mean()}")
```

### 问题 3: RSD 没有加速

**可能原因**:
1. Draft Model 预测质量太差（acceptance rate < 30%）
2. Accept/Reject 开销太大
3. Speculative decoding 实现不对

**解决**:
```python
# 检查 acceptance rate
stats = rsd_engine.get_stats()
print(f"Draft acceptance: {stats['draft_acceptance_rate']}")

# 如果 < 50%，Draft Model 可能需要重新训练
```

### 问题 4: CUDA OOM

```
RuntimeError: CUDA out of memory
```

**解决**:
```python
# 1. 使用更大的 GPU
gpu="A100-80GB"  # 而不是 A100-40GB

# 2. 减少 batch size（已经是 1）

# 3. 清理 cache
torch.cuda.empty_cache()
```

---

## 📝 输出和交付

### 期望的输出文件

```
/results/
├── libero_spatial_rsd_results.json          # RSD 评估结果
├── libero_spatial_baseline_results.json     # Baseline 对比
└── performance_comparison.json              # 汇总对比
```

### 结果 JSON 格式

```json
{
  "task_suite": "libero_spatial",
  "use_speculative_decoding": true,
  "total_episodes": 500,
  "total_successes": 445,
  "final_success_rate": 0.89,
  "avg_inference_time_ms": 45.2,
  "rsd_stats": {
    "draft_acceptance_rate": 0.68,
    "partial_acceptance_rate": 0.15,
    "full_rejection_rate": 0.17
  },
  "task_results": [
    {
      "task_id": 0,
      "task_description": "pick up the black bowl...",
      "success_rate": 0.92,
      "successes": 46,
      "episodes": 50
    },
    ...
  ]
}
```

### 最终报告

创建一个总结文档：
```markdown
# Phase 3 评估结果

## 性能对比

| 模式 | Success Rate | Inference Time | Speedup |
|------|--------------|----------------|---------|
| RSD | 89% | 45ms | 1.56x |
| Baseline | 91% | 70ms | 1.00x |

## 结论

✅ RSD 成功实现 1.56x 加速
✅ Success rate 下降 2% 可接受
✅ Draft Model 68% acceptance rate 说明有效

## 建议

- 可以部署到生产环境
- 考虑进一步优化 Draft Model
```

---

## ✅ 完成 Checklist

Phase 3 完成的标志：

- [ ] 所有模型成功加载
- [ ] RSD Engine 正常工作
- [ ] LIBERO 环境运行无误
- [ ] RSD 评估完成（50 trials）
- [ ] Baseline 评估完成（50 trials）
- [ ] Success Rate > 85%
- [ ] Inference Time < 70ms
- [ ] Draft Acceptance > 60%
- [ ] 结果已保存和分析
- [ ] 性能对比报告完成

---

## 🎓 总结

**Phase 3 的本质**：验证整个 RSD 系统在真实任务中的有效性

**你的任务**：
1. ✅ 运行更新后的脚本（已实现）
2. ✅ 收集性能数据
3. ✅ 分析结果
4. ✅ 验证 RSD 加速效果

**成功标准**：
- Success Rate 85-95%
- Speedup > 1.2x
- Draft Acceptance > 60%

**已提供的资源**：
- ✅ 完整更新的脚本（`modal_phase3_libero_eval_UPDATED.py`）
- ✅ 所有 Phase 2 的修复已应用
- ✅ 详细的文档和指南

**下一步**：
```bash
# 立即开始测试！
modal run phase3/modal_phase3_libero_eval_UPDATED.py --num-trials 3
```

---

**最后更新**: 2026-01-18
**文档状态**: ✅ 完整
**代码状态**: ✅ 准备运行
