# Mock Test Suite for Phase 3 Integration

## 🎯 目的

在昂贵的A100 GPU上大规模训练之前，先在便宜的GPU（甚至CPU）上测试整个pipeline的集成，避免在最后LIBERO评测阶段才发现问题。

## 📊 测试内容

### 1. Checkpoint格式验证
- ✅ 验证Phase 1 (RFSQ) checkpoint能正确加载
- ✅ 验证Phase 2 (Draft Model) checkpoint能正确加载
- ✅ 验证Phase 2 (Main Model RFSQ Head) checkpoint能正确加载

### 2. 数据流测试
- ✅ Mock OpenVLA hidden state (4096-dim) → Draft Model
- ✅ Draft Model → Coarse tokens (L0-L2)
- ✅ Mock OpenVLA hidden state → Main Model RFSQ Head
- ✅ Main Model → All tokens (L0-L7)
- ✅ Token comparison & acceptance
- ✅ Final tokens → RFSQ Decoder
- ✅ RFSQ Decoder → Actions (7-dim)

### 3. Shape匹配验证
- ✅ 所有中间tensor的shape正确
- ✅ 没有维度mismatch错误
- ✅ Reshape操作正确

### 4. LIBERO集成（可选）
- ✅ LIBERO库能正确导入
- ✅ 环境初始化正常
- ✅ Action能被环境接受

## 🚀 快速开始

### 方法1：一键运行（推荐）

```bash
# Clone repo
git clone https://github.com/Oscar-Ge/RVQExperiment.git
cd RVQExperiment

# 运行完整测试
bash mock_test/run_integration_test.sh
```

### 方法2：分步运行

```bash
# Step 1: 生成mock checkpoints
python mock_test/generate_mock_checkpoints.py --output-dir ./mock_models

# 输出示例：
# ✅ Saved to: ./mock_models/rfsq_robust_best.pt (Mock MSE: 0.010000)
# ✅ Saved to: ./mock_models/best_draft_with_projection.pt (Mock Accuracy: 0.915)
# ✅ Saved to: ./mock_models/openvla_rfsq_robust/best_rfsq_head.pt (Mock Accuracy: 0.925)

# Step 2: 运行集成测试
python mock_test/test_phase3_integration.py --models-dir ./mock_models

# 输出示例：
# 📦 Test 1: Checkpoint Loading
#    ✅ All checkpoints loaded successfully!
# 🔬 Test 2: RSD Pipeline Integration
#    ✅ Pipeline test passed!

# Step 3: (可选) 测试LIBERO集成
python mock_test/test_phase3_integration.py --models-dir ./mock_models --test-libero
```

### 方法3：真实LIBERO测试（推荐用于SSH租GPU）

```bash
# SSH到你租的GPU后
# 1. 先安装LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# 2. 运行真实LIBERO测试（用Mock OpenVLA）
python mock_test/test_libero_with_mock_openvla.py \
    --models-dir ./mock_models \
    --device cuda \
    --num-episodes 3 \
    --task-id 0
```

**为什么这样测试？**
- ✅ 用真实LIBERO（测试最容易出问题的环境集成）
- ✅ 用Mock OpenVLA（避免加载7B模型，节省显存~14GB）
- ✅ 验证RSD pipeline在真实环境中能正常运行
- ✅ 不需要昂贵的A100（便宜GPU如T4即可）

## 📁 文件结构

```
mock_test/
├── README.md                           # 本文件
├── generate_mock_checkpoints.py        # 生成假checkpoint
├── test_phase3_integration.py          # 基础集成测试（不需要LIBERO）
├── test_libero_with_mock_openvla.py    # 真实LIBERO测试（推荐！）
└── run_integration_test.sh             # 一键运行脚本
```

## 🧪 测试输出示例

### 成功的输出

```
============================================================
🧪 Phase 3 Integration Test Suite
============================================================
Models directory: /path/to/mock_models
Device: cpu

============================================================
📦 Test 1: Checkpoint Loading
============================================================

1️⃣ Loading RFSQ Decoder...
   ✅ Loaded (MSE: 0.01)

2️⃣ Loading Draft Model...
   ✅ Loaded (Accuracy: 0.915)

3️⃣ Loading RFSQ Head...
   ✅ Loaded (Accuracy: 0.925)

✅ All checkpoints loaded successfully!

============================================================
🔬 Test 2: RSD Pipeline Integration
============================================================

🧪 Test 2.1: With Speculative Decoding
============================================================
🚀 RSD Inference Engine - Mock Test
============================================================
Input shape: torch.Size([1, 4096])
Use speculative decoding: True

1️⃣ Draft Model Prediction:
   Logits shape: torch.Size([1, 3, 128, 7])
   Tokens shape: torch.Size([1, 3, 128])
   Time: 12.34ms

2️⃣ Main Model Prediction:
   Logits shape: torch.Size([1, 8, 128, 7])
   Tokens shape: torch.Size([1, 8, 128])
   Time: 23.45ms

3️⃣ Token Comparison:
   Acceptance rate: 85.2%
   Layer 0: 87.5%
   Layer 1: 84.4%
   Layer 2: 83.6%

4️⃣ Token Reshaping:
   After reshape: torch.Size([1, 8, 16, 8])

5️⃣ RFSQ Decoding:
   Actions shape: torch.Size([1, 8, 7])
   Time: 3.21ms

6️⃣ Final Output:
   Actions shape: (8, 7)
   Actions range: [-0.523, 0.487]
   Total time: 45.67ms
============================================================

✅ Pipeline test passed!

============================================================
✅ All Tests Passed!
============================================================

🎯 Next steps:
   1. If all tests pass locally, deploy to Modal with real checkpoints
   2. Run actual Phase 3 evaluation with LIBERO
   3. Monitor for any integration issues
```

### 失败的输出（示例）

```
============================================================
📦 Test 1: Checkpoint Loading
============================================================

1️⃣ Loading RFSQ Decoder...
❌ RFSQ checkpoint not found: ./mock_models/rfsq_robust_best.pt

   Please run: python mock_test/generate_mock_checkpoints.py --output-dir ./mock_models

❌ Test 1 failed: Could not load checkpoints
```

## 🔍 常见问题

### Q1: 为什么需要mock测试？

**A**: 上次实验在最后LIBERO评测阶段才发现集成问题，导致前面的训练白费。Mock测试可以提前发现：
- Checkpoint格式不兼容
- Shape mismatch
- 数据流断裂
- 模型加载失败

### Q2: Mock checkpoint和真实checkpoint有什么区别？

**A**:
- **格式相同**：完全相同的state_dict结构
- **权重不同**：Mock使用随机初始化，真实使用训练好的权重
- **精度不同**：Mock输出是随机的，真实会有正确的预测

**关键**：Mock测试的是**集成**，不是**精度**。

### Q3: 测试通过后还会出问题吗？

**A**: Mock测试只能保证**集成正确**，不能保证：
- 训练收敛（需要真实训练验证）
- 模型精度（需要真实checkpoint）
- LIBERO任务成功率（需要完整评测）

但至少可以确保：
- ✅ 代码不会crash
- ✅ Shape都匹配
- ✅ Pipeline能跑通

### Q4: 如果Mock测试失败怎么办？

**A**: 这正是Mock测试的价值！在A100训练之前就发现问题。

检查：
1. 是否正确生成了mock checkpoints？
2. Checkpoint路径是否正确？
3. 模型定义是否和checkpoint匹配？
4. 查看错误信息，定位具体问题

### Q5: 需要GPU吗？

**A**: **不需要**！Mock测试可以在CPU上运行。

```bash
# CPU运行（默认）
python mock_test/test_phase3_integration.py --device cpu

# GPU运行（如果有）
python mock_test/test_phase3_integration.py --device cuda
```

## 📊 预期时间消耗

| 步骤 | CPU | GPU (T4) | GPU (A100) |
|------|-----|----------|------------|
| 生成Mock Checkpoints | 5s | 2s | 1s |
| 集成测试 | 30s | 10s | 5s |
| **总计** | **35s** | **12s** | **6s** |

**对比真实训练**：
- Phase 1训练: ~2-3小时
- Phase 2训练: ~6-10小时
- Phase 3评测: ~2-3小时
- **总计**: ~10-16小时

**节省**: Mock测试只需35秒，可以避免浪费10-16小时！

## 🎯 测试覆盖率

| 组件 | 测试内容 | 覆盖率 |
|------|---------|--------|
| **Phase 1 RFSQ** | Checkpoint加载、decode功能 | ✅ 100% |
| **Phase 2 Draft** | Checkpoint加载、前向推理 | ✅ 100% |
| **Phase 2 Main** | Checkpoint加载、前向推理 | ✅ 100% |
| **Token合并** | Draft + Main token comparison | ✅ 100% |
| **RFSQ Decoder** | Indices → Actions解码 | ✅ 100% |
| **Shape匹配** | 所有中间tensor | ✅ 100% |
| **LIBERO集成** | 环境初始化 | ⚠️ 简化测试 |

## 🚨 注意事项

### 1. Mock vs 真实训练

Mock测试**不能替代**真实训练，只是**前置检查**。

```
Mock测试 → 集成验证 ✅
   ↓
真实训练 → 精度验证 ✅
   ↓
LIBERO评测 → 任务成功率 ✅
```

### 2. Checkpoint兼容性

Mock checkpoints使用和真实训练**完全相同**的模型定义。如果真实训练修改了模型结构，需要同步更新mock generator。

### 3. LIBERO环境

LIBERO测试需要：
- MuJoCo
- robosuite
- LIBERO benchmark

在本地可能无法运行，但可以在Modal环境测试。

## 📖 后续步骤

### 如果Mock测试通过 ✅

1. **部署到Modal**：上传真实代码到Modal
2. **Phase 1训练**：训练Robust RFSQ
3. **Phase 2训练**：训练Draft Model + Main Model
4. **Phase 3评测**：完整LIBERO评测
5. **监控**：密切关注是否有集成问题

### 如果Mock测试失败 ❌

1. **不要继续训练**！先修复集成问题
2. **查看错误信息**：定位具体问题
3. **修复代码**：修复shape mismatch或逻辑错误
4. **重新测试**：确保所有测试通过
5. **再部署训练**

## 🔗 相关文档

- **Phase 1 Improved**: `../phase1_improved/AGENT_GUIDE.md`
- **Phase 2 Draft Retrain**: `../phase2_draft_retrain/README.md`
- **Phase 3 Evaluation**: `../phase3/QUICK_START.md`
- **Migration Guide**: `../MIGRATION_TO_ROBUST_RFSQ.md`
- **Agent Action Plan**: `../AGENT_ACTION_PLAN.md`

---

**准备好了吗？开始Mock测试！**

```bash
bash mock_test/run_integration_test.sh
```
