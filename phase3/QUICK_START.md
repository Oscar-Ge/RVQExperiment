# Phase 3 Quick Start Guide

## ✅ 已完成的工作

所有Phase 3代码和文档已复制到 `RVQExperiment/phase3/` 并推送到GitHub！

---

## 📁 文件清单

```
phase3/
├── modal_phase3_libero_eval.py       # 主评估脚本（80%完成）
├── rsd_inference_engine.py           # RSD推理引擎（100%完成）
├── AGENT_INSTRUCTIONS.md             # 给实验Agent的详细指令
├── USER_INSTRUCTIONS.md              # 给你的指令（如何指导Agent）
├── PHASE3_EXPERIMENT_GUIDE.md        # 完整实验指南
├── PHASE3_README.md                  # 快速参考
├── LIBERO_SOLUTION.md                # LIBERO集成问题解决方案
├── COMPLETED_WORK_SUMMARY.md         # 工作总结
└── QUICK_START.md                    # 本文件
```

---

## 🚀 如何使用（3步走）

### Step 1: 给Agent的初始指令

复制以下内容发给你的实验Agent：

```
我需要你帮我完成Phase 3的LIBERO评估。请按以下步骤操作：

1. Clone这个仓库：
   git clone https://github.com/Oscar-Ge/RVQExperiment.git
   cd RVQExperiment/phase3

2. 仔细阅读两个文件：
   - AGENT_INSTRUCTIONS.md（你的详细工作指南）
   - PHASE3_EXPERIMENT_GUIDE.md（背景和技术细节）

3. 实现modal_phase3_libero_eval.py中的4个TODO：
   - Task 1: 加载Main Model (OpenVLA-OFT-RFSQ) - 第170行附近
   - Task 2: 加载Draft Model - 第190行附近
   - Task 3: 集成RSD Engine - 第250行附近
   - Task 4: 实现LIBERO环境循环 - 第300行附近

4. 使用debug模式测试：
   modal run modal_phase3_libero_eval.py --task-suite libero_spatial --num-trials 1

5. 修复所有错误，然后运行完整评估：
   modal run modal_phase3_libero_eval.py --task-suite libero_spatial --num-trials 50

请先确认你理解任务，然后开始实现。
```

### Step 2: 使用USER_INSTRUCTIONS.md指导Agent

当Agent遇到问题时，查阅 `USER_INSTRUCTIONS.md`，它包含：
- ✅ 如何回答Agent的问题
- ✅ 常见错误的解决方案
- ✅ 如何解释结果
- ✅ 调试工作流程

### Step 3: 验证结果

Agent完成后，检查：
- [ ] Success rate 80-95%
- [ ] Inference time < 60ms
- [ ] 所有10个任务都评估了
- [ ] 结果保存到了Modal volume

---

## 🔧 LIBERO问题解决方案

### 问题：为什么每次clone LIBERO都是空的？

**答案**：不要clone LIBERO到你的项目中！

**解决方案**：已在Modal image构建时自动处理
- ✅ LIBERO在Modal镜像构建时自动clone
- ✅ torch.load bug自动修复
- ✅ 不需要手动操作
- ✅ 修改不会丢失（在镜像中持久化）

详见：`LIBERO_SOLUTION.md`

---

## 📊 期望结果

### 成功的评估结果应该是：

```
============================================================
🎉 EVALUATION COMPLETE!
============================================================
   Task Suite: libero_spatial
   Total Episodes: 500
   Total Successes: 425-475
   Success Rate: 85.0% - 95.0%
   Avg Inference Time: 45-55 ms
   Speculative Decoding: True
============================================================
```

### 与Baseline对比

| Metric | Baseline (OpenVLA-OFT) | RSD (Expected) | Δ |
|--------|----------------------|----------------|---|
| Success Rate | 97.1% | 85-95% | -2~-12% |
| Inference Time | ~70ms | 45-55ms | **1.3-1.6x faster** |
| Batch Scalability | Poor (padding) | Excellent (fixed) | 🚀 |

---

## 🎯 Agent工作量估算

- **阅读文档**: 10-15分钟
- **实现4个TODO**: 1-1.5小时
- **Debug和测试**: 1-2小时
- **运行完整评估**: 2-3小时（GPU时间）

**总计**: 约4-6小时agent工作时间 + 2-3小时GPU时间

---

## 💡 重要提示

### 给Agent的提示

1. **从小测试开始**：`--num-trials 1`
2. **逐步增加**：1 → 5 → 10 → 50
3. **检查Modal logs**：`modal app logs rsd-phase3-libero-eval`
4. **一次实现一个TODO**：不要一次做所有

### 给你的提示

1. **耐心**：Agent可能需要几次迭代才能成功
2. **清晰**：使用USER_INSTRUCTIONS.md中的具体命令
3. **鼓励**：Agent做对时给予肯定
4. **调试**：使用USER_INSTRUCTIONS.md的调试流程

---

## 🚨 常见问题速查

### Q1: Agent说"cannot import prismatic"
**A**: 查看AGENT_INSTRUCTIONS.md的"Common Issues"部分

### Q2: Agent说"CUDA out of memory"
**A**: 启用4-bit quantization（USER_INSTRUCTIONS.md有代码）

### Q3: Agent说"LIBERO environment fails"
**A**: 检查torch.load fix是否applied（已在Modal image中）

### Q4: Success rate太低（<70%）
**A**: 检查：
- 动作是否正确归一化？
- RFSQ解码是否正确？
- Task description是否匹配？

---

## 📞 获取帮助

如果Agent完全卡住：

1. **检查文档**：
   - AGENT_INSTRUCTIONS.md（Agent端）
   - USER_INSTRUCTIONS.md（你的端）
   - PHASE3_EXPERIMENT_GUIDE.md（技术细节）

2. **简化任务**：
   - 只实现Task 1
   - 测试Task 1
   - 然后再做Task 2

3. **查看示例代码**：
   所有TODO都在AGENT_INSTRUCTIONS.md中有完整实现示例

---

## ✅ 成功标志

你知道成功了当：

- ✅ Agent能运行评估而不crash
- ✅ Success rate在80-95%范围内
- ✅ Inference time < 60ms
- ✅ 结果保存到Modal volume
- ✅ Experiment在Orchestra中标记为completed

---

## 🎉 下一步

完成Phase 3后：

1. **分析结果**：比较RSD vs Baseline
2. **Day 9**: 多模态歧义性测试
3. **Day 10**: 写论文和准备图表

---

**祝你好运！🚀**

所有代码和文档都已准备就绪。只需要Agent实现那~110行代码！
