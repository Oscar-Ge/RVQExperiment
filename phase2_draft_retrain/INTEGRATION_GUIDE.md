# Draft Model集成到Phase 3指南

## 🎯 目标

训练完成后，将新的Draft Model（带projection layer）集成到Phase 3的RSD Inference Engine中。

---

## ✅ 训练完成检查清单

在集成之前，确认训练成功：

```bash
# 检查checkpoint是否存在
modal volume ls rsd-models | grep best_draft_with_projection.pt

# 下载checkpoint验证
modal volume get rsd-models best_draft_with_projection.pt ./

# 检查checkpoint内容
python -c "
import torch
ckpt = torch.load('best_draft_with_projection.pt', weights_only=False)
print('Keys:', ckpt.keys())
print('Val Accuracy:', ckpt['val_accuracy'])
print('Has projection:', any('projection' in k for k in ckpt['model_state_dict'].keys()))
"
```

**期望输出**：
```
Keys: dict_keys(['model_state_dict', 'optimizer_state_dict', 'epoch', 'val_accuracy', ...])
Val Accuracy: 0.892
Has projection: True
```

---

## 🔧 集成步骤

### Step 1: 更新`rsd_engine_core.py`

**位置**：`phase3/rsd_engine_core.py`

**修改内容**：

#### A. 添加Draft Model定义

在文件开头添加Draft Model的定义（从训练脚本复制）：

```python
# 在rsd_engine_core.py开头添加

class DraftTransformerDecoder(nn.Module):
    """Transformer Decoder for Draft Model"""

    def __init__(self, hidden_dim=512, num_heads=8, feedforward_dim=2048, max_seq_length=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,  # 推理时dropout=0
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.position_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_dim) * 0.02
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        pos_enc = self.position_encoding[:, :seq_len, :]
        hidden_states = hidden_states + pos_enc
        output = self.decoder_layer(hidden_states, hidden_states)
        return self.output_norm(output)


class RFSQDraftModelWithProjection(nn.Module):
    """Draft Model with Projection Layer"""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=512,
        num_coarse_layers=3,
        chunk_len=8,
        action_hidden_dim=16,
        grid_size=7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_coarse_layers = num_coarse_layers
        self.chunk_len = chunk_len
        self.action_hidden_dim = action_hidden_dim
        self.grid_size = grid_size

        # Projection Layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Decoder
        self.decoder = DraftTransformerDecoder(hidden_dim=hidden_dim)

        # Classification Heads
        output_size_per_head = chunk_len * action_hidden_dim * grid_size
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_size_per_head),
            )
            for _ in range(num_coarse_layers)
        ])

    def forward(self, openvla_hidden_states):
        """
        Args:
            openvla_hidden_states: [Batch, 4096]

        Returns:
            logits: [Batch, 3, 128, 7]
        """
        batch_size = openvla_hidden_states.shape[0]

        # Project
        projected = self.input_projection(openvla_hidden_states)
        x = projected.unsqueeze(1)

        # Decode
        decoder_output = self.decoder(x).squeeze(1)

        # Classify
        layer_outputs = []
        for head in self.classification_heads:
            logits = head(decoder_output)
            logits = logits.view(batch_size, 128, 7)
            layer_outputs.append(logits)

        return torch.stack(layer_outputs, dim=1)
```

#### B. 删除随机初始化的Projection

在`RSDInferenceEngine.__init__`中，删除这几行：

```python
# ❌ 删除这些行
# self.draft_projection = nn.Linear(
#     self.hidden_size,
#     self.draft_hidden_size
# ).to(device)
# self.draft_projection.eval()
```

因为新的Draft Model自带训练好的projection。

#### C. 更新Draft预测逻辑

在`generate_action`方法的Draft预测部分：

```python
# 旧代码（删除）
# draft_input = self.draft_projection(last_hidden_state)
# draft_input = draft_input.unsqueeze(1)

# 新代码（直接传4096维）
draft_logits = self.draft_model(last_hidden_state)  # [1, 3, 128, 7]
```

Draft Model会自动处理projection。

---

### Step 2: 更新`modal_phase3_libero_eval.py`

**位置**：`phase3/modal_phase3_libero_eval.py`

#### A. 导入Draft Model定义

在文件开头添加：

```python
from rsd_engine_core import (
    RSDInferenceEngine,
    run_episode_with_chunks,
    RFSQDraftModelWithProjection,  # 新增
    DraftTransformerDecoder,        # 新增
)
```

#### B. 加载Draft Model

修改Draft Model加载部分（约第420-520行）：

```python
# 旧代码（删除整个原来的Draft Model定义）

# 新代码
if use_speculative_decoding:
    draft_model_path = "/models/best_draft_with_projection.pt"
    print(f"\n   Loading Draft Model (with projection) from {draft_model_path}")

    try:
        # 创建模型
        draft_model = RFSQDraftModelWithProjection(
            input_dim=4096,
            hidden_dim=512,
            num_coarse_layers=3,
            chunk_len=8,
            action_hidden_dim=16,
            grid_size=7,
        )

        # 加载权重
        if Path(draft_model_path).exists():
            checkpoint = torch.load(draft_model_path, map_location=device, weights_only=False)

            draft_model.load_state_dict(checkpoint['model_state_dict'])
            draft_model = draft_model.to(device)
            draft_model.eval()

            print(f"   ✅ Draft Model loaded (val_acc: {checkpoint.get('val_accuracy', 'unknown'):.3f})")
            print(f"   ✅ Projection layer included: {hasattr(draft_model, 'input_projection')}")
        else:
            print(f"   ❌ Draft Model not found at {draft_model_path}")
            draft_model = None

    except Exception as e:
        print(f"   ❌ Failed to load draft model: {e}")
        import traceback
        traceback.print_exc()
        draft_model = None
else:
    draft_model = None
    print(f"\n   Skipping Draft Model (speculative decoding disabled)")
```

---

### Step 3: 测试集成

运行验证测试：

```bash
# Test 1: 验证模型加载
modal run modal_phase3_libero_eval.py --num-trials 1 --use-speculative-decoding True

# 期望输出：
#    ✅ Draft Model loaded (val_acc: 0.892)
#    ✅ Projection layer included: True
#    ✅ RSD Inference Engine created
```

**检查logs**：

```
      Draft time: 12.3ms
      Draft tokens shape: torch.Size([1, 3, 128])
      Main time: 45.7ms
      Main tokens shape: torch.Size([1, 8, 8, 16])
      Acceptance rate: 72.5%
      ✅ Speculative Decoding working!
```

---

## 🧪 验证测试

### Test 1: 单次推理测试

```python
# test_draft_integration.py
import torch
from rsd_engine_core import RFSQDraftModelWithProjection

def test_draft_model():
    """测试Draft Model可以正确加载和推理"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    draft_model = RFSQDraftModelWithProjection(
        input_dim=4096,
        hidden_dim=512,
        num_coarse_layers=3,
    )

    checkpoint = torch.load('best_draft_with_projection.pt', map_location=device)
    draft_model.load_state_dict(checkpoint['model_state_dict'])
    draft_model = draft_model.to(device)
    draft_model.eval()

    print("✅ Model loaded")

    # 2. Test forward
    test_input = torch.randn(1, 4096).to(device)

    with torch.no_grad():
        output = draft_model(test_input)

    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Expected: [1, 3, 128, 7]")

    assert output.shape == (1, 3, 128, 7), f"Shape mismatch! Got {output.shape}"

    # 3. Check projection
    assert hasattr(draft_model, 'input_projection')
    print(f"✅ Projection exists: {draft_model.input_projection.weight.shape}")

    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_draft_model()
```

### Test 2: Phase 3集成测试

```bash
# 运行5个trials测试
modal run modal_phase3_libero_eval.py \
    --task-suite libero_spatial \
    --num-trials 5 \
    --use-speculative-decoding True
```

**期望结果**：
- Success rate: >80%
- Inference time: 45-55ms（相比baseline的70ms）
- Draft acceptance rate: 60-80%

---

## 📊 性能对比

集成后，应该看到：

| Metric | Phase 3 (无Draft) | Phase 3 (新Draft) | 改进 |
|--------|------------------|------------------|------|
| Success Rate | 85-95% | 85-95% | ✅ 保持 |
| Inference Time | ~70ms | 45-55ms | ✅ 1.3-1.6x faster |
| Draft Acceptance | N/A | 60-80% | ✅ 有效加速 |
| GPU Memory | 14GB | 14GB | ✅ 相同 |

---

## 🚨 故障排查

### 问题1: "Draft Model加载失败"

**检查**：
```bash
# Checkpoint是否存在？
modal volume ls rsd-models | grep best_draft_with_projection.pt

# Checkpoint是否损坏？
python -c "import torch; torch.load('best_draft_with_projection.pt')"
```

### 问题2: "Shape mismatch"

**调试**：
```python
# 在generate_action中添加
print(f"Last hidden state: {last_hidden_state.shape}")  # 应该是[1, 4096]
print(f"Draft logits: {draft_logits.shape}")  # 应该是[1, 3, 128, 7]
```

### 问题3: "Acceptance rate = 0%"

**原因**：Token comparison逻辑还是placeholder。

**解决**：实现真正的layer-wise comparison（TODO in rsd_engine_core.py:253）

### 问题4: "成功率下降到<70%"

**检查**：
1. Draft Model的训练准确率是否>85%？
2. Main Model是否用了正确的fine-tuned版本？
3. RFSQ decoder是否正确工作？

---

## ✅ 集成成功标志

运行完整评估后，应该看到：

```
============================================================
🎉 EVALUATION COMPLETE!
============================================================
   Task Suite: libero_spatial
   Total Episodes: 50
   Total Successes: 44
   Success Rate: 88.0%
   Avg Inference Time: 48.3 ms
   Speculative Decoding: True
============================================================

📊 RSD Engine Statistics:
   Total inferences: 12,450
   Avg inference time: 48.3ms
   Avg draft time: 12.1ms
   Avg main time: 28.5ms
   Draft acceptance rate: 68.2%
============================================================
```

**对比baseline（无Draft）**：
- Inference time从70ms降到48ms → **1.45x speedup** ✅
- Success rate保持在85-95% → **性能不降** ✅
- Draft acceptance 68% → **有效加速** ✅

---

## 🎯 下一步

集成成功后：

1. **运行完整评估**：
   ```bash
   modal run modal_phase3_libero_eval.py --num-trials 50
   ```

2. **记录结果**：保存到实验报告

3. **优化（可选）**：
   - 调整acceptance threshold
   - 实现真正的token comparison
   - 尝试不同的Draft architecture

---

## 📝 文档更新

集成完成后，更新这些文档：

1. **phase3/CRITICAL_FIX.md**：标记为"已解决"
2. **phase3/README_FIXES.md**：更新状态
3. **FINAL_RESULTS.md**（新建）：记录最终性能

---

**准备好集成了吗？开始吧！🚀**
