# Draft Model 重新训练方案

## 🎯 目标

训练一个**包含Projection Layer的Draft Model**，能够：
1. 接受OpenVLA的4096维hidden states作为输入
2. 预测RFSQ的前3个coarse layers（L0, L1, L2）
3. 达到>85%的coarse layer accuracy
4. 在Phase 3中实现1.3-1.6x推理加速

---

## 🏗️ 新架构设计

### 模型结构

```python
class RFSQDraftModelWithProjection(nn.Module):
    def __init__(
        self,
        input_dim=4096,         # OpenVLA hidden size
        hidden_dim=512,         # Draft model hidden size
        num_coarse_layers=3,    # 预测L0, L1, L2
        chunk_len=8,
        action_hidden_dim=16,   # RFSQ hidden dim
        grid_size=7,            # RFSQ vocab size
    ):
        super().__init__()

        # 🔑 新增：Projection Layer (4096 → 512)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer Decoder（保持不变）
        self.decoder = DraftTransformerDecoder(
            hidden_dim=hidden_dim,
            num_heads=8,
            feedforward_dim=2048,
        )

        # Classification Heads（保持不变）
        # 预测每个coarse layer的tokens
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, chunk_len * action_hidden_dim * grid_size),
            )
            for _ in range(num_coarse_layers)
        ])

    def forward(self, openvla_hidden_states):
        """
        Args:
            openvla_hidden_states: [Batch, 4096] from OpenVLA

        Returns:
            logits: [Batch, Num_Coarse_Layers=3, Chunk*Hidden=128, Grid=7]
        """
        # Step 1: Project 4096 → 512
        projected = self.input_projection(openvla_hidden_states)  # [B, 512]

        # Step 2: Add sequence dimension
        x = projected.unsqueeze(1)  # [B, 1, 512]

        # Step 3: Transformer Decoder
        decoder_output = self.decoder(x)  # [B, 1, 512]
        decoder_output = decoder_output.squeeze(1)  # [B, 512]

        # Step 4: Predict coarse layers
        layer_outputs = []
        for head in self.classification_heads:
            logits = head(decoder_output)  # [B, 128*7=896]
            # Reshape to [B, Chunk*Hidden=128, Grid=7]
            logits = logits.view(-1, 128, 7)
            layer_outputs.append(logits)

        # Stack: [B, 3, 128, 7]
        return torch.stack(layer_outputs, dim=1)
```

### 数据流

```
LIBERO Episode
  ↓
Observations (image + task description)
  ↓
[OpenVLA Frozen Forward]
  ↓
Hidden States [Batch, Seq, 4096]
  ↓ (取最后一个token)
Last Hidden State [Batch, 4096]
  ↓
┌─────────────────────────────────────┐
│ Draft Model with Projection         │
├─────────────────────────────────────┤
│ 1. Projection: [B, 4096] → [B, 512] │
│ 2. Decoder: [B, 1, 512] → [B, 512]  │
│ 3. Heads: [B, 512] → [B, 3, 128, 7] │
└─────────────────────────────────────┘
  ↓
Predicted Logits [B, 3, 128, 7]
  ↓
Cross-Entropy Loss with Ground Truth
  (Ground Truth from RFSQ encoding)
```

---

## 📊 训练配置

### 数据集

**来源**：LIBERO数据集（libero_spatial）

**处理流程**：
1. 收集episodes（使用OpenVLA rollout或已有数据）
2. 对每个observation：
   - 通过**frozen OpenVLA**提取4096维hidden states
   - 对应的action通过RFSQ encoder得到ground truth tokens
3. 构建训练pairs：`(hidden_4096, rfsq_tokens_L0_L1_L2)`

**数据量**：
- 训练episodes：200-500
- 验证episodes：50
- 每个episode约300 steps
- 总训练样本：60k-150k

### 训练超参数

```python
config = {
    # Model
    'input_dim': 4096,
    'hidden_dim': 512,
    'num_coarse_layers': 3,
    'chunk_len': 8,
    'action_hidden_dim': 16,
    'grid_size': 7,

    # Training
    'num_episodes': 200,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,

    # Scheduler
    'scheduler': 'cosine',
    'warmup_steps': 1000,

    # Hardware
    'device': 'cuda',
    'gpu': 'A100',
    'mixed_precision': True,  # bfloat16

    # Validation
    'val_every': 5,  # epochs
    'save_best': True,
}
```

### Loss函数

```python
def compute_loss(logits, targets):
    """
    Args:
        logits: [Batch, 3, 128, 7] - Draft predictions
        targets: [Batch, 3, 128] - Ground truth RFSQ tokens (L0-L2)

    Returns:
        loss: scalar
        accuracies: [3] - accuracy for each layer
    """
    batch_size, num_layers, seq_len, vocab_size = logits.shape

    # Flatten for cross-entropy
    logits_flat = logits.view(-1, vocab_size)  # [B*3*128, 7]
    targets_flat = targets.view(-1)  # [B*3*128]

    # Cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat)

    # Per-layer accuracy
    preds = torch.argmax(logits, dim=-1)  # [B, 3, 128]
    accuracies = []
    for layer_idx in range(num_layers):
        acc = (preds[:, layer_idx] == targets[:, layer_idx]).float().mean()
        accuracies.append(acc.item())

    return loss, accuracies
```

---

## 🔧 实现步骤

### Phase 1: 数据准备（1-2小时）

**脚本**：`collect_openvla_features.py`

```python
@app.function(...)
def collect_training_data(num_episodes=200):
    """收集训练数据"""

    # 1. 加载OpenVLA (frozen)
    openvla = AutoModelForVision2Seq.from_pretrained(
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    openvla.eval()

    # 2. 加载RFSQ encoder (frozen)
    rfsq_encoder = ActionRFSQAE(...)
    rfsq_encoder.load_state_dict(...)
    rfsq_encoder.eval()

    # 3. 收集数据
    training_data = []
    for episode_idx in range(num_episodes):
        # Run episode with OpenVLA
        env.reset()
        for step in range(300):
            # Get observation
            obs = env.get_obs()

            # Extract OpenVLA hidden states
            with torch.no_grad():
                inputs = processor(...)
                outputs = openvla(**inputs, output_hidden_states=True)
                hidden_4096 = outputs.hidden_states[-1][:, -1, :]

            # Get action and encode to RFSQ
            action = openvla.predict_action(...)
            with torch.no_grad():
                _, rfsq_codes = rfsq_encoder(action_tensor)
                # rfsq_codes: [1, 8, 16, 8] (Batch, Chunk, Hidden, Layers)

            # Extract coarse layers (L0, L1, L2)
            coarse_tokens = rfsq_codes[0, :, :, :3]  # [8, 16, 3]

            training_data.append({
                'hidden_state': hidden_4096.cpu(),
                'coarse_tokens': coarse_tokens.cpu(),
            })

            env.step(action)

    # 4. Save
    torch.save(training_data, '/data/draft_training_data.pt')
    return len(training_data)
```

### Phase 2: 训练Draft Model（3-4小时）

**脚本**：`modal_train_draft_with_projection.py`

```python
@app.function(...)
def train_draft_model():
    """训练Draft Model"""

    # 1. Load data
    data = torch.load('/data/draft_training_data.pt')
    train_loader = create_dataloader(data, batch_size=32)

    # 2. Create model
    model = RFSQDraftModelWithProjection(
        input_dim=4096,
        hidden_dim=512,
        num_coarse_layers=3,
    ).to(device)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    # 4. Training loop
    best_acc = 0.0
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            hidden = batch['hidden_state'].to(device)
            targets = batch['coarse_tokens'].to(device)

            # Forward
            logits = model(hidden)

            # Loss
            loss, accs = compute_loss(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        if epoch % 5 == 0:
            val_acc = validate(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc)

    return best_acc
```

### Phase 3: 验证和集成（30分钟）

**验证脚本**：

```python
def test_draft_model():
    """测试Draft Model"""

    # 1. Load model
    model = RFSQDraftModelWithProjection(...)
    checkpoint = torch.load('/models/best_draft_with_projection.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Test forward pass
    test_hidden = torch.randn(1, 4096).to(device)
    logits = model(test_hidden)

    print(f"✅ Input shape: {test_hidden.shape}")
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Expected: [1, 3, 128, 7]")

    assert logits.shape == (1, 3, 128, 7), "Shape mismatch!"

    # 3. Check projection weights
    assert hasattr(model, 'input_projection')
    print(f"✅ Projection weights shape: {model.input_projection.weight.shape}")

    return True
```

---

## 📈 预期训练曲线

### Epoch 0-10: 快速学习

```
Epoch  | Train Loss | Val Acc L0 | Val Acc L1 | Val Acc L2 | Avg
-------|------------|------------|------------|------------|------
1      | 1.850      | 45.2%      | 42.8%      | 40.1%      | 42.7%
5      | 0.920      | 68.5%      | 65.2%      | 61.8%      | 65.2%
10     | 0.450      | 82.1%      | 79.5%      | 76.3%      | 79.3%
```

### Epoch 10-30: 持续改进

```
Epoch  | Train Loss | Val Acc L0 | Val Acc L1 | Val Acc L2 | Avg
-------|------------|------------|------------|------------|------
15     | 0.320      | 87.2%      | 84.6%      | 81.5%      | 84.4%
20     | 0.250      | 89.5%      | 87.1%      | 84.2%      | 86.9%
25     | 0.210      | 90.8%      | 88.5%      | 85.9%      | 88.4%
30     | 0.180      | 91.2%      | 89.1%      | 86.5%      | 88.9%
```

### Epoch 30-50: 收敛

```
Epoch  | Train Loss | Val Acc L0 | Val Acc L1 | Val Acc L2 | Avg
-------|------------|------------|------------|------------|------
35     | 0.165      | 91.5%      | 89.4%      | 86.8%      | 89.2%
40     | 0.155      | 91.7%      | 89.6%      | 87.1%      | 89.5%
45     | 0.148      | 91.8%      | 89.7%      | 87.2%      | 89.6%
50     | 0.145      | 91.9%      | 89.8%      | 87.3%      | 89.7%
```

**目标**：平均accuracy > 85%（达成✅）

---

## ✅ 成功标准

训练完成后，checkpoint应该满足：

1. **模型结构**：
   ```python
   checkpoint.keys() == ['model_state_dict', 'optimizer_state_dict',
                         'epoch', 'val_accuracy', 'config']
   ```

2. **包含projection weights**：
   ```python
   'input_projection.weight' in checkpoint['model_state_dict']
   'input_projection.bias' in checkpoint['model_state_dict']
   ```

3. **准确率**：
   ```python
   checkpoint['val_accuracy'] > 0.85  # 85%
   ```

4. **可以加载和使用**：
   ```python
   model = RFSQDraftModelWithProjection(...)
   model.load_state_dict(checkpoint['model_state_dict'])

   # Test
   hidden = torch.randn(1, 4096)
   logits = model(hidden)
   assert logits.shape == (1, 3, 128, 7)
   ```

---

## 🔗 与Phase 3集成

训练完成后，更新`rsd_engine_core.py`：

```python
class RSDInferenceEngine:
    def __init__(self, ...):
        # ...

        # ✅ 不再需要随机初始化projection
        # self.draft_projection = nn.Linear(4096, 512).to(device)

        # ✅ Draft Model自带训练好的projection
        if self.draft_model is not None:
            assert hasattr(self.draft_model, 'input_projection'), \
                "Draft Model must have trained projection layer!"
            print("✅ Using trained projection from Draft Model")
```

---

## 🚨 常见问题

### Q1: 训练数据从哪来？

**A**: 两个选项：
1. **选项A**：使用OpenVLA rollout收集新数据（推荐）
2. **选项B**：如果有Phase 2的LIBERO数据，重新提取OpenVLA features

### Q2: 如果accuracy <85%怎么办？

**A**: 检查：
- 数据质量（OpenVLA是否frozen？RFSQ编码是否正确？）
- 增加训练数据（200 → 500 episodes）
- 调整学习率（1e-4 → 5e-5）
- 增加训练epochs（50 → 100）

### Q3: 训练需要多久？

**A**:
- 数据收集：1-2小时（200 episodes）
- 模型训练：3-4小时（A100, 50 epochs）
- 总计：4-6小时

### Q4: 能否复用原来的Draft weights？

**A**:
可以尝试，但需要：
1. 只训练projection layer（冻结Draft其他部分）
2. 然后fine-tune整个模型

但从头训练更干净，推荐直接重新训练。

---

## 📊 资源需求

| 资源 | 需求 | 说明 |
|------|------|------|
| GPU | A100 (40GB) | 数据收集+训练都需要 |
| 时间 | 4-6小时 | 端到端 |
| 存储 | ~5GB | 训练数据 + checkpoints |
| Modal credits | 估算$10-15 | 取决于具体GPU时长 |

---

## 🎯 下一步

1. **现在**：阅读`modal_train_draft_with_projection.py`了解实现细节
2. **准备**：确保Modal环境和资源ready
3. **运行**：启动数据收集 → 训练
4. **验证**：测试新模型
5. **集成**：更新Phase 3并测试加速效果

**准备好了就开始吧！🚀**
