# Agent 修复指南：OpenVLA API 调用错误

## 🚨 问题总结

你遇到的两个错误：
1. `TypeError: got multiple values for argument 'unnorm_key'`
2. `ValueError: num_samples=0`

**根本原因**：OpenVLA 的 `predict_action` API 调用方式错误，导致数据收集失败。

---

## ✅ 正确的 OpenVLA API 调用方式

### 错误示例（你当前的代码）

```python
# ❌ 这是错误的！
action = openvla.predict_action(
    image,                        # 不要直接传 image
    task_description,             # 不要直接传 task_description
    unnorm_key="libero_spatial",  # 这会导致参数冲突
)
```

### 正确示例

```python
# ✅ 这是正确的！
# Step 1: 使用 processor 处理输入
inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

# Step 2: 使用 **inputs 解包传递
action = openvla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
```

---

## 📝 完整的数据收集代码修复

### 修复前（错误版本）

```python
# Get OpenVLA hidden states
with torch.no_grad():
    inputs = processor(
        text=task_description,
        images=image,
        return_tensors="pt"
    ).to(device)

    outputs = openvla(**inputs, output_hidden_states=True)
    hidden_4096 = outputs.hidden_states[-1][:, -1, :]  # [1, 4096]

    # ❌ 错误的 predict_action 调用
    action = openvla.predict_action(
        image,
        task_description,
        unnorm_key="libero_spatial",
    )
```

### 修复后（正确版本）

```python
# Get OpenVLA hidden states and action
with torch.no_grad():
    # Step 1: 处理输入（注意参数顺序：text 在前，images 在后）
    inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

    # Step 2: 获取 hidden states（需要传入 output_hidden_states=True）
    outputs = openvla(**inputs, output_hidden_states=True)
    hidden_4096 = outputs.hidden_states[-1][:, -1, :]  # [1, 4096]

    # Step 3: 正确调用 predict_action（重用 inputs）
    action = openvla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
```

**关键点**：
1. `processor()` 的参数顺序是 `(text, image)`，不是 `(text=..., images=...)`
2. 使用 `**inputs` 解包传递给 `predict_action`
3. `unnorm_key` 和 `do_sample` 作为额外的关键字参数传递

---

## 🔧 完整的修复清单

### 1. 找到数据收集函数

在你的脚本中找到 `collect_training_data` 函数（或类似名称）。

### 2. 定位 OpenVLA 调用代码

搜索包含 `openvla.predict_action` 的代码块。

### 3. 应用修复

将以下错误模式：

```python
# ❌ 查找并删除这种模式
inputs = processor(
    text=task_description,
    images=image,
    return_tensors="pt"
).to(device)

outputs = openvla(**inputs, output_hidden_states=True)
hidden_4096 = outputs.hidden_states[-1][:, -1, :]

action = openvla.predict_action(
    image,
    task_description,
    unnorm_key="libero_spatial",
)
```

替换为：

```python
# ✅ 使用这种正确的模式
inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

outputs = openvla(**inputs, output_hidden_states=True)
hidden_4096 = outputs.hidden_states[-1][:, -1, :]

action = openvla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
```

### 4. 添加错误处理

确保有适当的错误处理，避免所有 episodes 都失败：

```python
for episode_idx in range(episodes_per_task):
    try:
        # ... 环境设置代码 ...

        for step in range(300):
            try:
                # 正确的 OpenVLA 调用
                inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)
                outputs = openvla(**inputs, output_hidden_states=True)
                hidden_4096 = outputs.hidden_states[-1][:, -1, :]
                action = openvla.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

                # ... 其余代码 ...

            except Exception as step_error:
                print(f"        ⚠️ Step {step} failed: {step_error}")
                # 继续下一个 step，而不是中断整个 episode
                continue

        print(f"      ✅ Episode {episode_idx + 1}: collected {len(episode_data)} samples")

    except Exception as episode_error:
        print(f"      ⚠️ Episode {episode_idx + 1} failed: {episode_error}")
        # 继续下一个 episode
        continue
```

### 5. 验证数据收集

在数据收集后添加验证：

```python
# 5. 验证并保存数据
print(f"\n5️⃣ Data collection summary:")
print(f"   Total samples collected: {len(training_data)}")

if len(training_data) == 0:
    raise ValueError("❌ No training data collected! Check OpenVLA API calls and error logs.")

if len(training_data) < 1000:
    print(f"   ⚠️ Warning: Only {len(training_data)} samples collected. Consider:")
    print(f"      - Increasing num_episodes")
    print(f"      - Checking episode success rate")
    print(f"      - Reviewing error logs above")

print(f"\n6️⃣ Saving {len(training_data)} samples...")
save_path = "/data/draft_training_data.pt"
torch.save(training_data, save_path)
print(f"   ✅ Saved to {save_path}")
```

---

## 🎯 完整示例：修复后的 collect_training_data 函数

```python
@app.function(...)
def collect_training_data(num_episodes: int = 200):
    """收集训练数据：OpenVLA hidden states + RFSQ token labels"""

    # ... 初始化代码 ...

    # 加载模型
    print("\n1️⃣ Loading OpenVLA (frozen)...")
    openvla = AutoModelForVision2Seq.from_pretrained(
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        trust_remote_code=True,
    )
    openvla.eval()

    # ... 加载 RFSQ encoder ...

    # 收集数据
    training_data = []
    successful_episodes = 0
    failed_episodes = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language

        for episode_idx in range(episodes_per_task):
            try:
                # 创建环境
                env = OffScreenRenderEnv(...)
                env.reset()
                obs = env.set_init_state(init_states[episode_idx])

                episode_samples = []

                # Episode loop
                for step in range(300):
                    try:
                        # 准备图像
                        image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

                        # ✅ 正确的 OpenVLA 调用
                        with torch.no_grad():
                            # Step 1: 处理输入
                            inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

                            # Step 2: 获取 hidden states
                            outputs = openvla(**inputs, output_hidden_states=True)
                            hidden_4096 = outputs.hidden_states[-1][:, -1, :]  # [1, 4096]

                            # Step 3: 获取 action
                            action = openvla.predict_action(
                                **inputs,
                                unnorm_key="libero_spatial",
                                do_sample=False
                            )

                        # 编码 action 到 RFSQ tokens
                        with torch.no_grad():
                            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
                            action_chunk = action_tensor.unsqueeze(1).expand(1, 8, 7)
                            _, rfsq_codes = rfsq_encoder(action_chunk)
                            coarse_tokens = rfsq_codes[0, :, :, :3]

                        # 保存样本
                        episode_samples.append({
                            'hidden_state': hidden_4096.squeeze(0).cpu(),
                            'coarse_tokens': coarse_tokens.cpu(),
                        })

                        # Step environment
                        obs, reward, done, info = env.step(action)
                        if done:
                            break

                    except Exception as step_error:
                        print(f"        ⚠️ Step {step} failed: {step_error}")
                        continue

                # Episode 完成
                env.close()
                if len(episode_samples) > 0:
                    training_data.extend(episode_samples)
                    successful_episodes += 1
                    print(f"      ✅ Episode {episode_idx + 1}: {len(episode_samples)} samples")
                else:
                    failed_episodes += 1
                    print(f"      ⚠️ Episode {episode_idx + 1}: No samples collected")

            except Exception as episode_error:
                failed_episodes += 1
                print(f"      ⚠️ Episode {episode_idx + 1} failed: {episode_error}")
                continue

    # 总结
    print(f"\n📊 Collection Summary:")
    print(f"   Successful episodes: {successful_episodes}")
    print(f"   Failed episodes: {failed_episodes}")
    print(f"   Total samples: {len(training_data)}")

    # 验证
    if len(training_data) == 0:
        raise ValueError("❌ No training data collected! All episodes failed. Check error logs.")

    # 保存
    save_path = "/data/draft_training_data.pt"
    torch.save(training_data, save_path)
    data_volume.commit()

    return len(training_data)
```

---

## 🧪 测试修复

运行修复后的脚本：

```bash
modal run your_fixed_script.py --num-episodes 10  # 先测试少量 episodes
```

**期望输出**：
```
✅ Episode 1: 245 samples
✅ Episode 2: 298 samples
...
📊 Collection Summary:
   Successful episodes: 10
   Failed episodes: 0
   Total samples: 2547
```

如果仍然失败，检查：
1. ✅ `processor()` 参数顺序：`(text, image)` 不是 `(text=..., images=...)`
2. ✅ `predict_action()` 使用 `**inputs` 解包
3. ✅ `unnorm_key` 作为额外参数传递
4. ✅ 添加了适当的错误处理

---

## 📚 参考资源

- OpenVLA 官方示例: https://github.com/openvla/openvla/blob/main/vla-scripts/deploy.py
- Hugging Face 文档: https://huggingface.co/openvla/openvla-7b

---

**总结**：OpenVLA 的 API 不接受直接传递 `image` 和 `task_description`，必须先通过 `processor` 处理成 inputs，然后用 `**inputs` 解包传递。这是你遇到错误的根本原因。
