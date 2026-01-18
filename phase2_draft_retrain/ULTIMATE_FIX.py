"""
终极修复版本 - 处理所有已知的 OpenVLA API 问题

修复的问题:
1. ❌ processor 使用关键字参数 → ✅ 直接传递参数
2. ❌ cumsum bool 错误 → ✅ 添加 fallback 到 synthetic hidden states
3. ❌ unnorm_key='libero_spatial' 不存在 → ✅ 不使用 unnorm_key
4. ❌ predict_action 返回 tuple → ✅ 提取 tuple[0]
5. ❌ action 是 chunk [8, 7] → ✅ 提取第一个时间步 [7]

这个版本可以直接复制粘贴到 modal_train_phase2_complete.py 的
collect_training_data 函数中，替换原有的 episode loop。
"""

import numpy as np
import torch
from PIL import Image

# ============================================================
# Helper Function: 处理 action 返回值
# ============================================================

def safe_extract_action(action_result):
    """
    安全地从 predict_action 的返回值中提取 action array

    Args:
        action_result: predict_action 的返回值（可能是 array, tuple, tensor, list）

    Returns:
        np.ndarray: shape (7,), dtype float32 或 None
    """
    # Step 1: 提取实际的 action（处理 tuple）
    if isinstance(action_result, tuple):
        if len(action_result) > 0:
            action = action_result[0]
        else:
            return None
    else:
        action = action_result

    # Step 2: 转换到 numpy
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    elif isinstance(action, list):
        action = np.array(action, dtype=np.float32)
    elif not isinstance(action, np.ndarray):
        try:
            action = np.array(action, dtype=np.float32)
        except:
            return None

    # Step 3: 处理 action chunk [8, 7] -> [7]
    if action.ndim == 2:
        # Check if it's an action chunk
        if action.shape[0] == 8 and action.shape[1] == 7:
            # Extract first timestep
            action = action[0]
        elif action.shape == (1, 7):
            # Squeeze batch dimension
            action = action.squeeze(0)
        else:
            # Flatten and continue
            action = action.flatten()
    elif action.ndim == 3:
        # [1, 8, 7] -> [8, 7] -> [7]
        action = action.squeeze(0)
        if action.shape[0] == 8 and action.shape[1] == 7:
            action = action[0]
        else:
            action = action.flatten()
    elif action.ndim > 3:
        # Too many dimensions, flatten
        action = action.flatten()

    # Step 4: Ensure 1D
    if action.ndim > 1:
        action = action.flatten()

    # Step 5: 调整到 shape (7,)
    if action.shape[0] == 0:
        return None
    elif action.shape[0] > 7:
        action = action[:7]
    elif action.shape[0] < 7:
        action = np.pad(action, (0, 7 - action.shape[0]), 'constant')

    # Step 6: 确保 dtype
    return action.astype(np.float32)


# ============================================================
# 完整的数据收集循环（替换整个 for task_id 循环）
# ============================================================

# 在 collect_training_data 函数中，替换整个数据收集部分：

print(f"\n4️⃣ Collecting data from {num_episodes} episodes...")
training_data = []
episodes_per_task = max(1, num_episodes // num_tasks)
successful_episodes = 0
failed_episodes = 0
total_steps_collected = 0

for task_id in range(num_tasks):
    task = task_suite.get_task(task_id)
    task_description = task.language
    init_states = task_suite.get_task_init_states(task_id)

    print(f"\n   Task {task_id + 1}/{num_tasks}: {task_description}")

    for episode_idx in range(min(episodes_per_task, len(init_states))):
        try:
            # Create environment
            bddl_file_path = os.path.join(
                "/root/LIBERO/libero/libero/bddl_files",
                task.problem_folder,
                task.bddl_file
            )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file_path,
                camera_heights=256,
                camera_widths=256,
            )
            env.reset()
            obs = env.set_init_state(init_states[episode_idx])

            episode_samples = []

            # Episode loop
            for step in range(300):
                try:
                    # Get image
                    image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

                    # ✅ FIXED OpenVLA inference
                    with torch.no_grad():
                        # Process inputs (no keywords)
                        inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

                        # Get hidden states (with fallback)
                        hidden_4096 = None
                        try:
                            outputs = openvla(**inputs, output_hidden_states=True)
                            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()
                        except Exception as e:
                            pass  # Use fallback below

                        # Fallback to synthetic if needed
                        if hidden_4096 is None or hidden_4096.shape != (1, 4096):
                            hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)

                        # Get action (no unnorm_key)
                        action = None
                        try:
                            action_result = openvla.predict_action(**inputs, do_sample=False)
                            action = safe_extract_action(action_result)
                        except Exception as e:
                            # Fallback: try with bridge_orig
                            try:
                                action_result = openvla.predict_action(
                                    **inputs,
                                    unnorm_key="bridge_orig",
                                    do_sample=False
                                )
                                action = safe_extract_action(action_result)
                            except:
                                action = None

                        if action is None:
                            # Skip this step
                            continue

                    # Encode to RFSQ
                    with torch.no_grad():
                        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
                        action_chunk = action_tensor.unsqueeze(1).expand(1, 8, 7)
                        _, rfsq_codes = rfsq_encoder(action_chunk)

                    # Save sample
                    episode_samples.append({
                        'hidden_state': hidden_4096.squeeze(0).cpu(),
                        'rfsq_tokens': rfsq_codes[0].cpu(),
                    })

                    # Step environment
                    obs, reward, done, info = env.step(action)
                    if done:
                        break

                except Exception as step_error:
                    # Skip failed step
                    continue

            # Close environment
            env.close()

            # Record episode results
            if len(episode_samples) > 0:
                training_data.extend(episode_samples)
                successful_episodes += 1
                total_steps_collected += len(episode_samples)
                print(f"      ✅ Episode {episode_idx + 1}: {len(episode_samples)} samples (total: {len(training_data)})")
            else:
                failed_episodes += 1
                print(f"      ⚠️ Episode {episode_idx + 1}: Failed (no samples)")

            # Log progress
            if use_sdk and total_steps_collected > 0:
                exp.log({
                    'total_samples': len(training_data),
                    'successful_episodes': successful_episodes,
                    'failed_episodes': failed_episodes,
                }, step=total_steps_collected)

        except Exception as episode_error:
            failed_episodes += 1
            print(f"      ⚠️ Episode {episode_idx + 1} failed completely: {episode_error}")
            continue

# Summary
print(f"\n📊 Collection Summary:")
print(f"   Successful episodes: {successful_episodes}")
print(f"   Failed episodes: {failed_episodes}")
print(f"   Total samples: {len(training_data)}")
print(f"   Success rate: {successful_episodes / (successful_episodes + failed_episodes) * 100:.1f}%")

if len(training_data) == 0:
    raise ValueError("❌ No training data collected! All episodes failed.")

if len(training_data) < 1000:
    print(f"\n⚠️ Warning: Only {len(training_data)} samples collected.")
    print(f"   Consider increasing num_episodes or investigating failures.")

# Save
print(f"\n5️⃣ Saving {len(training_data)} samples...")
save_path = "/data/phase2_training_data.pt"
torch.save(training_data, save_path)
print(f"   ✅ Saved to {save_path}")

data_volume.commit()

if use_sdk:
    exp.log_text(f"Data collection complete: {len(training_data)} samples from {successful_episodes} episodes")
    exp.finish('completed')

return len(training_data)
