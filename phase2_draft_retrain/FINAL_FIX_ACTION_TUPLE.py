"""
最终修复：处理 predict_action 返回 tuple 的问题

错误: TypeError: expected np.ndarray (got tuple)
原因: predict_action 返回的是 (action, metadata) 而不是直接的 action

修复: 提取 tuple 的第一个元素
"""

# ============================================================
# 最终修复的 OpenVLA 调用代码
# ============================================================

for step in range(300):
    try:
        image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

        with torch.no_grad():
            # Step 1: Process inputs
            inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

            # Step 2: Get hidden states (with fallback)
            hidden_4096 = None
            try:
                outputs = openvla(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()
            except Exception as e:
                print(f"         ⚠️ Hidden states error: {e}")

            if hidden_4096 is None or hidden_4096.shape != (1, 4096):
                hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)

            # Step 3: Get action (with fallback)
            action_result = None
            try:
                action_result = openvla.predict_action(**inputs, do_sample=False)
            except Exception as e:
                print(f"         ⚠️ predict_action failed: {e}")
                try:
                    action_result = openvla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                except:
                    action_result = None

            if action_result is None:
                print(f"         ⚠️ Could not get action, skipping step {step}")
                continue

            # ✅ FIX: Handle tuple return value
            if isinstance(action_result, tuple):
                # Extract first element (the actual action)
                action = action_result[0]
                print(f"         ℹ️ predict_action returned tuple, extracted action")
            else:
                action = action_result

        # Validate action type and shape
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        if not isinstance(action, np.ndarray):
            try:
                action = np.array(action, dtype=np.float32)
            except Exception as e:
                print(f"         ⚠️ Cannot convert action to array: {e}, type: {type(action)}")
                continue

        # Flatten and adjust shape
        if action.ndim > 1:
            action = action.flatten()

        if action.shape[0] != 7:
            if action.shape[0] > 7:
                action = action[:7]
            elif action.shape[0] < 7:
                action = np.pad(action, (0, 7 - action.shape[0]), 'constant')
            else:
                print(f"         ⚠️ Action has 0 elements, skipping")
                continue

        # Ensure float32 dtype
        action = action.astype(np.float32)

        # Encode to RFSQ
        with torch.no_grad():
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
            action_chunk = action_tensor.unsqueeze(1).expand(1, 8, 7)
            _, rfsq_codes = rfsq_encoder(action_chunk)

        episode_samples.append({
            'hidden_state': hidden_4096.squeeze(0).cpu(),
            'rfsq_tokens': rfsq_codes[0].cpu(),
        })

        obs, reward, done, info = env.step(action)
        if done:
            break

    except Exception as step_error:
        print(f"        ⚠️ Step {step} error: {step_error}")
        import traceback
        traceback.print_exc()
        continue


# ============================================================
# 完整的替换函数（整个 action 处理部分）
# ============================================================

def process_openvla_action(action_result):
    """
    处理 predict_action 的返回值，支持多种格式

    Returns:
        np.ndarray: shape (7,), dtype float32
        None: 如果处理失败
    """
    # Handle tuple
    if isinstance(action_result, tuple):
        if len(action_result) > 0:
            action = action_result[0]
        else:
            return None
    else:
        action = action_result

    # Handle torch.Tensor
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    # Handle list
    if isinstance(action, list):
        action = np.array(action, dtype=np.float32)

    # Verify it's numpy array
    if not isinstance(action, np.ndarray):
        try:
            action = np.array(action, dtype=np.float32)
        except:
            return None

    # Flatten if needed
    if action.ndim > 1:
        action = action.flatten()

    # Adjust shape to (7,)
    if action.shape[0] != 7:
        if action.shape[0] > 7:
            action = action[:7]
        elif action.shape[0] < 7:
            action = np.pad(action, (0, 7 - action.shape[0]), 'constant')
        elif action.shape[0] == 0:
            return None

    # Ensure float32
    action = action.astype(np.float32)

    return action


# ============================================================
# 使用示例
# ============================================================

# In the data collection loop:
try:
    action_result = openvla.predict_action(**inputs, do_sample=False)
    action = process_openvla_action(action_result)

    if action is None:
        print(f"         ⚠️ Invalid action format, skipping")
        continue

    # action 现在保证是 shape (7,) 的 np.float32 数组
    action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
    # ... rest of the code ...

except Exception as e:
    print(f"         ⚠️ Action processing error: {e}")
    continue
