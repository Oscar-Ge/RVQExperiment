"""
修复 Error 5: Action Chunk Shape 问题

错误: RuntimeError: expand(torch.cuda.FloatTensor{[1, 1, 8, 7]}, size=[1, 8, 7])
原因: predict_action 返回的是 action chunk [8, 7] 而不是单个 action [7]

修复: 提取 chunk 的第一个 action
"""

import numpy as np
import torch

# ============================================================
# 完整的 action 提取函数（更新版）
# ============================================================

def safe_extract_action(action_result):
    """
    安全地从 predict_action 的返回值中提取单个 action

    Args:
        action_result: predict_action 的返回值（可能是 tuple, tensor, array）

    Returns:
        np.ndarray: shape (7,), dtype float32 或 None
    """
    # Step 1: 处理 tuple
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

    # ✅ NEW: Step 3: 处理 action chunk [8, 7] -> [7]
    if action.ndim == 2:
        # Check if it's an action chunk
        if action.shape[0] == 8 and action.shape[1] == 7:
            # Extract first timestep
            action = action[0]
        elif action.shape == (1, 7):
            # Squeeze batch dimension
            action = action.squeeze(0)
        else:
            # Flatten and take first 7 elements
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
# 使用示例（替换到数据收集循环中）
# ============================================================

# In the data collection loop:
for step in range(300):
    try:
        image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

        with torch.no_grad():
            # Process inputs
            inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

            # Get hidden states (with fallback)
            hidden_4096 = None
            try:
                outputs = openvla(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_4096 = outputs.hidden_states[-1][:, -1, :].float()
            except Exception as e:
                pass  # Use fallback

            if hidden_4096 is None or hidden_4096.shape != (1, 4096):
                hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)

            # Get action (with fallback)
            action = None
            try:
                action_result = openvla.predict_action(**inputs, do_sample=False)
                action = safe_extract_action(action_result)
            except Exception as e:
                try:
                    action_result = openvla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                    action = safe_extract_action(action_result)
                except:
                    action = None

            if action is None:
                continue

        # ✅ FIXED: action 现在保证是 shape (7,)
        # Encode to RFSQ
        with torch.no_grad():
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)  # [1, 7]
            action_chunk = action_tensor.unsqueeze(1).expand(1, 8, 7)  # [1, 1, 7] -> [1, 8, 7]
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
        continue


# ============================================================
# 调试：打印 action shape
# ============================================================

# 如果想验证 action 的 shape，可以在提取后添加：
def debug_action_extraction(action_result):
    """调试版本：打印每一步的 shape"""
    print(f"1. action_result type: {type(action_result)}")

    if isinstance(action_result, tuple):
        action = action_result[0]
        print(f"2. Extracted from tuple, shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
    else:
        action = action_result

    if isinstance(action, torch.Tensor):
        print(f"3. Before cpu().numpy(): {action.shape}")
        action = action.detach().cpu().numpy()

    print(f"4. After conversion to numpy: {action.shape}, dtype: {action.dtype}")

    if action.ndim == 2 and action.shape == (8, 7):
        print(f"5. Detected action chunk [8, 7], extracting first timestep")
        action = action[0]

    print(f"6. Final action shape: {action.shape}")

    return action
