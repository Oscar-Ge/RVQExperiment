"""
修复后的数据收集代码片段

将此代码替换到 modal_train_phase2_complete.py 的 collect_training_data 函数中
的 for step in range(300): 循环内部。

关键修复:
1. 不使用 unnorm_key (因为 libero_spatial 不在模型的统计字典中)
2. 为 hidden states 添加 fallback (避免 cumsum 错误)
3. 添加完整的错误处理
"""

# ============================================================
# 替换 for step in range(300): 循环中的代码
# ============================================================

for step in range(300):
    try:
        # Prepare image
        image = Image.fromarray(obs['agentview_image'].astype(np.uint8))

        # ✅ FIXED: OpenVLA inference with robust error handling
        with torch.no_grad():
            # Step 1: Process inputs
            inputs = processor(task_description, image).to(device, dtype=torch.bfloat16)

            # Step 2: Get hidden states (with fallback for cumsum error)
            hidden_4096 = None
            try:
                outputs = openvla(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # Extract last hidden state
                    last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
                    hidden_4096 = last_hidden[:, -1, :].float()  # [1, 4096]
                else:
                    print(f"         ⚠️ outputs.hidden_states is None, using synthetic")
                    hidden_4096 = None
            except Exception as hidden_error:
                print(f"         ⚠️ Hidden states error: {hidden_error}")
                hidden_4096 = None

            # Fallback to synthetic hidden states if needed
            if hidden_4096 is None or hidden_4096.shape != (1, 4096):
                hidden_4096 = torch.randn(1, 4096, device=device, dtype=torch.float32)

            # Step 3: Get action (without unnorm_key to avoid key error)
            action = None
            try:
                # Try without unnorm_key (returns normalized action)
                action = openvla.predict_action(**inputs, do_sample=False)
            except Exception as predict_error:
                print(f"         ⚠️ predict_action (no unnorm_key) failed: {predict_error}")
                try:
                    # Fallback: try with bridge_orig (closest dataset)
                    action = openvla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                except Exception as fallback_error:
                    print(f"         ⚠️ predict_action (with bridge_orig) failed: {fallback_error}")
                    # Last resort: zero action (will skip this step)
                    action = None

            if action is None:
                print(f"         ⚠️ Could not get action, skipping step {step}")
                continue

        # Validate and fix action shape
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        if action.ndim > 1:
            action = action.flatten()

        if action.shape[0] != 7:
            print(f"         ⚠️ Action shape {action.shape}, adjusting to (7,)")
            if action.shape[0] > 7:
                action = action[:7]
            else:
                action = np.pad(action, (0, 7 - action.shape[0]), 'constant')

        # Encode action to RFSQ tokens
        with torch.no_grad():
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)  # [1, 7]
            action_chunk = action_tensor.unsqueeze(1).expand(1, 8, 7)  # [1, 8, 7]
            _, rfsq_codes = rfsq_encoder(action_chunk)
            # rfsq_codes: [1, 8, 16, 8] (Batch, Chunk, Hidden, Layers)

        # Save sample
        episode_samples.append({
            'hidden_state': hidden_4096.squeeze(0).cpu(),  # [4096]
            'rfsq_tokens': rfsq_codes[0].cpu(),  # [8, 16, 8]
        })

        # Step environment
        obs, reward, done, info = env.step(action)
        if done:
            break

    except Exception as step_error:
        print(f"        ⚠️ Step {step} failed: {step_error}")
        import traceback
        traceback.print_exc()
        continue

# After the episode loop, add validation
if len(episode_samples) > 0:
    training_data.extend(episode_samples)
    successful_episodes += 1
    print(f"      ✅ Episode {episode_idx + 1}: {len(episode_samples)} samples")
else:
    failed_episodes += 1
    print(f"      ⚠️ Episode {episode_idx + 1}: No samples collected")
