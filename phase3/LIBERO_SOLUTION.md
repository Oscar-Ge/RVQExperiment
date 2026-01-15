# LIBERO Integration Solution

## 🚨 Problem

每次clone LIBERO仓库时：
1. 目录是空的（可能是submodule问题）
2. 使用`git clone --recursive`后，你做的修改（torch.load fix）会消失
3. 无法将修改merge到LIBERO主仓库（因为你不是维护者）

---

## ✅ Solution: 不要把LIBERO包含在项目中

### 核心思想

**不要**把LIBERO作为项目的一部分。而是在Modal镜像构建时动态clone和patch。

---

## 🔧 Implementation (Already Done in modal_phase3_libero_eval.py)

### 在Modal Image中处理LIBERO

```python
# 在Modal image构建时clone和修复LIBERO
eval_image = eval_image.run_commands(
    # 1. Clone LIBERO到Modal容器中
    "cd /root && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git",

    # 2. 应用torch.load fix
    "cd /root/LIBERO && sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' "
    "libero/libero/benchmark/__init__.py",

    # 3. 安装LIBERO
    "cd /root/LIBERO && uv pip install --system -e .",

    # 4. 安装依赖
    "uv pip install --system mujoco dm-control robosuite",
)
```

### 为什么这样可行

1. ✅ **每次构建镜像时**，LIBERO都会被clone和patch
2. ✅ **修改在镜像中持久化**，不需要在项目中保存
3. ✅ **不影响你的git仓库**，保持clean
4. ✅ **可复现**，任何人运行你的代码都会得到同样的环境

---

## 📋 What NOT to Do

### ❌ 方案1：把LIBERO作为git submodule

**问题**:
```bash
git submodule add https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

- 每次`git submodule update`会覆盖你的修改
- 无法commit你的修改到主LIBERO仓库
- 团队成员需要记得`git submodule init`

### ❌ 方案2：直接clone到项目中

**问题**:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# 然后修改文件
```

- LIBERO/.git会被添加到你的仓库（混乱）
- Git会忽略LIBERO目录（如果在.gitignore中）
- 团队成员需要手动clone和patch

### ❌ 方案3：Fork LIBERO并修改

**问题**:
```bash
# Fork到你的账号
git clone https://github.com/YOUR_USERNAME/LIBERO.git
```

- 需要维护你的fork（merge upstream changes）
- 其他人必须使用你的fork
- 对于一个简单的一行修改来说太重了

---

## ✅ Recommended: Modal Image Build Approach

### Step 1: 项目结构（不包含LIBERO）

```
RVQExperiment/
├── phase3/
│   ├── modal_phase3_libero_eval.py    # 包含LIBERO安装逻辑
│   ├── rsd_inference_engine.py
│   └── AGENT_INSTRUCTIONS.md
└── .gitignore                         # 包含 LIBERO/ (如果有的话)
```

### Step 2: .gitignore 确保LIBERO不被追踪

```bash
# 如果你确实在本地有LIBERO目录用于测试
echo "LIBERO/" >> .gitignore
```

### Step 3: Modal Image自动处理一切

在`modal_phase3_libero_eval.py`中（已实现）：

```python
# Build evaluation image with LIBERO + OpenVLA dependencies
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    # ... 其他依赖 ...
)

# Clone and install LIBERO (with torch.load fix already applied)
eval_image = eval_image.run_commands(
    "cd /root && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git",
    # Apply torch.load fix
    "cd /root/LIBERO && sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' "
    "libero/libero/benchmark/__init__.py",
    # Install LIBERO
    "cd /root/LIBERO && uv pip install --system -e .",
    # Install additional robot deps
    "uv pip install --system mujoco dm-control robosuite",
)
```

**这段代码已经在你的评估脚本中了！**

---

## 🎯 Alternative: Runtime Monkey Patching

如果你不想在image build时修改文件，可以在运行时patch：

### 方案：在Python中动态修复

```python
# 在modal_phase3_libero_eval.py的函数开始处添加
def run_libero_evaluation(...):
    import sys
    sys.path.insert(0, "/root/LIBERO")

    # Monkey patch torch.load in LIBERO
    import torch
    from libero.libero import benchmark

    # 保存原始torch.load
    original_torch_load = torch.load

    # 定义包装函数
    def patched_torch_load(*args, **kwargs):
        # 强制添加weights_only=False
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    # 替换torch.load
    torch.load = patched_torch_load

    # 现在LIBERO会使用patched版本
    benchmark_dict = benchmark.get_benchmark_dict()
    # ...
```

**优点**:
- 不需要修改LIBERO源码
- 更flexible
- 可以在需要时开关

**缺点**:
- Monkey patching可能有副作用
- 不如直接修改文件清晰

---

## 📝 Verification

### 确认LIBERO正确安装和patch

在Modal函数中添加测试：

```python
@app.function(image=eval_image, ...)
def test_libero_setup():
    import sys
    sys.path.insert(0, "/root/LIBERO")

    # 测试1: LIBERO可以导入
    try:
        from libero.libero import benchmark
        print("✓ LIBERO imported successfully")
    except Exception as e:
        print(f"✗ LIBERO import failed: {e}")
        return False

    # 测试2: torch.load patch生效
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_spatial"]()

        # 尝试加载init states（会触发torch.load）
        task = task_suite.get_task(0)
        init_states = task_suite.get_task_init_states(0)

        print(f"✓ torch.load works, got {len(init_states)} initial states")
        return True
    except Exception as e:
        print(f"✗ torch.load failed: {e}")
        return False
```

运行测试：
```bash
modal run phase3/modal_phase3_libero_eval.py::test_libero_setup
```

---

## 🚀 Best Practices

### DO ✅

1. **在Modal image中处理外部依赖**
   - Clone在image build时
   - Patch在image build时
   - Install在image build时

2. **使用.gitignore**
   ```bash
   # .gitignore
   LIBERO/
   __pycache__/
   *.pyc
   .env
   ```

3. **文档化依赖**
   在README中说明：
   ```markdown
   ## Dependencies

   LIBERO is automatically cloned and patched during Modal image build.
   You don't need to clone it manually.
   ```

### DON'T ❌

1. **不要把LIBERO添加到git**
   ```bash
   git add LIBERO/  # ❌ 不要这样做
   ```

2. **不要使用git submodule**（除非你需要特定版本控制）

3. **不要在本地修改然后期望同步到Modal**
   - 本地的LIBERO和Modal中的是独立的

---

## 📊 Summary

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **Modal Image Build** (当前) | ✅ 自动化<br>✅ 可复现<br>✅ 不污染项目 | - | ⭐⭐⭐⭐⭐ |
| Runtime Monkey Patch | ✅ Flexible | ⚠️ 副作用 | ⭐⭐⭐ |
| Git Submodule | - | ❌ 修改会丢失 | ⭐ |
| Fork LIBERO | ✅ 版本控制 | ❌ 维护负担 | ⭐⭐ |
| 直接添加到git | - | ❌ 混乱 | ❌ |

---

## ✅ Current Status

**你的代码已经使用了最佳方案！**

在`modal_phase3_libero_eval.py`中：
- ✅ LIBERO在Modal image build时自动clone
- ✅ torch.load bug自动修复
- ✅ 不需要手动操作
- ✅ 团队成员运行时自动获得正确环境

**你不需要在本地有LIBERO目录！**

---

## 🎯 Action Items

### For Your Repository

1. **确保.gitignore包含LIBERO**:
   ```bash
   echo "LIBERO/" >> .gitignore
   ```

2. **如果已经commit了LIBERO，删除它**:
   ```bash
   git rm -r LIBERO/
   git commit -m "Remove LIBERO (now handled by Modal image)"
   ```

3. **文档说明**:
   在README中添加：
   ```markdown
   ## LIBERO Setup

   LIBERO is automatically installed during Modal image build.
   No manual setup required.

   The torch.load compatibility fix is automatically applied.
   ```

### For Running Evaluation

**Nothing!** Just run:
```bash
modal run phase3/modal_phase3_libero_eval.py
```

Modal会自动：
1. Build image（如果需要）
2. Clone LIBERO
3. Apply patch
4. Install dependencies
5. Run evaluation

---

## 🎉 Conclusion

**问题解决！**

- ✅ 不需要在项目中包含LIBERO
- ✅ 修改不会丢失（在Modal image中持久化）
- ✅ 可复现（任何人运行都会得到相同环境）
- ✅ Clean git history（不包含外部依赖）

**你的Modal代码已经正确实现了这个方案！**
