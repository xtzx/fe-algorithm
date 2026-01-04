# 02. 虚拟环境 (venv)

## 本节目标

- 理解虚拟环境的原理
- 掌握 venv 的创建和使用
- 对比 Node.js 的 node_modules

---

## 什么是虚拟环境

虚拟环境是**隔离的 Python 环境**，每个项目可以有独立的依赖。

```
类比 Node.js:
Python venv ≈ 项目级 node_modules
全局 Python ≈ 全局 npm
```

---

## 为什么需要虚拟环境

```
项目 A: 需要 requests==2.25.0
项目 B: 需要 requests==2.28.0

没有虚拟环境 → 冲突！
有虚拟环境 → 各自隔离 ✓
```

---

## 创建虚拟环境

### 使用 venv（标准库）

```bash
# 创建
python3 -m venv .venv

# 激活（macOS/Linux）
source .venv/bin/activate

# 激活（Windows）
.venv\Scripts\activate

# 退出
deactivate
```

### 使用 uv（更快）

```bash
# 创建
uv venv

# 激活
source .venv/bin/activate
```

---

## 虚拟环境结构

```
.venv/
├── bin/                    # macOS/Linux
│   ├── activate            # 激活脚本
│   ├── python -> python3.12
│   ├── python3 -> python3.12
│   ├── python3.12
│   └── pip
├── include/                # C 头文件
├── lib/
│   └── python3.12/
│       └── site-packages/  # 安装的包
└── pyvenv.cfg              # 配置文件
```

### pyvenv.cfg

```ini
home = /usr/local/bin
include-system-site-packages = false
version = 3.12.0
```

---

## 工作原理

### 1. PATH 修改

```bash
# 激活前
which python
# /usr/local/bin/python3

# 激活后
source .venv/bin/activate
which python
# /path/to/project/.venv/bin/python
```

### 2. 解释器隔离

```python
import sys

# 激活虚拟环境后
print(sys.executable)
# /path/to/project/.venv/bin/python

print(sys.prefix)
# /path/to/project/.venv
```

### 3. site-packages 隔离

```python
import site
print(site.getsitepackages())
# ['/path/to/project/.venv/lib/python3.12/site-packages']
```

---

## 与 Node.js 对比

| 特性 | Python venv | Node.js node_modules |
|------|-------------|---------------------|
| 位置 | 项目根目录 `.venv` | 项目根目录 `node_modules` |
| 激活 | 需要 `source activate` | 自动（运行时查找） |
| 隔离级别 | 解释器级别 | 包级别 |
| 包存储 | 扁平 | 嵌套/扁平 |
| 共享 | 不共享 | 不共享 |

---

## 最佳实践

### 1. 始终使用虚拟环境

```bash
# 创建项目时立即创建虚拟环境
mkdir myproject && cd myproject
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 将 .venv 加入 .gitignore

```gitignore
# .gitignore
.venv/
venv/
env/
```

### 3. 使用 .python-version

```bash
# .python-version
3.12
```

配合 pyenv 使用：
```bash
pyenv install 3.12.0
pyenv local 3.12.0
```

### 4. VS Code 配置

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

---

## 常见问题

### 激活后 pip 仍然是全局的

```bash
# 检查 pip 路径
which pip
# 应该是 .venv/bin/pip

# 如果不是，使用完整路径
.venv/bin/pip install package
# 或
python -m pip install package
```

### 虚拟环境损坏

```bash
# 删除重建
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 系统包访问

```bash
# 创建时允许访问系统包（不推荐）
python3 -m venv --system-site-packages .venv
```

---

## pyenv + venv

管理多个 Python 版本：

```bash
# 安装 pyenv
brew install pyenv

# 安装多个 Python 版本
pyenv install 3.10.12
pyenv install 3.11.6
pyenv install 3.12.0

# 设置项目 Python 版本
pyenv local 3.12.0

# 创建虚拟环境
python -m venv .venv
```

---

## 其他虚拟环境工具

### virtualenv

```bash
pip install virtualenv
virtualenv .venv
```

比 venv 功能更多，但现在 venv 已经足够。

### conda

```bash
conda create -n myenv python=3.12
conda activate myenv
```

用于科学计算，可以管理非 Python 依赖。

---

## IDE 集成

### VS Code

1. 创建 .venv
2. Cmd+Shift+P → "Python: Select Interpreter"
3. 选择 .venv/bin/python

### PyCharm

1. Settings → Project → Python Interpreter
2. Add Interpreter → Virtualenv Environment
3. 选择 Existing environment → .venv/bin/python

---

## 自动激活

### direnv

```bash
# 安装 direnv
brew install direnv

# 项目根目录创建 .envrc
echo "source .venv/bin/activate" > .envrc

# 允许
direnv allow

# 进入目录自动激活
cd myproject
# 自动激活
```

### zsh-autoenv

```bash
# .autoenv.zsh
source .venv/bin/activate
```

---

## 本节要点

1. **虚拟环境** 隔离项目依赖
2. **venv** 是标准库模块
3. **激活** 修改 PATH，指向虚拟环境
4. **site-packages** 存储安装的包
5. 始终将 `.venv` 加入 `.gitignore`
6. 使用 pyenv 管理多 Python 版本

