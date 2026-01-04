# 01. 开发环境配置

## 🎯 本节目标

- 安装 Python 3.12+
- 了解 pyenv 版本管理
- 配置 VS Code
- 掌握基本运行方式

---

## 📦 Python 安装

### macOS

```bash
# 方式 1：Homebrew（推荐）
brew install python@3.12

# 方式 2：官网下载
# https://www.python.org/downloads/

# 验证安装
python3 --version
# Python 3.12.x
```

### Windows

```powershell
# 方式 1：Microsoft Store
# 搜索 Python 3.12

# 方式 2：官网下载
# https://www.python.org/downloads/
# ⚠️ 安装时勾选 "Add Python to PATH"

# 验证安装
python --version
```

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv

# 验证
python3.12 --version
```

---

## 🔄 pyenv 版本管理

> 类似 Node.js 的 nvm，管理多个 Python 版本

### 安装 pyenv

```bash
# macOS
brew install pyenv

# 添加到 shell 配置（~/.zshrc 或 ~/.bashrc）
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# 重启 shell
source ~/.zshrc
```

### 常用命令

```bash
# 查看可安装的版本
pyenv install --list | grep 3.12

# 安装指定版本
pyenv install 3.12.0

# 设置全局版本
pyenv global 3.12.0

# 设置项目版本（当前目录）
pyenv local 3.12.0

# 查看已安装版本
pyenv versions
```

### JS 对照：nvm vs pyenv

| nvm (Node.js) | pyenv (Python) |
|---------------|----------------|
| `nvm install 18` | `pyenv install 3.12` |
| `nvm use 18` | `pyenv local 3.12` |
| `nvm alias default 18` | `pyenv global 3.12` |
| `.nvmrc` | `.python-version` |

---

## 💻 VS Code 配置

### 1. 安装 Python 插件

1. 打开 VS Code
2. 按 `Cmd+Shift+X`（Extensions）
3. 搜索 "Python"
4. 安装 Microsoft 官方的 Python 插件

### 2. 推荐设置

在 `settings.json` 中添加：

```json
{
  // Python 解释器路径
  "python.defaultInterpreterPath": "python3",

  // 格式化
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python",
    "editor.tabSize": 4
  },

  // Linting
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,

  // 类型检查
  "python.analysis.typeCheckingMode": "basic"
}
```

### 3. 选择 Python 解释器

1. 按 `Cmd+Shift+P`
2. 输入 "Python: Select Interpreter"
3. 选择你安装的 Python 版本

---

## 🏃 运行 Python 代码

### 方式 1：REPL（交互式）

```bash
$ python3
Python 3.12.0 (main, Oct  2 2023, 00:00:00)
>>> print("Hello!")
Hello!
>>> 1 + 1
2
>>> exit()
```

> 类似 Node.js 的 `node` 命令进入交互模式

### 方式 2：命令行直接执行

```bash
# 执行单行代码
python3 -c "print('Hello from CLI')"

# JS 对照
# node -e "console.log('Hello from CLI')"
```

### 方式 3：运行脚本文件

```bash
# 创建文件 hello.py
echo 'print("Hello, Python!")' > hello.py

# 运行
python3 hello.py

# JS 对照
# node hello.js
```

### 方式 4：可执行脚本（Unix）

```python
#!/usr/bin/env python3
# hello.py

print("Hello, executable!")
```

```bash
chmod +x hello.py
./hello.py
```

---

## 📁 项目结构最佳实践

```
my-project/
├── .python-version      # pyenv 版本文件
├── requirements.txt     # 依赖列表（类似 package.json）
├── src/                 # 源代码
│   └── main.py
├── tests/               # 测试
│   └── test_main.py
└── README.md
```

### JS 对照

| Python | JavaScript |
|--------|------------|
| `requirements.txt` | `package.json` |
| `pip install -r requirements.txt` | `npm install` |
| `.python-version` | `.nvmrc` |
| `venv/` | `node_modules/` |

---

## ✅ 环境检查清单

```bash
# 1. Python 版本
python3 --version
# 应该 >= 3.12

# 2. pip 包管理器
pip3 --version
# 或 python3 -m pip --version

# 3. 虚拟环境支持
python3 -m venv --help
# 应该显示帮助信息

# 4. VS Code Python 插件
# 打开 .py 文件，右下角应显示 Python 版本
```

---

## 🔗 相关资源

- [Python 官网](https://www.python.org/)
- [pyenv GitHub](https://github.com/pyenv/pyenv)
- [VS Code Python 文档](https://code.visualstudio.com/docs/python/python-tutorial)

