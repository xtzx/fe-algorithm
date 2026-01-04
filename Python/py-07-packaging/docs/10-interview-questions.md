# 10. 面试题

## 1. wheel 和 sdist 的区别？

**答案**：

| 特性 | wheel | sdist |
|------|-------|-------|
| 格式 | ZIP (.whl) | tar.gz |
| 内容 | 预编译产物 | 源代码 |
| 安装 | 解压即用，快 | 需要构建，慢 |
| C 扩展 | 已编译，平台特定 | 需要编译器 |
| 文件大小 | 较大 | 较小 |

**wheel 优势**：
- 安装速度快（无需编译）
- 不需要构建工具链
- 可预编译 C 扩展

**sdist 优势**：
- 平台无关
- 文件小
- 审计源码

---

## 2. pyproject.toml 的作用？

**答案**：

pyproject.toml 是 Python 项目的统一配置文件（PEP 517/518/621）：

1. **项目元数据**：name、version、description
2. **依赖声明**：dependencies、optional-dependencies
3. **构建系统**：build-system
4. **工具配置**：pytest、ruff、mypy 等

类似于 Node.js 的 package.json。

```toml
[project]
name = "mypackage"
version = "1.0.0"
dependencies = ["requests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 3. 什么是 editable install？

**答案**：

Editable install（可编辑安装）让修改源码立即生效，无需重新安装。

```bash
pip install -e .
```

**原理**：
- 不复制代码到 site-packages
- 创建 .pth 文件指向源码目录
- Python 导入时直接读取源码

**用途**：
- 开发时使用
- 修改后无需重新安装
- 测试本地更改

---

## 4. 如何处理依赖冲突？

**答案**：

**诊断**：
```bash
pip check  # 检查冲突
pipdeptree  # 查看依赖树
```

**解决方案**：

1. **调整版本约束**：放宽或收紧版本要求
2. **升级/降级包**：找兼容版本
3. **联系维护者**：报告兼容性问题
4. **使用约束文件**：`-c constraints.txt`
5. **分离环境**：不同项目使用不同虚拟环境

---

## 5. lockfile 的作用是什么？

**答案**：

Lockfile 锁定完整依赖树的精确版本：

**作用**：
- 可重复安装（今天和明天相同）
- 团队一致（每个人相同版本）
- CI/CD 一致（测试和生产相同）

**原理**：
```
pyproject.toml: requests>=2.28
    ↓ 解析
lockfile: requests==2.28.0
          certifi==2023.7.22
          ...
```

**对比 npm**：
```
requirements.lock ≈ package-lock.json
```

---

## 6. uv 和 pip 的区别？

**答案**：

| 特性 | uv | pip |
|------|----|----|
| 实现语言 | Rust | Python |
| 速度 | 极快 (10-100x) | 慢 |
| 依赖解析 | 快速、确定性 | 较慢 |
| 锁定 | 原生支持 | 需要 pip-tools |
| API 兼容 | 兼容 pip | - |

**uv 优势**：
- 安装速度快 10-100 倍
- 更好的依赖解析算法
- 原生锁定支持
- drop-in 替代 pip

---

## 7. 如何创建一个 Python 包？

**答案**：

1. **创建目录结构**：
```
mypackage/
├── pyproject.toml
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── main.py
└── tests/
```

2. **编写 pyproject.toml**：
```toml
[project]
name = "mypackage"
version = "0.1.0"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

3. **构建**：
```bash
python -m build
```

4. **安装测试**：
```bash
pip install dist/*.whl
```

---

## 8. 如何发布到 PyPI？

**答案**：

1. **注册 PyPI 账号**：https://pypi.org/

2. **配置认证**：
```bash
# ~/.pypirc
[pypi]
username = __token__
password = pypi-xxxx...
```

3. **构建包**：
```bash
python -m build
```

4. **上传**：
```bash
pip install twine
twine upload dist/*
```

5. **验证**：
```bash
pip install mypackage
```

---

## 9. PEP 517 是什么？

**答案**：

PEP 517 定义了构建系统的标准接口：

**核心概念**：
- **build frontend**：pip、build
- **build backend**：setuptools、hatchling、poetry-core

**pyproject.toml 声明**：
```toml
[build-system]
requires = ["hatchling"]  # 构建依赖
build-backend = "hatchling.build"  # 后端入口
```

**好处**：
- 构建工具可互换
- 不依赖 setup.py
- 声明式配置

---

## 10. 如何管理多 Python 版本？

**答案**：

**使用 pyenv**：
```bash
# 安装
brew install pyenv

# 安装 Python 版本
pyenv install 3.10.12
pyenv install 3.11.6
pyenv install 3.12.0

# 设置全局默认
pyenv global 3.12.0

# 设置项目版本
pyenv local 3.11.6

# 查看版本
pyenv versions
```

**配合 venv**：
```bash
pyenv local 3.12.0
python -m venv .venv
```

**tox 多版本测试**：
```ini
[tox]
envlist = py310,py311,py312

[testenv]
deps = pytest
commands = pytest
```

