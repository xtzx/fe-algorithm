# 03. 包格式：wheel vs sdist

## 本节目标

- 理解 wheel 和 sdist 的区别
- 了解包的构建过程
- 掌握不同 build backend 的选择

---

## 两种包格式

### sdist (Source Distribution)

源码分发包，包含源代码。

```bash
# 文件名格式
mypackage-1.0.0.tar.gz

# 包含
├── mypackage/
│   ├── __init__.py
│   └── main.py
├── pyproject.toml
├── setup.py (可选)
└── PKG-INFO
```

### wheel (Binary Distribution)

预编译分发包，安装更快。

```bash
# 文件名格式
mypackage-1.0.0-py3-none-any.whl

# 包含
├── mypackage/
│   ├── __init__.py
│   └── main.py
└── mypackage-1.0.0.dist-info/
    ├── METADATA
    ├── WHEEL
    └── RECORD
```

---

## 对比

| 特性 | wheel | sdist |
|------|-------|-------|
| 格式 | ZIP | tar.gz |
| 安装速度 | 快（解压即用） | 慢（需要构建） |
| 跨平台 | 可能受限 | 通用 |
| C 扩展 | 预编译 | 需要编译器 |
| 文件大小 | 较大（含编译产物） | 较小 |

---

## wheel 文件名解析

```
{name}-{version}-{python}-{abi}-{platform}.whl

numpy-1.24.0-cp312-cp312-macosx_11_0_arm64.whl
│     │       │     │     └── 平台：macOS ARM64
│     │       │     └── ABI：CPython 3.12
│     │       └── Python 版本：CPython 3.12
│     └── 版本
└── 包名
```

### 通用 wheel

```bash
# 纯 Python 包
mypackage-1.0.0-py3-none-any.whl
#                │   │    └── 任意平台
#                │   └── 无 ABI 要求
#                └── Python 3
```

---

## 构建包

### 使用 build

```bash
# 安装 build
pip install build

# 构建 sdist 和 wheel
python -m build

# 只构建 wheel
python -m build --wheel

# 只构建 sdist
python -m build --sdist
```

### 输出

```
dist/
├── mypackage-1.0.0-py3-none-any.whl
└── mypackage-1.0.0.tar.gz
```

---

## 构建系统 (Build Backend)

### hatchling（推荐）

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

特点：
- 现代、快速
- 原生支持 src 布局
- 简单配置

### setuptools

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

特点：
- 历史悠久，广泛支持
- 复杂项目
- C 扩展支持好

### poetry-core

```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

特点：
- poetry 生态
- 自动处理

### flit_core

```toml
[build-system]
requires = ["flit_core>=3.4"]
build-backend = "flit_core.buildapi"
```

特点：
- 极简
- 只支持纯 Python

---

## 构建后端对比

| 后端 | 速度 | C 扩展 | 复杂度 | 推荐场景 |
|------|------|--------|--------|----------|
| hatchling | 快 | 有限 | 低 | 新项目 |
| setuptools | 中 | 完美 | 中 | 复杂项目 |
| poetry-core | 快 | 有限 | 低 | poetry 用户 |
| flit | 快 | 不支持 | 极低 | 简单包 |

---

## 安装过程

### 安装 wheel

```bash
pip install package.whl
# 1. 解压 wheel
# 2. 复制到 site-packages
# 完成（无需编译）
```

### 安装 sdist

```bash
pip install package.tar.gz
# 1. 解压源码
# 2. 读取 pyproject.toml
# 3. 安装构建依赖
# 4. 调用 build backend
# 5. 生成 wheel
# 6. 安装 wheel
```

---

## C 扩展包

对于包含 C/C++ 代码的包：

### 平台特定 wheel

```bash
# 预编译的 NumPy
numpy-1.24.0-cp312-cp312-macosx_11_0_arm64.whl  # macOS ARM
numpy-1.24.0-cp312-cp312-manylinux_x86_64.whl   # Linux x86
numpy-1.24.0-cp312-cp312-win_amd64.whl          # Windows
```

### 没有 wheel 时

```bash
pip install some-c-package
# 需要：
# - C 编译器（gcc/clang/MSVC）
# - 相关开发库
```

---

## 发布到 PyPI

### 使用 twine

```bash
# 安装 twine
pip install twine

# 检查包
twine check dist/*

# 上传到 TestPyPI
twine upload --repository testpypi dist/*

# 上传到 PyPI
twine upload dist/*
```

### 使用 hatch

```bash
hatch publish
```

### 使用 poetry

```bash
poetry publish
```

---

## editable install

开发时安装，修改源码立即生效：

```bash
pip install -e .
# 或
pip install -e ".[dev]"
```

原理：
- 不复制代码到 site-packages
- 创建 .egg-link 或 .pth 文件指向源码
- 修改立即生效

---

## 本节要点

1. **wheel**: 预编译，安装快
2. **sdist**: 源码，需要构建
3. **纯 Python 包**: py3-none-any.whl
4. **C 扩展包**: 平台特定 wheel
5. **build backend**: hatchling（推荐）、setuptools
6. **editable install**: 开发时使用 `-e`

