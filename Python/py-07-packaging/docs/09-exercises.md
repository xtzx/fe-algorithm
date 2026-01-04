# 09. 练习题

## 练习 1：创建完整项目

从零开始创建一个 Python 包项目：

1. 使用 src 布局创建目录结构
2. 编写 pyproject.toml
3. 创建简单的 CLI 命令
4. 添加开发依赖（pytest, ruff）
5. 使用 editable install 安装

**预期结构**：
```
mypackage/
├── pyproject.toml
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── cli.py
└── tests/
    └── test_cli.py
```

---

## 练习 2：依赖锁定

1. 创建包含多个依赖的 pyproject.toml
2. 使用 uv 生成 requirements.lock
3. 在新虚拟环境中使用 lockfile 安装
4. 验证依赖版本一致

```bash
# 参考命令
uv pip compile pyproject.toml -o requirements.lock
uv pip sync requirements.lock
pip list
```

---

## 练习 3：多环境依赖

1. 创建包含 dev、test、docs 组的 pyproject.toml
2. 分别生成各环境的 lockfile
3. 测试安装不同组合

```toml
[project.optional-dependencies]
dev = ["ruff", "mypy"]
test = ["pytest", "coverage"]
docs = ["mkdocs"]
```

---

## 练习 4：CLI 工具

创建一个命令行工具，包含多个子命令：

```python
# 目标：
# mycli greet NAME --times 3
# mycli version
# mycli config show
```

使用 click 或 typer 实现。

---

## 练习 5：私有源配置

1. 创建 pip.conf 配置国内镜像
2. 添加 extra-index-url
3. 测试安装速度

```ini
# ~/.config/pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 练习 6：构建和分发

1. 构建 wheel 和 sdist
2. 检查构建产物
3. 在新环境中安装 wheel

```bash
python -m build
ls dist/
pip install dist/*.whl
```

---

## 练习 7：Poetry 工作流

使用 poetry 管理项目：

1. `poetry init` 初始化
2. `poetry add` 添加依赖
3. `poetry add --group dev` 添加开发依赖
4. `poetry lock` 锁定
5. `poetry install` 安装
6. `poetry run` 运行命令

---

## 练习 8：CI 配置

编写 GitHub Actions 配置：

1. 缓存依赖
2. 安装项目
3. 运行测试
4. 构建包

```yaml
# .github/workflows/ci.yml
name: CI
on: [push]
jobs:
  test:
    # 完成配置
```

---

## 练习 9：Docker 化

创建 Dockerfile：

1. 使用多阶段构建
2. 安装依赖
3. 复制代码
4. 设置入口点

---

## 练习 10：版本管理

实现动态版本：

1. 在 `__init__.py` 定义 `__version__`
2. 配置 pyproject.toml 动态读取
3. CLI 中输出版本号

```toml
[project]
dynamic = ["version"]

[tool.hatch.version]
path = "src/mypackage/__init__.py"
```

