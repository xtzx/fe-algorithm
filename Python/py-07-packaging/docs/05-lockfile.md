# 05. lockfile 最佳实践

## 本节目标

- 理解为什么需要 lockfile
- 掌握 lockfile 的生成和使用
- 了解最佳实践

---

## 为什么需要 lockfile

### 问题：依赖漂移

```toml
# pyproject.toml
dependencies = ["requests>=2.28"]

# 今天安装
pip install .
# 安装 requests==2.28.0

# 一个月后安装
pip install .
# 安装 requests==2.31.0（可能有 bug）
```

### 解决：锁定版本

```txt
# requirements.lock
requests==2.28.0
certifi==2023.7.22
charset-normalizer==3.2.0
idna==3.4
urllib3==2.0.4
```

---

## 对比 package-lock.json

| 特性 | Python lockfile | npm package-lock.json |
|------|-----------------|----------------------|
| 锁定范围 | 完整依赖树 | 完整依赖树 |
| 生成工具 | uv/poetry/pip-tools | npm/yarn/pnpm |
| 文件格式 | txt/toml | json |
| 自动更新 | 需手动 | npm install 时 |

---

## 使用 uv 锁定

### 生成 lockfile

```bash
# 从 pyproject.toml 生成
uv pip compile pyproject.toml -o requirements.lock

# 包含开发依赖
uv pip compile pyproject.toml --extra dev -o requirements-dev.lock

# 指定 Python 版本
uv pip compile pyproject.toml --python-version 3.12 -o requirements.lock
```

### 安装精确版本

```bash
# 从 lockfile 安装
uv pip sync requirements.lock

# 开发环境
uv pip sync requirements-dev.lock
```

### 更新依赖

```bash
# 更新所有
uv pip compile pyproject.toml --upgrade -o requirements.lock

# 只更新特定包
uv pip compile pyproject.toml --upgrade-package requests -o requirements.lock
```

---

## 使用 poetry 锁定

### 自动生成 poetry.lock

```bash
# 添加依赖时自动更新
poetry add requests

# 手动更新锁定
poetry lock

# 不更新，只重新生成
poetry lock --no-update
```

### poetry.lock 结构

```toml
[[package]]
name = "requests"
version = "2.28.0"
description = "Python HTTP for Humans."
python-versions = ">=3.7"

[package.dependencies]
certifi = ">=2017.4.17"
charset-normalizer = ">=2,<3"
idna = ">=2.5,<4"
urllib3 = ">=1.21.1,<1.27"

[metadata]
lock-version = "2.0"
python-versions = "^3.10"
content-hash = "..."
```

---

## 使用 pip-tools 锁定

### 工作流

```bash
# 1. 创建 requirements.in
cat > requirements.in << EOF
requests>=2.28
fastapi>=0.100
EOF

# 2. 编译锁定
pip-compile requirements.in -o requirements.txt

# 3. 开发依赖
cat > requirements-dev.in << EOF
-c requirements.txt  # 约束基础依赖
pytest>=7.0
ruff>=0.1
EOF
pip-compile requirements-dev.in -o requirements-dev.txt

# 4. 安装
pip-sync requirements.txt requirements-dev.txt
```

---

## 多环境锁定

### 策略一：分离 lockfile

```
requirements/
├── base.in          # 基础依赖
├── dev.in           # 开发依赖
├── test.in          # 测试依赖
├── requirements.txt     # 编译后
├── requirements-dev.txt
└── requirements-test.txt
```

### 策略二：约束文件

```bash
# requirements.in
requests>=2.28

# requirements-dev.in
-c requirements.txt  # 使用 base 作为约束
pytest>=7.0
```

### 策略三：poetry 分组

```toml
[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.28"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"

[tool.poetry.group.test.dependencies]
coverage = "^7.0"
```

---

## 处理私有包

### 方法一：直接 URL

```toml
# pyproject.toml
dependencies = [
    "private-pkg @ git+https://github.com/company/private-pkg.git@v1.0.0",
]
```

### 方法二：私有 index

```bash
# pip.conf
[global]
extra-index-url = https://pypi.company.com/simple/
```

```bash
# 编译时指定
pip-compile --extra-index-url https://pypi.company.com/simple/
```

### 方法三：本地包

```toml
dependencies = [
    "local-pkg @ file:///path/to/local-pkg",
]
```

---

## CI/CD 最佳实践

### 缓存策略

```yaml
# GitHub Actions
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('requirements.lock') }}

- name: Install dependencies
  run: |
    pip install uv
    uv pip sync requirements.lock
```

### 可重复构建

```yaml
# 使用锁定文件
- run: uv pip sync requirements.lock

# 不要这样做
- run: pip install -r requirements.txt  # 可能版本不同
```

### 定期更新

```yaml
# 每周自动更新依赖
name: Update Dependencies
on:
  schedule:
    - cron: '0 0 * * 1'  # 每周一

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          pip install uv
          uv pip compile pyproject.toml --upgrade -o requirements.lock
      - uses: peter-evans/create-pull-request@v5
        with:
          title: Update dependencies
```

---

## lockfile 检查

### 验证一致性

```bash
# pip-compile 检查
pip-compile --dry-run requirements.in

# poetry 检查
poetry lock --check

# 输出差异
diff requirements.txt <(pip-compile --dry-run requirements.in)
```

### CI 中验证

```yaml
- name: Check lockfile is up to date
  run: |
    uv pip compile pyproject.toml -o /tmp/requirements.lock
    diff requirements.lock /tmp/requirements.lock
```

---

## 常见问题

### 依赖冲突

```bash
# 查看冲突
pip check

# 查看依赖树
pip install pipdeptree
pipdeptree
```

### lockfile 过期

```bash
# 定期更新
uv pip compile pyproject.toml --upgrade -o requirements.lock

# 只更新安全补丁
# 手动检查 CVE 并更新特定包
```

### 平台差异

```bash
# 生成特定平台的 lockfile
pip-compile --python-platform linux requirements.in -o requirements-linux.txt
pip-compile --python-platform darwin requirements.in -o requirements-darwin.txt
```

---

## 本节要点

1. **lockfile** 锁定完整依赖树，避免漂移
2. **uv pip compile** 生成锁定文件
3. **uv pip sync** 安装精确版本
4. **poetry lock** 自动管理
5. **CI 缓存** 基于 lockfile 哈希
6. **定期更新** 依赖以获取安全修复

