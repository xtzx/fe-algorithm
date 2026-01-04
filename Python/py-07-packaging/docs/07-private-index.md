# 07. 私有源与镜像配置

## 本节目标

- 配置 pip 使用镜像源
- 设置私有包仓库
- 处理企业内网环境

---

## pip 配置

### 配置文件位置

```
# macOS/Linux
~/.config/pip/pip.conf
# 或
~/.pip/pip.conf

# Windows
%APPDATA%\pip\pip.ini

# 虚拟环境
$VIRTUAL_ENV/pip.conf

# 项目级
./pip.conf
```

### 基本配置

```ini
# ~/.config/pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
```

---

## 国内镜像源

### 常用镜像

| 镜像 | URL |
|------|-----|
| 清华 | https://pypi.tuna.tsinghua.edu.cn/simple |
| 阿里云 | https://mirrors.aliyun.com/pypi/simple/ |
| 豆瓣 | https://pypi.doubanio.com/simple/ |
| 中科大 | https://pypi.mirrors.ustc.edu.cn/simple/ |

### 临时使用

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests
```

### 永久配置

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 多个包源

### extra-index-url

```ini
[global]
index-url = https://pypi.org/simple
extra-index-url =
    https://private.company.com/simple
    https://pypi.tuna.tsinghua.edu.cn/simple
```

### 查找顺序

1. 主 index-url
2. extra-index-url（按顺序）

### 安全注意

```ini
[global]
trusted-host =
    private.company.com
    pypi.tuna.tsinghua.edu.cn
```

---

## 私有仓库

### 使用 PyPI Server

```bash
# 安装 pypiserver
pip install pypiserver

# 启动
pypi-server run -p 8080 ./packages

# 上传包
pip install twine
twine upload --repository-url http://localhost:8080 dist/*

# 使用
pip install --extra-index-url http://localhost:8080/simple/ mypackage
```

### 使用 Artifactory/Nexus

```ini
[global]
index-url = https://artifactory.company.com/api/pypi/pypi-remote/simple
extra-index-url = https://artifactory.company.com/api/pypi/pypi-local/simple
```

### 认证

```ini
[global]
index-url = https://user:password@private.company.com/simple
```

或使用 keyring：
```bash
pip install keyring
keyring set private.company.com username
# 输入密码
```

---

## uv 配置

### 环境变量

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_EXTRA_INDEX_URL=https://private.company.com/simple
```

### 项目配置

```toml
# pyproject.toml
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
extra-index-url = ["https://private.company.com/simple"]
```

---

## poetry 配置

### 添加私有源

```bash
poetry source add private https://private.company.com/simple
```

### pyproject.toml

```toml
[[tool.poetry.source]]
name = "private"
url = "https://private.company.com/simple"
priority = "supplemental"  # 或 "primary"

[[tool.poetry.source]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
priority = "primary"
```

### 认证

```bash
poetry config http-basic.private username password
```

或使用环境变量：
```bash
export POETRY_HTTP_BASIC_PRIVATE_USERNAME=user
export POETRY_HTTP_BASIC_PRIVATE_PASSWORD=pass
```

---

## 企业环境配置

### 代理设置

```ini
[global]
proxy = http://proxy.company.com:8080
```

环境变量：
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

### SSL 证书

```ini
[global]
cert = /path/to/company-ca.crt
```

或禁用验证（不推荐）：
```bash
pip install --trusted-host private.company.com package
```

### 离线安装

```bash
# 下载包
pip download -d ./packages requests

# 离线安装
pip install --no-index --find-links=./packages requests
```

---

## CI/CD 配置

### GitHub Actions

```yaml
env:
  PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple

steps:
  - run: pip install -r requirements.txt
```

### GitLab CI

```yaml
variables:
  PIP_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple
  PIP_EXTRA_INDEX_URL: https://${CI_DEPLOY_USER}:${CI_DEPLOY_PASSWORD}@private.company.com/simple

script:
  - pip install -r requirements.txt
```

### Docker

```dockerfile
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt
```

---

## pyproject.toml 中指定源

### PEP 668 方式（未广泛支持）

```toml
# 未来标准
[project]
dependencies = [
    "public-package>=1.0",
    "private-package>=1.0",
]

[tool.pip]
index-url = "https://private.company.com/simple"
```

### 当前推荐方式

使用 pip.conf 或工具特定配置（poetry/uv）。

---

## 本节要点

1. **pip.conf** 配置默认源
2. **index-url** 主源，**extra-index-url** 额外源
3. **镜像源** 加速国内下载
4. **私有仓库** 托管内部包
5. **trusted-host** 信任非 HTTPS 源
6. **CI/CD** 使用环境变量配置

