# YAML 与 TOML 配置文件

> Python 中常用的配置文件格式

## 1. 格式对比

| 特性 | YAML | TOML | JSON |
|------|------|------|------|
| 可读性 | ⭐⭐⭐ 最佳 | ⭐⭐⭐ 很好 | ⭐⭐ 一般 |
| 注释支持 | ✓ `#` | ✓ `#` | ✗ |
| 数据类型 | 丰富 | 明确 | 基础 |
| 缩进敏感 | ✓ | ✗ | ✗ |
| Python 支持 | `pyyaml` | `tomllib` (3.11+) | 内置 |
| 典型用途 | 配置、CI/CD | 项目配置 | API、数据交换 |

## 2. YAML 处理

### 安装

```bash
pip install pyyaml
```

### 基础读写

```python
import yaml
from pathlib import Path

# YAML 文件示例
yaml_content = """
app:
  name: my-app
  version: 1.0.0
  debug: true

database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret

features:
  - logging
  - caching
  - monitoring
"""

# 从字符串解析
data = yaml.safe_load(yaml_content)
print(data["app"]["name"])  # "my-app"
print(data["features"])     # ["logging", "caching", "monitoring"]

# 从文件读取
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 写入文件
with open("output.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
```

### YAML 特殊语法

```yaml
# 多行字符串
description: |
  这是一段
  多行文本
  保留换行符

# 折叠字符串（换行变空格）
summary: >
  这是一段
  长文本
  会折叠成一行

# 锚点与引用
defaults: &defaults
  timeout: 30
  retries: 3

development:
  <<: *defaults  # 继承 defaults
  debug: true

production:
  <<: *defaults
  debug: false

# 日期时间
created_at: 2024-01-01 12:00:00
```

### 安全注意事项

```python
import yaml

# ⚠️ 危险：不要使用 yaml.load()
# data = yaml.load(content)  # 可能执行恶意代码

# ✅ 安全：使用 yaml.safe_load()
data = yaml.safe_load(content)

# 或者使用安全加载器
data = yaml.load(content, Loader=yaml.SafeLoader)
```

## 3. TOML 处理

### Python 3.11+ 内置支持

```python
import tomllib  # Python 3.11+
from pathlib import Path

# TOML 文件示例
toml_content = """
[project]
name = "my-app"
version = "1.0.0"

[project.dependencies]
requests = "^2.28.0"
pydantic = "^2.0.0"

[database]
host = "localhost"
port = 5432
enabled = true

[database.credentials]
username = "admin"
password = "secret"

[[servers]]
name = "server1"
ip = "192.168.1.1"

[[servers]]
name = "server2"
ip = "192.168.1.2"
"""

# 从字符串解析
data = tomllib.loads(toml_content)
print(data["project"]["name"])  # "my-app"
print(data["servers"])          # 列表

# 从文件读取（必须用二进制模式）
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
```

### Python 3.10 及以下

```python
# 需要安装 tomli
# pip install tomli

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python 3.10-
```

### 写入 TOML

```python
# tomllib 只支持读取，写入需要 tomli-w
# pip install tomli-w

import tomli_w

data = {
    "project": {
        "name": "my-app",
        "version": "1.0.0",
    },
    "database": {
        "host": "localhost",
        "port": 5432,
    },
}

# 生成 TOML 字符串
toml_str = tomli_w.dumps(data)
print(toml_str)

# 写入文件
with open("output.toml", "wb") as f:
    tomli_w.dump(data, f)
```

## 4. pyproject.toml 示例

现代 Python 项目的标准配置文件：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
version = "0.1.0"
description = "A sample package"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0.0",
    "requests>=2.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
my-cli = "my_package.cli:main"

[tool.ruff]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## 5. 配置文件加载模式

### 分层配置

```python
from pathlib import Path
import yaml
import tomllib


def load_config(config_path: Path) -> dict:
    """根据文件扩展名加载配置"""
    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    elif suffix == ".toml":
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    elif suffix == ".json":
        import json
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def merge_configs(*configs: dict) -> dict:
    """合并多个配置（后者覆盖前者）"""
    result = {}
    for config in configs:
        deep_merge(result, config)
    return result


def deep_merge(base: dict, override: dict) -> None:
    """深度合并字典"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


# 使用示例
default_config = load_config(Path("config/default.yaml"))
env_config = load_config(Path("config/production.yaml"))
final_config = merge_configs(default_config, env_config)
```

### 与 pydantic-settings 集成

```python
from pathlib import Path
import tomllib
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从多个来源加载配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
    )

    app_name: str = "my-app"
    debug: bool = False
    database_url: str = ""

    @classmethod
    def from_toml(cls, path: Path) -> "Settings":
        """从 TOML 文件加载"""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)


# 优先级：环境变量 > .env > TOML 默认值
settings = Settings.from_toml(Path("config.toml"))
```

## 6. 与 JS/TS 对比

| Python | JS/TS |
|--------|-------|
| `yaml.safe_load()` | `js-yaml.load()` |
| `tomllib.load()` | `@iarna/toml.parse()` |
| `pyproject.toml` | `package.json` |
| `pydantic-settings` | `dotenv` + `zod` |

## 7. 最佳实践

### 选择配置格式

```
YAML: 适合复杂、嵌套深、需要注释的配置
TOML: 适合简单、扁平的项目配置
JSON: 适合程序间数据交换，不适合手写配置
```

### 配置文件组织

```
config/
├── default.yaml      # 默认配置
├── development.yaml  # 开发环境覆盖
├── production.yaml   # 生产环境覆盖
└── local.yaml        # 本地覆盖（不提交）
```

### 敏感信息处理

```python
# ❌ 不要在配置文件中存储密码
database:
  password: "my-secret-password"

# ✅ 使用环境变量
database:
  password: ${DATABASE_PASSWORD}

# 或者使用 .env 文件
# DATABASE_PASSWORD=my-secret-password
```

## 小结

| 格式 | 适用场景 | Python 模块 |
|------|---------|-------------|
| YAML | CI/CD、复杂配置 | `pyyaml` |
| TOML | 项目配置、简单设置 | `tomllib` (3.11+) |
| JSON | API 数据、程序交换 | `json` (内置) |

