# 02. logging 最佳实践

## 本节目标

- 理解日志级别策略
- 配置生产级日志
- 处理第三方库日志

---

## logging vs print

| 特性 | logging | print |
|------|---------|-------|
| 级别控制 | ✓ | ✗ |
| 输出目标 | 多种 Handler | 只有 stdout |
| 格式化 | 丰富 | 基础 |
| 线程安全 | ✓ | ✓ |
| 生产环境 | ✓ | ✗ |
| 性能 | 可禁用 | 始终执行 |

---

## 日志级别

| 级别 | 数值 | 用途 |
|------|-----|------|
| DEBUG | 10 | 详细调试信息，生产环境禁用 |
| INFO | 20 | 一般运行信息 |
| WARNING | 30 | 警告，程序仍可运行 |
| ERROR | 40 | 错误，部分功能失败 |
| CRITICAL | 50 | 严重错误，程序可能无法继续 |

### 级别选择原则

```python
import logging

# DEBUG: 调试信息
logging.debug(f"处理数据: {data}")

# INFO: 正常操作
logging.info("用户登录成功")

# WARNING: 需要注意
logging.warning("磁盘空间不足 10%")

# ERROR: 错误但程序继续
logging.error(f"请求失败: {error}")

# CRITICAL: 严重错误
logging.critical("数据库连接丢失")
```

---

## 基本配置

### 快速配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Hello, logging!")
```

### 输出示例

```
2024-01-01 12:00:00 - root - INFO - Hello, logging!
```

---

## 格式化选项

| 占位符 | 说明 |
|--------|------|
| `%(name)s` | Logger 名称 |
| `%(levelname)s` | 日志级别 |
| `%(message)s` | 日志消息 |
| `%(asctime)s` | 时间 |
| `%(filename)s` | 文件名 |
| `%(lineno)d` | 行号 |
| `%(funcName)s` | 函数名 |
| `%(process)d` | 进程 ID |
| `%(thread)d` | 线程 ID |

### 推荐格式

```python
# 开发环境
DEV_FORMAT = "%(levelname)s - %(name)s - %(message)s"

# 生产环境
PROD_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
```

---

## Logger 层级

```python
import logging

# 获取 Logger（推荐使用 __name__）
logger = logging.getLogger(__name__)

# Logger 层级
# root
# └── myapp
#     ├── myapp.module1
#     └── myapp.module2
```

### 模块最佳实践

```python
# myapp/module.py
import logging

logger = logging.getLogger(__name__)  # myapp.module

def do_something():
    logger.info("Doing something")
```

---

## Handler 配置

### 多个输出目标

```python
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# 创建 logger
logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)

# 控制台 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(console_format)

# 文件 Handler（轮转）
file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_format)

# 添加 Handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)
```

### 常用 Handler

| Handler | 用途 |
|---------|------|
| StreamHandler | 输出到控制台 |
| FileHandler | 输出到文件 |
| RotatingFileHandler | 按大小轮转 |
| TimedRotatingFileHandler | 按时间轮转 |
| SMTPHandler | 发送邮件 |
| HTTPHandler | 发送到 HTTP 服务 |

---

## 使用配置文件

### logging.conf（INI 格式）

```ini
[loggers]
keys=root,myapp

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=WARNING
handlers=console

[logger_myapp]
level=DEBUG
handlers=console,file
qualname=myapp
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=standard
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=DEBUG
formatter=standard
args=('app.log', 'a', 10485760, 5)

[formatter_standard]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

```python
import logging.config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("myapp")
```

### 字典配置

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
        },
    },
    "loggers": {
        "myapp": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## 结构化日志

### 使用 extra

```python
logger.info(
    "用户操作",
    extra={
        "user_id": 123,
        "action": "login",
        "ip": "192.168.1.1",
    }
)
```

### 使用 structlog（推荐）

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()
logger.info("user_login", user_id=123, ip="192.168.1.1")
# {"event": "user_login", "user_id": 123, "ip": "192.168.1.1", "timestamp": "..."}
```

---

## 处理第三方库日志

```python
import logging

# 降低第三方库日志级别
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# 或完全禁用
logging.getLogger("noisy_library").disabled = True
```

---

## 生产环境配置

```python
import logging
import sys
import os

def setup_logging():
    """配置生产环境日志"""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 根 logger 配置
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 降低第三方库级别
    for name in ["urllib3", "requests", "httpx"]:
        logging.getLogger(name).setLevel(logging.WARNING)

# 应用启动时调用
setup_logging()
```

---

## 本节要点

1. **使用 logging 而非 print** 生产环境必须
2. **Logger 层级** 使用 `__name__` 命名
3. **Handler** 控制输出目标
4. **级别策略** DEBUG 开发，INFO/WARNING 生产
5. **结构化日志** 便于分析和检索
6. **第三方库** 适当降低日志级别

