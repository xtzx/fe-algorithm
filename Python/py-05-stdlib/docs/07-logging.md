# 07. logging - 日志

## 本节目标

- 掌握日志级别和基本配置
- 学会使用 Logger、Handler、Formatter
- 了解最佳实践

---

## 日志级别

```python
import logging

# 级别（从低到高）
logging.DEBUG     # 10 - 调试信息
logging.INFO      # 20 - 常规信息
logging.WARNING   # 30 - 警告（默认级别）
logging.ERROR     # 40 - 错误
logging.CRITICAL  # 50 - 严重错误
```

---

## 基本使用

### basicConfig

```python
import logging

# 配置（只能调用一次）
logging.basicConfig(level=logging.DEBUG)

# 输出日志
logging.debug("调试信息")
logging.info("普通信息")
logging.warning("警告信息")
logging.error("错误信息")
logging.critical("严重错误")
```

### 配置选项

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",          # 输出到文件
    filemode="a",                # 追加模式
    encoding="utf-8"             # Python 3.9+
)
```

### 常用格式符

| 格式符 | 含义 |
|--------|------|
| `%(asctime)s` | 时间 |
| `%(name)s` | Logger 名称 |
| `%(levelname)s` | 级别名称 |
| `%(message)s` | 日志消息 |
| `%(filename)s` | 文件名 |
| `%(lineno)d` | 行号 |
| `%(funcName)s` | 函数名 |

---

## Logger 对象

```python
import logging

# 获取 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建 formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# 添加 handler
logger.addHandler(console_handler)

# 使用
logger.info("Hello, logging!")
```

---

## Handler 类型

```python
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 文件输出
file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# 按大小轮转
rotating_handler = RotatingFileHandler(
    "app.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5           # 保留 5 个备份
)

# 按时间轮转
timed_handler = TimedRotatingFileHandler(
    "app.log",
    when="midnight",        # 每天轮转
    interval=1,
    backupCount=30          # 保留 30 天
)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
```

---

## 异常日志

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = 1 / 0
except Exception:
    # 自动记录异常信息
    logger.exception("发生错误")

    # 或者
    logger.error("发生错误", exc_info=True)
```

---

## 结构化日志

```python
import logging

logger = logging.getLogger(__name__)

# 使用 extra 添加额外字段
logger.info("用户登录", extra={"user_id": 123, "ip": "192.168.1.1"})

# 自定义 formatter 支持 extra
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        result = super().format(record)
        if hasattr(record, "user_id"):
            result += f" [user_id={record.user_id}]"
        return result
```

---

## 最佳实践

### 1. 使用 __name__ 作为 logger 名称

```python
import logging

# 推荐
logger = logging.getLogger(__name__)

# 不推荐
logger = logging.getLogger("my_logger")
```

### 2. 模块级配置

```python
# myapp/__init__.py
import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# myapp/module.py
import logging

logger = logging.getLogger(__name__)

def do_something():
    logger.info("Doing something")
```

### 3. 懒惰求值

```python
import logging

logger = logging.getLogger(__name__)

# 不好：总是计算参数
logger.debug("Data: %s" % expensive_operation())

# 好：只在需要时计算
logger.debug("Data: %s", expensive_operation())
```

### 4. 不要在库中配置 logging

```python
# 库代码中
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # 避免 "No handler found" 警告

# 让使用者自己配置
```

---

## 完整示例

```python
import logging
import sys
from pathlib import Path

def setup_logging(
    level: str = "INFO",
    log_file: str = None
):
    """设置日志配置"""

    # 创建 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # 格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# 使用
if __name__ == "__main__":
    setup_logging(level="DEBUG", log_file="app.log")

    logger = logging.getLogger(__name__)
    logger.debug("调试信息")
    logger.info("应用启动")
```

---

## 本节要点

1. 五个级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
2. `basicConfig` 快速配置（只能调用一次）
3. Logger + Handler + Formatter 灵活配置
4. 使用 `__name__` 作为 logger 名称
5. `logger.exception()` 记录异常堆栈
6. 库代码添加 `NullHandler`


