# 12. 面试题

## 1. pathlib 和 os.path 的区别？

**答案**：

| 特性 | pathlib | os.path |
|------|---------|---------|
| 风格 | 面向对象 | 函数式 |
| 路径拼接 | `/` 运算符 | `os.path.join()` |
| 链式调用 | 支持 | 不支持 |
| 文件操作 | `read_text()` 等 | 需配合 `open()` |
| 版本 | Python 3.4+ | 一直存在 |

```python
# os.path
import os.path
path = os.path.join("dir", "file.txt")
if os.path.exists(path):
    with open(path) as f:
        content = f.read()

# pathlib（推荐）
from pathlib import Path
path = Path("dir") / "file.txt"
if path.exists():
    content = path.read_text()
```

**结论**：新代码推荐使用 pathlib，更直观、更 Pythonic。

---

## 2. Counter 最常见的用法？

**答案**：

```python
from collections import Counter

# 1. 统计元素频率
words = ["apple", "banana", "apple", "cherry"]
counter = Counter(words)
print(counter)  # Counter({'apple': 2, 'banana': 1, 'cherry': 1})

# 2. 获取最常见元素
print(counter.most_common(2))  # [('apple', 2), ('banana', 1)]

# 3. 统计字符
char_count = Counter("hello")
print(char_count)  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

# 4. 数学运算
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)  # Counter({'a': 4, 'b': 3})
print(c1 - c2)  # Counter({'a': 2})

# 5. 检查是否是子集（Python 3.10+）
c1 <= c2  # c1 是 c2 的子集？
```

---

## 3. 如何处理 Python 的时区问题？

**答案**：

```python
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

# 1. 创建 aware datetime（带时区）
utc_now = datetime.now(timezone.utc)
shanghai_now = datetime.now(ZoneInfo("Asia/Shanghai"))

# 2. 给 naive datetime 添加时区
naive = datetime(2024, 1, 15, 12, 0)
aware = naive.replace(tzinfo=ZoneInfo("Asia/Shanghai"))

# 3. 时区转换
utc_time = datetime(2024, 1, 15, 4, 0, tzinfo=timezone.utc)
shanghai_time = utc_time.astimezone(ZoneInfo("Asia/Shanghai"))
print(shanghai_time)  # 2024-01-15 12:00:00+08:00

# 4. 存储和传输：使用 UTC
# 5. 显示：转换为用户时区
```

**最佳实践**：
- 内部处理用 UTC
- 存储用 ISO 格式或时间戳
- 显示时转换为用户时区
- Python 3.9+ 用 `zoneinfo`，之前用 `pytz`

---

## 4. 正则表达式的贪婪和非贪婪？

**答案**：

```python
import re

text = "<div>content</div>"

# 贪婪匹配（默认）：匹配尽可能多
pattern = r"<.*>"
print(re.search(pattern, text).group())  # <div>content</div>

# 非贪婪匹配：匹配尽可能少，加 ?
pattern = r"<.*?>"
print(re.search(pattern, text).group())  # <div>
```

| 贪婪 | 非贪婪 | 含义 |
|------|--------|------|
| `*` | `*?` | 0 或多个 |
| `+` | `+?` | 1 或多个 |
| `?` | `??` | 0 或 1 个 |
| `{n,m}` | `{n,m}?` | n 到 m 个 |

---

## 5. logging 的最佳实践是什么？

**答案**：

```python
import logging

# 1. 使用 __name__ 作为 logger 名称
logger = logging.getLogger(__name__)

# 2. 不在库代码中配置 logging
logger.addHandler(logging.NullHandler())

# 3. 在应用入口配置一次
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 4. 使用 % 格式而非 f-string（懒惰求值）
logger.debug("Processing %s", expensive_function())

# 5. 使用 exception() 记录异常
try:
    risky_operation()
except Exception:
    logger.exception("操作失败")

# 6. 生产环境使用文件轮转
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
```

---

## 6. 如何解析命令行参数？

**答案**：

```python
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="我的程序")

# 位置参数（必填）
parser.add_argument("input", help="输入文件")

# 可选参数
parser.add_argument("-o", "--output", help="输出文件")
parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
parser.add_argument("-n", "--number", type=int, default=10, help="数量")
parser.add_argument("--format", choices=["json", "csv"], default="json")

# 解析
args = parser.parse_args()

# 使用
print(args.input)
print(args.output)
print(args.verbose)
```

**其他选项**：
- `sys.argv` - 最基础
- `click` - 第三方，装饰器风格
- `typer` - 第三方，基于类型提示

---

## 7. typing 模块中 Optional 和 Union 的区别？

**答案**：

```python
from typing import Optional, Union

# Optional[X] = Union[X, None]
# 表示可能为 None

def find_user(id: int) -> Optional[str]:
    """返回用户名或 None"""
    pass

# Union[X, Y, Z]
# 表示可能是 X、Y 或 Z 中的任一种

def process(value: Union[int, str, float]) -> str:
    """接受 int、str 或 float"""
    pass
```

**Python 3.10+ 语法**：

```python
# Optional[str] 等价于
def func() -> str | None:
    pass

# Union[int, str] 等价于
def func(value: int | str):
    pass
```

**区别总结**：
- `Optional[X]` 是 `Union[X, None]` 的简写
- `Optional` 语义更明确：表示"可能没有值"
- `Union` 更通用：表示"多种类型之一"

---

## 8. 如何生成安全的随机数？

**答案**：

```python
import secrets

# 1. 生成安全的随机 token
token = secrets.token_urlsafe(32)  # URL 安全的 32 字节 token
token = secrets.token_hex(16)       # 16 字节的十六进制字符串

# 2. 生成随机整数
n = secrets.randbelow(100)  # [0, 100)

# 3. 安全地比较字符串（防止时序攻击）
secrets.compare_digest(user_input, stored_password)

# 4. 安全选择
api_keys = ["key1", "key2", "key3"]
selected = secrets.choice(api_keys)
```

**random vs secrets**：

| 场景 | 模块 |
|------|------|
| 模拟、游戏 | `random` |
| 密码、令牌 | `secrets` |
| 加密密钥 | `secrets` |
| 验证码 | `secrets` |
| 抽奖 | `secrets`（公平） |

---

## 9. json.dumps 处理自定义对象？

**答案**：

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

# 方法 1：使用 default 参数
def custom_encoder(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

data = {"time": datetime.now()}
json.dumps(data, default=custom_encoder)

# 方法 2：自定义 JSONEncoder
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json.dumps(data, cls=CustomEncoder)

# 方法 3：dataclass + asdict
@dataclass
class User:
    name: str
    age: int

user = User("Alice", 25)
json.dumps(asdict(user))
```

---

## 10. 如何用 Python 处理环境变量？

**答案**：

```python
import os

# 1. 读取环境变量
value = os.environ.get("MY_VAR")           # 不存在返回 None
value = os.environ.get("MY_VAR", "default") # 带默认值
value = os.getenv("MY_VAR", "default")      # 同上

# 2. 必需的环境变量
try:
    api_key = os.environ["API_KEY"]
except KeyError:
    raise RuntimeError("API_KEY environment variable is required")

# 3. 设置环境变量（仅当前进程）
os.environ["MY_VAR"] = "value"

# 4. 删除环境变量
del os.environ["MY_VAR"]

# 5. 遍历所有环境变量
for key, value in os.environ.items():
    print(f"{key}={value}")

# 6. 实际项目：使用 python-dotenv
# pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件
```

**最佳实践**：
- 敏感信息用环境变量，不要硬编码
- 使用 `.env` 文件本地开发
- `.env` 文件加入 `.gitignore`
- 提供 `.env.example` 模板


