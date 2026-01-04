# P05: 标准库精选

> 面向 JS/TS 资深工程师的 Python 标准库教程

## 学完后能做

- 熟练使用常见标准库
- 不依赖第三方库完成常见任务
- 理解标准库设计思想

## 快速开始

```bash
cd examples
python3 01_pathlib_demo.py
```

## 目录结构

```
py-05-stdlib/
├── README.md
├── docs/
│   ├── 01-pathlib.md              # 文件路径
│   ├── 02-collections.md          # 高级容器
│   ├── 03-datetime.md             # 日期时间
│   ├── 04-re.md                   # 正则表达式
│   ├── 05-json.md                 # JSON 处理
│   ├── 06-os-shutil.md            # 系统操作
│   ├── 07-logging.md              # 日志
│   ├── 08-argparse.md             # 命令行参数
│   ├── 09-typing.md               # 类型提示基础
│   ├── 09a-typing-generics.md     # 泛型 Generic
│   ├── 09b-typing-callable.md     # Callable 与 ParamSpec
│   ├── 09c-typing-advanced.md     # 高级类型
│   ├── 09d-typing-guards-protocol.md # TypeGuard 与 Protocol
│   ├── 10-other-modules.md        # 其他模块
│   ├── 11-exercises.md            # 练习题
│   └── 12-interview-questions.md  # 面试题
├── examples/
├── exercises/
├── project/
│   └── file_organizer/
└── scripts/
```

## Python 标准库 vs JavaScript

| 功能 | Python | JavaScript/Node.js |
|------|--------|------------|
| 文件路径 | `pathlib.Path` | `path` |
| 日期时间 | `datetime` | `Date` |
| 正则表达式 | `re` | `RegExp` |
| JSON | `json` | `JSON` |
| 命令行参数 | `argparse` | `process.argv` / `yargs` |
| 日志 | `logging` | `console` / `winston` |
| 类型提示 | `typing` | TypeScript |

## 核心模块速查

### pathlib - 文件路径

```python
from pathlib import Path

p = Path("data/file.txt")
p = Path.home() / "Documents" / "file.txt"

p.exists()
p.read_text()
p.write_text("content")
```

### collections - 高级容器

```python
from collections import Counter, defaultdict, deque

counter = Counter(["a", "b", "a", "c"])
d = defaultdict(list)
queue = deque([1, 2, 3])
```

### datetime - 日期时间

```python
from datetime import datetime, timedelta

now = datetime.now()
tomorrow = now + timedelta(days=1)
formatted = now.strftime("%Y-%m-%d")
```

### re - 正则表达式

```python
import re

pattern = r"\d+"
matches = re.findall(pattern, "123 abc 456")
```

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| pathlib 路径拼接 | 字符串拼接会出错 | 使用 `/` 运算符 |
| datetime 时区 | 默认无时区 | 使用 `timezone` |
| 正则贪婪匹配 | 默认贪婪 | 使用 `?` 非贪婪 |
| json 自定义对象 | 不能直接序列化 | 使用 `default` 参数 |

## 学习路径

### 标准库
1. [pathlib](docs/01-pathlib.md)
2. [collections](docs/02-collections.md)
3. [datetime](docs/03-datetime.md)
4. [re](docs/04-re.md)
5. [json](docs/05-json.md)
6. [os 和 shutil](docs/06-os-shutil.md)
7. [logging](docs/07-logging.md)
8. [argparse](docs/08-argparse.md)
9. [typing 基础](docs/09-typing.md)
10. [其他模块](docs/10-other-modules.md)

### 类型系统深度
9a. [泛型 Generic](docs/09a-typing-generics.md) - TypeVar、Generic 类、协变/逆变
9b. [Callable 与 ParamSpec](docs/09b-typing-callable.md) - 函数类型、装饰器签名
9c. [高级类型](docs/09c-typing-advanced.md) - Literal、TypedDict、Final、overload
9d. [TypeGuard 与 Protocol](docs/09d-typing-guards-protocol.md) - 类型守卫、结构化子类型

## 小项目：文件整理器

```bash
python3 project/file_organizer/main.py ./test_files
```

按扩展名分类文件、重命名、生成报告。
