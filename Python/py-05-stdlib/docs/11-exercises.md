# 11. 练习题

## pathlib（5 道）

### 练习 1：统计文件类型

统计指定目录下各类型文件的数量。

```python
from pathlib import Path
from collections import Counter

def count_file_types(directory: str) -> dict[str, int]:
    """统计目录下各扩展名文件数量"""
    # 你的代码
    pass

# 测试
# count_file_types(".")
# 输出：{'.py': 10, '.md': 5, '.txt': 3}
```

### 练习 2：查找大文件

查找目录中超过指定大小的文件。

```python
from pathlib import Path

def find_large_files(directory: str, min_size_mb: float) -> list[Path]:
    """查找大于 min_size_mb MB 的文件"""
    # 你的代码
    pass
```

### 练习 3：批量重命名

将目录下所有 `.jpeg` 文件重命名为 `.jpg`。

```python
from pathlib import Path

def rename_extensions(directory: str, old_ext: str, new_ext: str) -> int:
    """重命名扩展名，返回重命名的文件数"""
    # 你的代码
    pass
```

### 练习 4：目录树

生成目录树结构字符串。

```python
from pathlib import Path

def generate_tree(directory: str, prefix: str = "") -> str:
    """生成目录树字符串"""
    # 你的代码
    pass

# 输出示例：
# mydir/
# ├── file1.txt
# ├── subdir/
# │   └── file2.txt
# └── file3.txt
```

### 练习 5：同步目录

检查两个目录的差异（哪些文件只在一个目录中）。

```python
from pathlib import Path

def compare_directories(dir1: str, dir2: str) -> dict:
    """比较两个目录，返回只在各自目录中的文件"""
    # 你的代码
    pass

# 输出：{'only_in_dir1': [...], 'only_in_dir2': [...]}
```

---

## collections（5 道）

### 练习 6：词频统计

统计文本中出现频率最高的 N 个单词。

```python
from collections import Counter

def top_words(text: str, n: int) -> list[tuple[str, int]]:
    """返回出现频率最高的 n 个单词"""
    # 你的代码
    pass
```

### 练习 7：分组统计

按首字母分组单词。

```python
from collections import defaultdict

def group_by_first_letter(words: list[str]) -> dict[str, list[str]]:
    """按首字母分组"""
    # 你的代码
    pass

# 输入：["apple", "banana", "avocado", "blueberry"]
# 输出：{'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry']}
```

### 练习 8：滑动窗口

实现滑动窗口最大值。

```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """返回大小为 k 的滑动窗口的最大值列表"""
    # 你的代码
    pass

# 输入：nums=[1,3,-1,-3,5,3,6,7], k=3
# 输出：[3,3,5,5,6,7]
```

### 练习 9：LRU 缓存

使用 OrderedDict 实现 LRU 缓存。

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        # 你的代码
        pass

    def put(self, key: str, value):
        # 你的代码
        pass
```

### 练习 10：配置合并

使用 ChainMap 实现多级配置。

```python
from collections import ChainMap

def get_config(cli_args: dict, env_vars: dict, defaults: dict) -> ChainMap:
    """创建配置链，优先级：cli > env > defaults"""
    # 你的代码
    pass
```

---

## datetime（5 道）

### 练习 11：工作日计算

计算两个日期之间的工作日数量。

```python
from datetime import date

def count_workdays(start: date, end: date) -> int:
    """计算工作日数量（不含周末）"""
    # 你的代码
    pass
```

### 练习 12：日期格式化

将各种格式的日期字符串转换为标准格式。

```python
from datetime import datetime

def normalize_date(date_str: str) -> str:
    """将日期字符串转换为 YYYY-MM-DD 格式"""
    # 支持格式：2024-01-15, 15/01/2024, Jan 15, 2024
    # 你的代码
    pass
```

### 练习 13：年龄计算

计算精确年龄（年、月、日）。

```python
from datetime import date

def calculate_age(birthday: date) -> tuple[int, int, int]:
    """返回 (年, 月, 日)"""
    # 你的代码
    pass
```

### 练习 14：时区转换

将时间从一个时区转换到另一个时区。

```python
from datetime import datetime
from zoneinfo import ZoneInfo

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """时区转换"""
    # 你的代码
    pass

# convert_timezone(dt, "Asia/Shanghai", "America/New_York")
```

### 练习 15：日程冲突

检测日程是否有冲突。

```python
from datetime import datetime

def has_conflict(schedules: list[tuple[datetime, datetime]]) -> bool:
    """检测日程列表是否有冲突"""
    # 你的代码
    pass
```

---

## re（5 道）

### 练习 16：提取邮箱

从文本中提取所有邮箱地址。

```python
import re

def extract_emails(text: str) -> list[str]:
    """提取所有邮箱地址"""
    # 你的代码
    pass
```

### 练习 17：密码验证

验证密码强度（至少 8 位，包含大小写字母和数字）。

```python
import re

def validate_password(password: str) -> bool:
    """验证密码强度"""
    # 你的代码
    pass
```

### 练习 18：Markdown 链接转换

将 Markdown 链接转换为 HTML 链接。

```python
import re

def markdown_to_html_links(text: str) -> str:
    """将 [text](url) 转换为 <a href="url">text</a>"""
    # 你的代码
    pass
```

### 练习 19：日志解析

解析日志行，提取时间、级别和消息。

```python
import re

def parse_log_line(line: str) -> dict:
    """解析日志行"""
    # 输入：2024-01-15 14:30:45 [INFO] User logged in
    # 输出：{'time': '2024-01-15 14:30:45', 'level': 'INFO', 'message': 'User logged in'}
    # 你的代码
    pass
```

### 练习 20：模板替换

实现简单的模板变量替换。

```python
import re

def render_template(template: str, variables: dict) -> str:
    """替换 {{variable}} 为对应的值"""
    # 你的代码
    pass

# 输入：template="Hello, {{name}}!", variables={"name": "Alice"}
# 输出："Hello, Alice!"
```

---

## 综合（5 道）

### 练习 21：配置文件解析器

解析 INI 格式的配置文件。

```python
from pathlib import Path

def parse_ini(path: str) -> dict[str, dict[str, str]]:
    """解析 INI 文件"""
    # 你的代码
    pass

# 输入文件：
# [database]
# host = localhost
# port = 5432
```

### 练习 22：简单日志系统

实现一个简单的日志系统。

```python
import logging
from pathlib import Path

class SimpleLogger:
    def __init__(self, name: str, log_file: str = None):
        # 你的代码
        pass

    def info(self, message: str): pass
    def error(self, message: str): pass
    def debug(self, message: str): pass
```

### 练习 23：命令行计算器

使用 argparse 实现命令行计算器。

```python
import argparse

def main():
    # python calc.py add 1 2
    # python calc.py mul 3 4
    # 你的代码
    pass
```

### 练习 24：文件哈希比较

比较两个目录中同名文件的哈希值。

```python
from pathlib import Path
import hashlib

def compare_directories_by_hash(dir1: str, dir2: str) -> dict:
    """比较两个目录中同名文件是否相同"""
    # 返回：{'same': [...], 'different': [...], 'only_in_dir1': [...], 'only_in_dir2': [...]}
    # 你的代码
    pass
```

### 练习 25：API Token 生成器

生成安全的 API Token。

```python
import secrets
import hashlib
from datetime import datetime

def generate_api_token(user_id: str) -> str:
    """生成 API Token"""
    # 你的代码
    pass

def verify_api_token(token: str) -> bool:
    """验证 Token 格式"""
    # 你的代码
    pass
```

---

## 参考答案提示

### 练习 1 答案

```python
def count_file_types(directory: str) -> dict[str, int]:
    counter = Counter()
    for f in Path(directory).rglob("*"):
        if f.is_file():
            counter[f.suffix or "no_ext"] += 1
    return dict(counter)
```

### 练习 6 答案

```python
def top_words(text: str, n: int) -> list[tuple[str, int]]:
    words = text.lower().split()
    counter = Counter(words)
    return counter.most_common(n)
```

### 练习 11 答案

```python
def count_workdays(start: date, end: date) -> int:
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0-4 是周一到周五
            count += 1
        current += timedelta(days=1)
    return count
```


