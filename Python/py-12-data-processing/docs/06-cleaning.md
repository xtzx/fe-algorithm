# 05. 数据清洗

## 本节目标

- 掌握常见数据清洗技术
- 处理空值和异常值
- 字符串规范化

---

## 数据清洗流程

```
原始数据 → 空值处理 → 类型转换 → 规范化 → 去重 → 验证 → 干净数据
```

---

## 空值处理

### 识别空值

```python
def is_empty(value) -> bool:
    """判断是否为空值"""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False

# 各种空值形式
empty_values = [None, "", "  ", "null", "NULL", "N/A", "n/a", "-"]
```

### 处理空值

```python
def clean_empty(value, default=None):
    """清洗空值"""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("null", "n/a", "none", "-"):
            return default
    return value

# 使用
clean_empty("  ")       # None
clean_empty("null")     # None
clean_empty("hello")    # "hello"
clean_empty(None, "")   # ""
```

---

## 类型转换

### 安全类型转换

```python
def safe_int(value, default: int = 0) -> int:
    """安全转换为整数"""
    if value is None:
        return default
    try:
        # 处理带小数点的字符串
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default

def safe_float(value, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default

def safe_bool(value, default: bool = False) -> bool:
    """安全转换为布尔值"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1", "on")
    return bool(value)
```

---

## 字符串规范化

```python
import re
import unicodedata

def normalize_string(s: str) -> str:
    """规范化字符串"""
    if not s:
        return ""

    # 1. 去除首尾空白
    s = s.strip()

    # 2. Unicode 规范化
    s = unicodedata.normalize("NFKC", s)

    # 3. 合并多个空格
    s = re.sub(r"\s+", " ", s)

    return s

def normalize_name(name: str) -> str:
    """规范化姓名"""
    if not name:
        return ""
    # 去空白 + 标题化
    return normalize_string(name).title()

def normalize_email(email: str) -> str:
    """规范化邮箱"""
    if not email:
        return ""
    return email.strip().lower()

def normalize_phone(phone: str) -> str:
    """规范化电话号码"""
    if not phone:
        return ""
    # 只保留数字
    return re.sub(r"\D", "", phone)
```

---

## 去重

### 基于单字段去重

```python
def dedupe_by_field(items: list[dict], field: str) -> list[dict]:
    """按字段去重，保留第一个"""
    seen = set()
    result = []
    for item in items:
        key = item.get(field)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

# 使用
data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 1, "name": "Alice"},  # 重复
]
unique = dedupe_by_field(data, "id")
# [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
```

### 基于多字段去重

```python
def dedupe_by_fields(items: list[dict], fields: list[str]) -> list[dict]:
    """按多字段去重"""
    seen = set()
    result = []
    for item in items:
        key = tuple(item.get(f) for f in fields)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
```

---

## 异常值处理

### 数值范围检查

```python
def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制值在范围内"""
    return max(min_val, min(max_val, value))

def is_outlier(value: float, values: list[float], threshold: float = 3.0) -> bool:
    """检测异常值（基于标准差）"""
    import statistics
    if len(values) < 2:
        return False
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return False
    z_score = abs(value - mean) / std
    return z_score > threshold
```

### 处理异常值

```python
def remove_outliers(values: list[float], threshold: float = 3.0) -> list[float]:
    """移除异常值"""
    import statistics
    if len(values) < 2:
        return values
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return values
    return [v for v in values if abs(v - mean) / std <= threshold]
```

---

## 数据清洗器

```python
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class CleaningRule:
    """清洗规则"""
    field: str
    cleaner: Callable[[Any], Any]
    default: Any = None

class DataCleaner:
    """数据清洗器"""

    def __init__(self, rules: list[CleaningRule]):
        self.rules = {rule.field: rule for rule in rules}

    def clean(self, data: dict) -> dict:
        """清洗单条数据"""
        result = data.copy()
        for field, rule in self.rules.items():
            value = data.get(field, rule.default)
            try:
                result[field] = rule.cleaner(value)
            except Exception:
                result[field] = rule.default
        return result

    def clean_batch(self, items: list[dict]) -> list[dict]:
        """批量清洗"""
        return [self.clean(item) for item in items]

# 使用
cleaner = DataCleaner([
    CleaningRule("name", normalize_name, ""),
    CleaningRule("email", normalize_email, ""),
    CleaningRule("age", safe_int, 0),
])

dirty_data = {"name": "  alice  ", "email": "ALICE@EXAMPLE.COM", "age": "30"}
clean_data = cleaner.clean(dirty_data)
# {"name": "Alice", "email": "alice@example.com", "age": 30}
```

---

## 编码处理

```python
def fix_encoding(s: str) -> str:
    """修复常见编码问题"""
    if not s:
        return ""

    # 尝试修复 mojibake
    try:
        # 如果是错误编码的 UTF-8
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    return s

def remove_control_chars(s: str) -> str:
    """移除控制字符"""
    import unicodedata
    return "".join(c for c in s if not unicodedata.category(c).startswith("C"))
```

---

## 本节要点

1. **空值识别** 多种空值形式
2. **安全类型转换** 避免异常
3. **字符串规范化** trim + normalize
4. **去重** 单字段和多字段
5. **异常值** 检测和移除
6. **清洗器模式** 可复用的规则

