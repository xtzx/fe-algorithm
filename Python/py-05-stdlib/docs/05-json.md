# 05. json - JSON 处理

## 本节目标

- 掌握 JSON 序列化和反序列化
- 学会处理自定义对象
- 了解格式化选项

---

## 基本操作

### dumps - 对象转 JSON 字符串

```python
import json

data = {
    "name": "Alice",
    "age": 25,
    "skills": ["Python", "JavaScript"]
}

# 转为 JSON 字符串
json_str = json.dumps(data)
print(json_str)
# {"name": "Alice", "age": 25, "skills": ["Python", "JavaScript"]}
```

### loads - JSON 字符串转对象

```python
import json

json_str = '{"name": "Alice", "age": 25}'

# 转为 Python 对象
data = json.loads(json_str)
print(data)        # {'name': 'Alice', 'age': 25}
print(data["name"]) # Alice
```

---

## 文件操作

### dump - 写入文件

```python
import json

data = {"name": "Alice", "age": 25}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
```

### load - 从文件读取

```python
import json

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data)
```

---

## 格式化输出

### indent - 缩进

```python
import json

data = {"name": "Alice", "items": [1, 2, 3]}

# 格式化输出
print(json.dumps(data, indent=2))
# {
#   "name": "Alice",
#   "items": [
#     1,
#     2,
#     3
#   ]
# }
```

### 其他参数

```python
import json

data = {"name": "张三", "age": 25}

# ensure_ascii：中文不转义
print(json.dumps(data, ensure_ascii=False))
# {"name": "张三", "age": 25}

# sort_keys：键排序
data = {"b": 2, "a": 1}
print(json.dumps(data, sort_keys=True))
# {"a": 1, "b": 2}

# separators：自定义分隔符（紧凑）
print(json.dumps(data, separators=(",", ":")))
# {"b":2,"a":1}
```

---

## 自定义对象序列化

Python 默认不能序列化自定义对象。

### 方法一：default 参数

```python
import json
from datetime import datetime

data = {
    "name": "Alice",
    "created": datetime.now()
}

# 使用 default 处理无法序列化的对象
def json_encoder(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json_str = json.dumps(data, default=json_encoder)
print(json_str)
# {"name": "Alice", "created": "2024-01-15T14:30:45.123456"}
```

### 方法二：自定义 JSONEncoder

```python
import json
from datetime import datetime, date

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)

class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

user = User("Alice", 25)
print(json.dumps(user, cls=CustomEncoder))
# {"name": "Alice", "age": 25}
```

---

## 自定义反序列化

### object_hook 参数

```python
import json
from datetime import datetime

def json_decoder(dct):
    if "created" in dct:
        dct["created"] = datetime.fromisoformat(dct["created"])
    return dct

json_str = '{"name": "Alice", "created": "2024-01-15T14:30:45"}'
data = json.loads(json_str, object_hook=json_decoder)
print(type(data["created"]))  # <class 'datetime.datetime'>
```

---

## 类型对照表

| Python | JSON |
|--------|------|
| `dict` | object |
| `list`, `tuple` | array |
| `str` | string |
| `int`, `float` | number |
| `True` | true |
| `False` | false |
| `None` | null |

---

## 常见问题

### 处理 NaN 和 Infinity

```python
import json

data = {"value": float("nan")}

# 默认不符合 JSON 标准
print(json.dumps(data))  # {"value": NaN}

# 禁用（抛出异常）
try:
    json.dumps(data, allow_nan=False)
except ValueError as e:
    print(e)  # Out of range float values are not JSON compliant
```

### 处理 bytes

```python
import json
import base64

data = {"content": b"binary data"}

def encode_bytes(obj):
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    raise TypeError()

print(json.dumps(data, default=encode_bytes))
# {"content": "YmluYXJ5IGRhdGE="}
```

---

## Python json vs JavaScript JSON

| 功能 | Python | JavaScript |
|------|--------|------------|
| 序列化 | `json.dumps()` | `JSON.stringify()` |
| 反序列化 | `json.loads()` | `JSON.parse()` |
| 文件写入 | `json.dump(data, f)` | `fs.writeFileSync(f, JSON.stringify(data))` |
| 文件读取 | `json.load(f)` | `JSON.parse(fs.readFileSync(f))` |
| 自定义编码 | `default=` / `cls=` | `replacer` 参数 |
| 格式化 | `indent=` | `space` 参数 |

---

## 实际应用

### 配置文件

```python
import json
from pathlib import Path

def load_config(path):
    config_path = Path(path)
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}

def save_config(path, config):
    Path(path).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# 使用
config = load_config("config.json")
config["debug"] = True
save_config("config.json", config)
```

---

## 本节要点

1. `dumps`/`loads` 处理字符串
2. `dump`/`load` 处理文件
3. `indent` 格式化，`ensure_ascii=False` 中文
4. `default` 或 `JSONEncoder` 自定义序列化
5. `object_hook` 自定义反序列化


