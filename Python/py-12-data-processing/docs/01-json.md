# 01. JSON 处理

## 本节目标

- 掌握 JSON 读写
- 处理嵌套和大文件
- 了解 JSONL 格式

---

## json 模块基础

### 解析 JSON

```python
import json

# 从字符串解析
json_str = '{"name": "Alice", "age": 30}'
data = json.loads(json_str)
print(data["name"])  # Alice

# 从文件解析
with open("data.json", encoding="utf-8") as f:
    data = json.load(f)
```

### 生成 JSON

```python
import json

data = {"name": "Alice", "age": 30, "tags": ["python", "data"]}

# 转为字符串
json_str = json.dumps(data)
print(json_str)  # {"name": "Alice", "age": 30, "tags": ["python", "data"]}

# 格式化输出
json_str = json.dumps(data, indent=2)

# 写入文件
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

---

## 常用参数

| 参数 | 说明 |
|------|------|
| `indent` | 缩进空格数 |
| `ensure_ascii` | False 允许非 ASCII 字符 |
| `sort_keys` | 按键排序 |
| `default` | 自定义序列化函数 |

```python
# 中文处理
data = {"名字": "张三"}
json.dumps(data, ensure_ascii=False)  # {"名字": "张三"}

# 排序键
json.dumps(data, sort_keys=True)
```

---

## 自定义序列化

```python
import json
from datetime import datetime, date
from decimal import Decimal

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

data = {
    "time": datetime.now(),
    "price": Decimal("19.99"),
}

json.dumps(data, cls=CustomEncoder)
```

### 使用 default 参数

```python
def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json.dumps(data, default=serialize)
```

---

## 处理嵌套结构

```python
data = {
    "user": {
        "name": "Alice",
        "address": {
            "city": "Beijing",
            "street": "Main St"
        }
    },
    "orders": [
        {"id": 1, "total": 100},
        {"id": 2, "total": 200},
    ]
}

# 访问嵌套数据
city = data["user"]["address"]["city"]

# 安全访问（避免 KeyError）
city = data.get("user", {}).get("address", {}).get("city", "Unknown")
```

---

## JSONL (JSON Lines)

每行一个 JSON 对象，适合大文件和流式处理：

```
{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Charlie"}
```

### 读取 JSONL

```python
import json

def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# 使用
for item in read_jsonl("data.jsonl"):
    print(item)
```

### 写入 JSONL

```python
def write_jsonl(path: str, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 使用
data = [{"id": 1}, {"id": 2}, {"id": 3}]
write_jsonl("output.jsonl", data)
```

---

## JSONL vs JSON

| 特性 | JSON | JSONL |
|------|------|-------|
| 结构 | 单个对象/数组 | 每行一个对象 |
| 内存 | 需要全部加载 | 可逐行处理 |
| 追加 | 困难 | 简单 |
| 流式 | 不支持 | 支持 |
| 适用 | 小文件 | 大文件、日志 |

---

## 处理大文件

### 问题：内存不足

```python
# ❌ 全部加载到内存
with open("huge.json") as f:
    data = json.load(f)  # 可能 OOM
```

### 解决方案

**1. 使用 JSONL**

```python
# ✓ 逐行处理
for item in read_jsonl("huge.jsonl"):
    process(item)
```

**2. 使用 ijson（流式解析）**

```python
import ijson

with open("huge.json", "rb") as f:
    for item in ijson.items(f, "items.item"):
        process(item)
```

---

## 与 JS 对比

| 操作 | Python | JavaScript |
|------|--------|------------|
| 解析 | `json.loads(s)` | `JSON.parse(s)` |
| 序列化 | `json.dumps(obj)` | `JSON.stringify(obj)` |
| 格式化 | `json.dumps(obj, indent=2)` | `JSON.stringify(obj, null, 2)` |

---

## 本节要点

1. **json.loads/dumps** 处理字符串
2. **json.load/dump** 处理文件
3. **ensure_ascii=False** 支持中文
4. **JSONL** 适合大文件和流式处理
5. **自定义 Encoder** 处理特殊类型

