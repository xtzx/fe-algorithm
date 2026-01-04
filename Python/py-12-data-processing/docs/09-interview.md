# 08. 面试题

## 1. pydantic 和 dataclass 的区别？

**答案**：

| 特性 | pydantic | dataclass |
|------|----------|-----------|
| 运行时验证 | ✓ 自动验证 | ✗ 无验证 |
| 类型强制转换 | ✓ "30" → 30 | ✗ 保持原类型 |
| 序列化 | 内置 model_dump | 需要 asdict |
| JSON Schema | 自动生成 | 无 |
| 依赖 | 第三方库 | 标准库 |
| 性能 | v2 很快 | 更快 |

**使用场景**：
- 外部数据验证 → pydantic
- 内部数据结构 → dataclass

```python
# pydantic
user = User(age="30")  # 自动转换为 int
print(user.age)  # 30

# dataclass
user = User(age="30")  # 保持 str
print(user.age)  # "30"
```

---

## 2. 如何处理 pydantic 验证失败？

**答案**：

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

try:
    user = User(name="Alice", age="invalid")
except ValidationError as e:
    # 获取错误列表
    errors = e.errors()
    for error in errors:
        print(f"Field: {error['loc']}")
        print(f"Error: {error['msg']}")
        print(f"Type: {error['type']}")

    # JSON 格式
    print(e.json())
```

**最佳实践**：
- 在 API 边界捕获
- 返回友好的错误消息
- 记录详细日志

---

## 3. JSONL 的优势是什么？

**答案**：

**JSONL (JSON Lines)**：每行一个 JSON 对象。

| 特性 | JSON | JSONL |
|------|------|-------|
| 内存 | 全部加载 | 逐行处理 |
| 追加 | 困难 | 简单 |
| 流式 | 不支持 | 支持 |
| 大文件 | 容易 OOM | 可处理 |
| 错误恢复 | 整体失败 | 单行失败 |

**适用场景**：
- 日志文件
- 大数据集
- 流式处理
- 数据管道

```python
# 逐行处理大文件
with open("huge.jsonl") as f:
    for line in f:
        item = json.loads(line)
        process(item)
```

---

## 4. 如何处理大型 JSON 文件？

**答案**：

### 方法 1：转为 JSONL

```python
# 先转换格式
with open("large.json") as f:
    data = json.load(f)  # 一次性加载
with open("large.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

### 方法 2：流式解析 (ijson)

```python
import ijson

with open("large.json", "rb") as f:
    for item in ijson.items(f, "items.item"):
        process(item)
```

### 方法 3：分块处理

```python
# 如果是数组，可以分块
import json

def process_large_json(path, chunk_size=1000):
    with open(path) as f:
        # 假设是 JSON 数组
        data = json.load(f)
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            process_batch(chunk)
```

---

## 5. pydantic v1 和 v2 的区别？

**答案**：

| 特性 | v1 | v2 |
|------|----|----|
| 性能 | 慢 | 5-50x 更快 |
| 核心 | Python | Rust (pydantic-core) |
| 验证器 | @validator | @field_validator |
| 序列化 | .dict() | .model_dump() |
| JSON | .json() | .model_dump_json() |
| 配置 | class Config | model_config = ConfigDict() |
| 解析 | .parse_obj() | .model_validate() |

**v2 迁移**：
```python
# v1
class User(BaseModel):
    class Config:
        extra = "forbid"

    @validator("name")
    def validate_name(cls, v):
        return v.strip()

# v2
class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()
```

---

## 6. 如何自定义 pydantic 验证器？

**答案**：

### field_validator（单字段）

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()
```

### model_validator（跨字段）

```python
from pydantic import model_validator

class DateRange(BaseModel):
    start: str
    end: str

    @model_validator(mode="after")
    def validate_range(self):
        if self.start > self.end:
            raise ValueError("start must be before end")
        return self
```

---

## 7. Optional 和 Union 在 pydantic 中的区别？

**答案**：

```python
from typing import Optional, Union

class User(BaseModel):
    # Optional[str] = str | None
    nickname: str | None = None  # 可以是 str 或 None，默认 None

    # Union[str, int] = str | int
    id: str | int  # 可以是 str 或 int，必填

# Optional 等价于
nickname: Union[str, None] = None

# 注意：Optional 只是 Union[X, None] 的语法糖
```

**v2 推荐**：
```python
# 使用 Python 3.10+ 语法
name: str | None = None
id: str | int
```

---

## 8. 如何处理日期时间字段？

**答案**：

### 自动解析

```python
from datetime import date, datetime
from pydantic import BaseModel

class Event(BaseModel):
    date: date
    created_at: datetime

# 自动解析 ISO 格式
event = Event(
    date="2024-01-15",
    created_at="2024-01-15T10:30:00Z"
)
```

### 自定义格式

```python
from pydantic import field_validator

class Event(BaseModel):
    date_str: str

    @field_validator("date_str")
    @classmethod
    def parse_date(cls, v: str) -> str:
        from datetime import datetime
        # 支持多种格式
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError(f"Invalid date format: {v}")
```

### 序列化

```python
from pydantic import BaseModel
from datetime import datetime

class Event(BaseModel):
    created_at: datetime

event = Event(created_at=datetime.now())
print(event.model_dump_json())
# {"created_at": "2024-01-15T10:30:00"}
```

