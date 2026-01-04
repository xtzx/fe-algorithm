# 03. pydantic 核心

## 本节目标

- 掌握 pydantic BaseModel
- 理解字段类型和验证
- 序列化与反序列化

---

## pydantic vs dataclass

| 特性 | pydantic | dataclass |
|------|----------|-----------|
| 运行时验证 | ✓ | ✗ |
| 类型强制转换 | ✓ | ✗ |
| 序列化 | 内置 | 需要额外代码 |
| JSON Schema | 自动生成 | ✗ |
| 性能 | v2 很快 | 更快 |
| 依赖 | 第三方库 | 标准库 |

**选择建议**：
- 需要验证外部输入 → pydantic
- 内部数据结构 → dataclass

---

## 基本模型

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int
    is_active: bool = True  # 默认值

# 创建实例
user = User(name="Alice", email="alice@example.com", age=30)

# 访问属性
print(user.name)  # Alice
print(user.model_dump())  # {'name': 'Alice', ...}
```

---

## 类型强制转换

pydantic 自动转换兼容类型：

```python
class User(BaseModel):
    age: int

# 字符串自动转换为 int
user = User(age="30")
print(user.age)  # 30 (int)

# 不兼容类型会报错
User(age="abc")  # ValidationError
```

---

## 可选字段

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    nickname: str | None = None  # 可选
    age: int | None = None       # 可选

user = User(name="Alice")
print(user.nickname)  # None
```

### Python 3.10+ 类型语法

```python
# 新语法
name: str | None
items: list[str]

# 旧语法（也支持）
from typing import Optional, List
name: Optional[str]
items: List[str]
```

---

## 序列化

### model_dump() - 转为字典

```python
user = User(name="Alice", email="alice@example.com", age=30)

# 转为字典
data = user.model_dump()
# {'name': 'Alice', 'email': 'alice@example.com', 'age': 30}

# 排除字段
data = user.model_dump(exclude={"age"})

# 只包含字段
data = user.model_dump(include={"name", "email"})

# 排除 None 值
data = user.model_dump(exclude_none=True)
```

### model_dump_json() - 转为 JSON

```python
json_str = user.model_dump_json()
# '{"name":"Alice","email":"alice@example.com","age":30}'

# 格式化
json_str = user.model_dump_json(indent=2)
```

---

## 反序列化

### 从字典创建

```python
data = {"name": "Alice", "email": "alice@example.com", "age": 30}
user = User(**data)
# 或
user = User.model_validate(data)
```

### 从 JSON 创建

```python
json_str = '{"name": "Alice", "email": "alice@example.com", "age": 30}'
user = User.model_validate_json(json_str)
```

---

## 嵌套模型

```python
from pydantic import BaseModel

class Address(BaseModel):
    city: str
    street: str
    zip_code: str

class User(BaseModel):
    name: str
    address: Address  # 嵌套模型
    tags: list[str] = []

# 使用
data = {
    "name": "Alice",
    "address": {
        "city": "Beijing",
        "street": "Main St",
        "zip_code": "100000"
    },
    "tags": ["python", "data"]
}

user = User(**data)
print(user.address.city)  # Beijing
```

---

## 列表和字典

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    tags: list[str] = []
    metadata: dict[str, str] = {}
    scores: list[int] = []

user = User(
    name="Alice",
    tags=["python", "data"],
    metadata={"role": "admin"},
    scores=[90, 85, 95]
)
```

---

## 模型配置

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,  # 自动去除空白
        str_min_length=1,           # 字符串最小长度
        extra="forbid",             # 禁止额外字段
        frozen=True,                # 不可变
    )

    name: str
    email: str

# 额外字段报错
User(name="Alice", email="a@b.com", extra="field")  # ValidationError
```

### 常用配置

| 配置 | 说明 |
|------|------|
| `extra="forbid"` | 禁止额外字段 |
| `extra="ignore"` | 忽略额外字段 |
| `frozen=True` | 不可变 |
| `str_strip_whitespace=True` | 去除空白 |
| `validate_default=True` | 验证默认值 |

---

## JSON Schema

```python
class User(BaseModel):
    name: str
    age: int

# 生成 JSON Schema
schema = User.model_json_schema()
print(schema)
# {
#   'properties': {
#     'name': {'title': 'Name', 'type': 'string'},
#     'age': {'title': 'Age', 'type': 'integer'}
#   },
#   'required': ['name', 'age'],
#   'title': 'User',
#   'type': 'object'
# }
```

---

## 与 JS/Zod 对比

**Python (pydantic)**:
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int

user = User(name="Alice", email="a@b.com", age=30)
```

**JavaScript (zod)**:
```javascript
import { z } from 'zod';

const User = z.object({
  name: z.string(),
  email: z.string().email(),
  age: z.number().int(),
});

const user = User.parse({ name: "Alice", email: "a@b.com", age: 30 });
```

---

## 本节要点

1. **BaseModel** 是 pydantic 的核心
2. **自动类型转换** 字符串 → int 等
3. **model_dump()** 序列化为字典
4. **model_validate()** 从字典创建
5. **嵌套模型** 支持复杂结构
6. **ConfigDict** 配置模型行为

