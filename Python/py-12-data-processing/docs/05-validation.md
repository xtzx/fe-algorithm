# 04. 验证与约束

## 本节目标

- 掌握 Field() 约束
- 自定义验证器
- 处理验证错误

---

## Field() 约束

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)  # 0 <= age <= 150
    email: str = Field(..., pattern=r"^[\w.]+@[\w.]+$")
    score: float = Field(default=0.0, ge=0, le=100)

# ... 表示必填字段
```

### 常用约束

| 参数 | 类型 | 说明 |
|------|------|------|
| `min_length` | str/list | 最小长度 |
| `max_length` | str/list | 最大长度 |
| `ge` | 数字 | >= |
| `gt` | 数字 | > |
| `le` | 数字 | <= |
| `lt` | 数字 | < |
| `pattern` | str | 正则表达式 |
| `default` | any | 默认值 |
| `default_factory` | callable | 默认值工厂 |

---

## Field() 元数据

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        title="产品名称",
        description="产品的显示名称",
        examples=["iPhone", "MacBook"],
    )
    price: float = Field(
        ...,
        gt=0,
        title="价格",
        description="产品价格（元）",
    )
```

---

## 自定义验证器

### field_validator (pydantic v2)

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()  # 转小写

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()  # 去除空白

user = User(name="  Alice  ", email="ALICE@Example.COM")
print(user.name)   # "Alice"
print(user.email)  # "alice@example.com"
```

### 验证多个字段

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    first_name: str
    last_name: str

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_names(cls, v: str) -> str:
        if not v.isalpha():
            raise ValueError("Name must contain only letters")
        return v.title()
```

---

## model_validator

用于跨字段验证：

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        return self

# 测试
DateRange(start_date="2024-01-01", end_date="2024-12-31")  # OK
DateRange(start_date="2024-12-31", end_date="2024-01-01")  # Error
```

### mode 选项

| mode | 说明 |
|------|------|
| `before` | 验证前（原始数据） |
| `after` | 验证后（模型实例） |

---

## 处理验证错误

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

try:
    user = User(name="Alice", age="invalid")
except ValidationError as e:
    print(e.errors())
    # [{'type': 'int_parsing', 'loc': ('age',), 'msg': '...'}]

    print(e.json())  # JSON 格式错误

    # 遍历错误
    for error in e.errors():
        print(f"Field: {error['loc']}, Error: {error['msg']}")
```

### 错误结构

```python
{
    "type": "int_parsing",       # 错误类型
    "loc": ("age",),             # 字段路径
    "msg": "Input should be...", # 错误消息
    "input": "invalid",          # 输入值
}
```

---

## 日期时间处理

```python
from datetime import date, datetime
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    date: date
    created_at: datetime

# 自动解析字符串
event = Event(
    name="Meeting",
    date="2024-01-15",
    created_at="2024-01-15T10:30:00"
)

print(event.date)        # date(2024, 1, 15)
print(event.created_at)  # datetime(2024, 1, 15, 10, 30)
```

### 自定义日期格式

```python
from datetime import datetime
from pydantic import BaseModel, field_validator

class Event(BaseModel):
    date_str: str

    @field_validator("date_str")
    @classmethod
    def parse_date(cls, v: str) -> str:
        # 验证格式
        datetime.strptime(v, "%Y/%m/%d")
        return v
```

---

## 枚举验证

```python
from enum import Enum
from pydantic import BaseModel

class Status(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"

class Task(BaseModel):
    name: str
    status: Status

task = Task(name="Review", status="pending")
print(task.status)  # Status.PENDING

# 无效值会报错
Task(name="Review", status="invalid")  # ValidationError
```

---

## Literal 类型

```python
from typing import Literal
from pydantic import BaseModel

class Config(BaseModel):
    mode: Literal["dev", "prod", "test"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]

config = Config(mode="dev", log_level="INFO")

# 无效值报错
Config(mode="invalid", log_level="INFO")  # ValidationError
```

---

## Union 类型

```python
from pydantic import BaseModel

class Response(BaseModel):
    data: str | int | list[str]  # 多种类型

Response(data="hello")  # OK
Response(data=123)      # OK
Response(data=["a"])    # OK
```

---

## 本节要点

1. **Field()** 定义约束和元数据
2. **@field_validator** 自定义字段验证
3. **@model_validator** 跨字段验证
4. **ValidationError** 处理验证错误
5. **Enum/Literal** 限制可选值
6. **自动日期解析** 字符串 → datetime

