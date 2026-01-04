# 结构化输出

## 概述

让 LLM 输出符合预定义 schema 的结构化数据。

## 1. 为什么需要结构化输出

```python
# 非结构化输出 - 难以解析
response = "John is 30 years old and works as a software engineer"

# 结构化输出 - 易于处理
response = {"name": "John", "age": 30, "occupation": "software engineer"}
```

## 2. JSON Schema 约束

### 2.1 定义 Schema

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """人物信息"""
    name: str = Field(..., description="姓名")
    age: int = Field(..., ge=0, le=150, description="年龄")
    occupation: str = Field(..., description="职业")
```

### 2.2 生成 JSON Schema

```python
schema = Person.model_json_schema()
# {
#   "type": "object",
#   "properties": {
#     "name": {"type": "string", "description": "姓名"},
#     "age": {"type": "integer", "minimum": 0, "maximum": 150},
#     "occupation": {"type": "string", "description": "职业"}
#   },
#   "required": ["name", "age", "occupation"]
# }
```

## 3. 使用结构化客户端

```python
from llm_kit.client import StructuredClient

client = StructuredClient(llm_client)

person = client.generate(
    prompt="Extract information: John is a 30-year-old software engineer",
    schema=Person,
)

print(person.name)  # John
print(person.age)   # 30
```

## 4. 自动重试验证

```python
# 如果 LLM 输出无效 JSON，自动重试
person = client.generate(
    prompt="...",
    schema=Person,
    max_retries=3,  # 验证失败重试次数
)
```

重试流程：
1. 发送请求
2. 解析 JSON
3. Pydantic 验证
4. 验证失败 → 添加错误反馈 → 重试

## 5. 生成列表

```python
class Product(BaseModel):
    name: str
    price: float
    category: str

products = client.generate_list(
    prompt="List 3 popular electronic products",
    item_schema=Product,
    max_items=3,
)

for p in products:
    print(f"{p.name}: ${p.price}")
```

## 6. 复杂嵌套结构

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    employees: int
    address: Address
    departments: List[str]

company = client.generate(
    prompt="Generate a tech company profile",
    schema=Company,
)
```

## 7. 函数调用解析

```python
from llm_kit.client import FunctionCallParser

class GetWeatherArgs(BaseModel):
    """获取天气"""
    city: str
    unit: str = "celsius"

parser = FunctionCallParser()
parser.register("get_weather", GetWeatherArgs)

# 发送请求
response = client.chat(
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=parser.get_tools_definition(),
)

# 解析工具调用
if response.tool_calls:
    for call in response.tool_calls:
        func_name, args = parser.parse(call)
        # args 是 GetWeatherArgs 实例
        print(f"Call {func_name} with city={args.city}")
```

## 8. 最佳实践

1. **使用描述性字段**：在 Field 中添加 description
2. **设置合理约束**：使用 ge、le、min_length 等
3. **温度设为 0**：确保输出稳定
4. **实现重试**：处理偶发的格式错误
5. **验证后使用**：始终通过 Pydantic 验证


