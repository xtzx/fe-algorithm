# P12: 数据处理与模型化

> 掌握数据清洗、验证、序列化（为 AI/爬虫输入服务）

## 学完后能做

- 使用 pydantic 进行数据验证
- 处理 JSON/CSV/JSONL
- 数据清洗与转换

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行数据清洗管道
python -m data_lab clean data/dirty.csv -o data/clean.jsonl

# 生成数据报告
python -m data_lab report data/dirty.csv
```

## Python vs JavaScript 对比

| 概念 | Python | JavaScript |
|------|--------|------------|
| 数据验证 | pydantic | zod / yup |
| JSON 处理 | json 模块 | JSON.parse/stringify |
| CSV 处理 | csv 模块 | papaparse |
| 类型定义 | pydantic Model | TypeScript interface |

## 目录结构

```
py-12-data-processing/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-json.md           # JSON 处理
│   ├── 02-yaml-toml.md      # YAML/TOML 配置
│   ├── 03-csv.md            # CSV 处理
│   ├── 04-pydantic.md       # pydantic 核心
│   ├── 05-validation.md     # 验证与约束
│   ├── 06-cleaning.md       # 数据清洗
│   ├── 07-transformation.md # 数据转换
│   ├── 08-exercises.md      # 练习题
│   └── 09-interview.md      # 面试题
├── src/data_lab/
│   ├── __init__.py
│   ├── models.py            # pydantic 模型
│   ├── parsers.py           # 解析器
│   ├── cleaners.py          # 清洗器
│   ├── validators.py        # 验证器
│   ├── reporters.py         # 报告生成
│   └── cli.py               # CLI 工具
├── tests/
├── data/
│   ├── dirty.csv            # 脏数据示例
│   └── schema.json          # 数据 schema
└── scripts/
```

## 核心概念

### pydantic 模型

```python
from pydantic import BaseModel, Field, field_validator
from datetime import date

class User(BaseModel):
    name: str = Field(..., min_length=1)
    email: str
    age: int = Field(ge=0, le=150)
    birthday: date | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

# 使用
user = User(name="Alice", email="Alice@Example.com", age=30)
print(user.model_dump_json())
```

### 数据格式处理

```python
# JSON
import json
data = json.loads('{"name": "Alice"}')

# CSV
import csv
with open("data.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

# JSONL（流式）
with open("data.jsonl") as f:
    for line in f:
        item = json.loads(line)
```

## 数据清洗管道

```
dirty.csv → 解析 → 清洗 → 验证 → clean.jsonl
                              ↓
                         report.json
```

## 学习路径

1. [JSON 处理](docs/01-json.md)
2. [CSV 处理](docs/02-csv.md)
3. [pydantic 核心](docs/03-pydantic.md)
4. [验证与约束](docs/04-validation.md)
5. [数据清洗](docs/05-cleaning.md)
6. [数据转换](docs/06-transformation.md)

