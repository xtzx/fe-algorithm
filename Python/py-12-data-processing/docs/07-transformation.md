# 06. 数据转换

## 本节目标

- 掌握字段映射
- 数据聚合与统计
- 格式转换

---

## 字段映射

### 简单字段重命名

```python
def rename_fields(data: dict, mapping: dict[str, str]) -> dict:
    """重命名字段"""
    result = {}
    for key, value in data.items():
        new_key = mapping.get(key, key)
        result[new_key] = value
    return result

# 使用
data = {"firstName": "Alice", "lastName": "Smith"}
mapping = {"firstName": "first_name", "lastName": "last_name"}
result = rename_fields(data, mapping)
# {"first_name": "Alice", "last_name": "Smith"}
```

### 复杂字段转换

```python
from typing import Any, Callable

def transform_fields(
    data: dict,
    transformations: dict[str, Callable[[dict], Any]]
) -> dict:
    """转换字段"""
    result = data.copy()
    for field, transform in transformations.items():
        result[field] = transform(data)
    return result

# 使用
data = {"first_name": "Alice", "last_name": "Smith", "age": 30}
transformations = {
    "full_name": lambda d: f"{d['first_name']} {d['last_name']}",
    "birth_year": lambda d: 2024 - d["age"],
}
result = transform_fields(data, transformations)
# {"first_name": "Alice", ..., "full_name": "Alice Smith", "birth_year": 1994}
```

---

## 嵌套结构展开

```python
def flatten_dict(d: dict, separator: str = ".") -> dict:
    """展开嵌套字典"""
    result = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        else:
            result[prefix] = obj

    _flatten(d)
    return result

# 使用
data = {
    "user": {
        "name": "Alice",
        "address": {
            "city": "Beijing"
        }
    }
}
flat = flatten_dict(data)
# {"user.name": "Alice", "user.address.city": "Beijing"}
```

### 重建嵌套结构

```python
def unflatten_dict(d: dict, separator: str = ".") -> dict:
    """重建嵌套字典"""
    result = {}
    for key, value in d.items():
        parts = key.split(separator)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result
```

---

## 数据聚合

### 按字段分组

```python
from collections import defaultdict

def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    """按字段分组"""
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(groups)

# 使用
data = [
    {"city": "Beijing", "name": "Alice"},
    {"city": "Shanghai", "name": "Bob"},
    {"city": "Beijing", "name": "Charlie"},
]
grouped = group_by(data, "city")
# {"Beijing": [...], "Shanghai": [...]}
```

### 聚合统计

```python
def aggregate(
    items: list[dict],
    group_key: str,
    agg_field: str,
    agg_func: str = "sum"
) -> dict[str, float]:
    """聚合统计"""
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for item in items:
        groups[item[group_key]].append(item[agg_field])

    result = {}
    for key, values in groups.items():
        if agg_func == "sum":
            result[key] = sum(values)
        elif agg_func == "avg":
            result[key] = statistics.mean(values)
        elif agg_func == "count":
            result[key] = len(values)
        elif agg_func == "min":
            result[key] = min(values)
        elif agg_func == "max":
            result[key] = max(values)

    return result

# 使用
data = [
    {"city": "Beijing", "sales": 100},
    {"city": "Shanghai", "sales": 200},
    {"city": "Beijing", "sales": 150},
]
total_sales = aggregate(data, "city", "sales", "sum")
# {"Beijing": 250, "Shanghai": 200}
```

---

## 基础统计

```python
import statistics
from typing import Any

def basic_stats(values: list[float]) -> dict[str, float]:
    """计算基础统计"""
    if not values:
        return {"count": 0}

    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
    }

def field_stats(items: list[dict], field: str) -> dict[str, float]:
    """计算字段统计"""
    values = [item[field] for item in items if field in item and item[field] is not None]
    numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
    return basic_stats(numeric_values)
```

---

## 数据质量报告

```python
from dataclasses import dataclass, field

@dataclass
class FieldReport:
    """字段质量报告"""
    name: str
    total: int
    non_null: int
    null_count: int
    null_rate: float
    unique_count: int
    sample_values: list

@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_records: int
    fields: list[FieldReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "fields": [
                {
                    "name": f.name,
                    "total": f.total,
                    "non_null": f.non_null,
                    "null_count": f.null_count,
                    "null_rate": f.null_rate,
                    "unique_count": f.unique_count,
                    "sample_values": f.sample_values,
                }
                for f in self.fields
            ]
        }

def generate_report(items: list[dict]) -> DataQualityReport:
    """生成数据质量报告"""
    if not items:
        return DataQualityReport(total_records=0)

    total = len(items)
    all_fields = set()
    for item in items:
        all_fields.update(item.keys())

    fields = []
    for field_name in sorted(all_fields):
        values = [item.get(field_name) for item in items]
        non_null_values = [v for v in values if v is not None and v != ""]
        unique_values = set(str(v) for v in non_null_values)

        fields.append(FieldReport(
            name=field_name,
            total=total,
            non_null=len(non_null_values),
            null_count=total - len(non_null_values),
            null_rate=(total - len(non_null_values)) / total,
            unique_count=len(unique_values),
            sample_values=list(unique_values)[:5],
        ))

    return DataQualityReport(total_records=total, fields=fields)
```

---

## 格式转换管道

```python
from typing import Iterator
import json
import csv

class DataPipeline:
    """数据转换管道"""

    def __init__(self):
        self.transformers = []

    def add(self, transformer):
        self.transformers.append(transformer)
        return self

    def process(self, items: Iterator[dict]) -> Iterator[dict]:
        for item in items:
            result = item
            for transformer in self.transformers:
                result = transformer(result)
                if result is None:
                    break
            if result is not None:
                yield result

# 使用
pipeline = DataPipeline()
pipeline.add(lambda x: {**x, "name": x["name"].strip()})
pipeline.add(lambda x: {**x, "age": int(x["age"])})
pipeline.add(lambda x: x if x["age"] >= 18 else None)  # 过滤

for item in pipeline.process(raw_data):
    print(item)
```

---

## 本节要点

1. **字段映射** 重命名和转换
2. **嵌套展开** flatten/unflatten
3. **分组聚合** group_by + aggregate
4. **基础统计** count/sum/mean/etc
5. **质量报告** 空值率、唯一值
6. **管道模式** 链式转换

