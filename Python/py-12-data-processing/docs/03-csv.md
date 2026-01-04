# 02. CSV 处理

## 本节目标

- 掌握 csv 模块
- 使用 DictReader/DictWriter
- 处理编码和特殊字符

---

## csv 模块基础

### 读取 CSV

```python
import csv

# 基本读取
with open("data.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)  # ['col1', 'col2', 'col3']
```

### 写入 CSV

```python
import csv

data = [
    ["name", "age", "city"],
    ["Alice", 30, "Beijing"],
    ["Bob", 25, "Shanghai"],
]

with open("output.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)
```

**注意**: `newline=""` 防止 Windows 下多余空行。

---

## DictReader / DictWriter

### DictReader - 字典读取

```python
import csv

# data.csv:
# name,age,city
# Alice,30,Beijing
# Bob,25,Shanghai

with open("data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
        # {'name': 'Alice', 'age': '30', 'city': 'Beijing'}
```

### DictWriter - 字典写入

```python
import csv

data = [
    {"name": "Alice", "age": 30, "city": "Beijing"},
    {"name": "Bob", "age": 25, "city": "Shanghai"},
]

with open("output.csv", "w", encoding="utf-8", newline="") as f:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()  # 写入表头
    writer.writerows(data)
```

---

## 处理特殊情况

### 自定义分隔符

```python
# TSV (Tab 分隔)
with open("data.tsv", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        print(row)

# 自定义分隔符
with open("data.txt", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
```

### 引号处理

```python
import csv

# 包含逗号的字段
data = [
    ["name", "description"],
    ["Alice", "Hello, World"],  # 会自动加引号
]

with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # 所有字段加引号
    writer.writerows(data)
```

| 引号模式 | 说明 |
|----------|------|
| QUOTE_MINIMAL | 只在必要时加引号（默认） |
| QUOTE_ALL | 所有字段加引号 |
| QUOTE_NONNUMERIC | 非数字字段加引号 |
| QUOTE_NONE | 不加引号 |

---

## 编码处理

### UTF-8 BOM

```python
# Excel 生成的 UTF-8 CSV 可能有 BOM
with open("data.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

### 处理 GBK 编码

```python
with open("data.csv", encoding="gbk") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

---

## 大文件处理

### 流式处理

```python
import csv

def process_large_csv(path: str):
    """逐行处理大文件"""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # 处理每行
            process(row)

            # 进度报告
            if i % 10000 == 0:
                print(f"Processed {i} rows")
```

### 分块处理

```python
import csv
from itertools import islice

def chunked_reader(path: str, chunk_size: int = 1000):
    """分块读取"""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        while True:
            chunk = list(islice(reader, chunk_size))
            if not chunk:
                break
            yield chunk

for chunk in chunked_reader("large.csv", 1000):
    process_batch(chunk)
```

---

## CSV 转其他格式

### CSV → JSON

```python
import csv
import json

def csv_to_json(csv_path: str, json_path: str):
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

### CSV → JSONL

```python
import csv
import json

def csv_to_jsonl(csv_path: str, jsonl_path: str):
    with open(csv_path, encoding="utf-8") as fin:
        with open(jsonl_path, "w", encoding="utf-8") as fout:
            reader = csv.DictReader(fin)
            for row in reader:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
```

---

## 数据验证

```python
import csv

def validate_csv(path: str, required_fields: list[str]) -> list[str]:
    """验证 CSV 结构"""
    errors = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 检查必需字段
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            errors.append(f"Missing fields: {missing}")

        # 检查数据
        for i, row in enumerate(reader, start=2):  # 从第2行开始
            for field in required_fields:
                if not row.get(field):
                    errors.append(f"Row {i}: empty {field}")

    return errors
```

---

## 本节要点

1. **csv.reader/writer** 处理列表
2. **DictReader/DictWriter** 处理字典
3. **newline=""** 防止 Windows 空行
4. **utf-8-sig** 处理 BOM
5. **流式处理** 大文件
6. **delimiter** 自定义分隔符

