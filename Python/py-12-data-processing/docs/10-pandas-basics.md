# Pandas 基础入门

> 数据分析的瑞士军刀

## 什么是 Pandas

Pandas 是 Python 最流行的数据分析库，提供了高效的数据结构和分析工具。

```bash
pip install pandas
```

---

## 核心数据结构

### Series（一维）

```python
import pandas as pd

# 从列表创建
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# 自定义索引
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s['a'])  # 1

# 从字典创建
s = pd.Series({'name': 'Alice', 'age': 30, 'city': 'Beijing'})
```

### DataFrame（二维）

```python
import pandas as pd

# 从字典创建
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Beijing', 'Shanghai', 'Guangzhou']
})

print(df)
#       name  age       city
# 0    Alice   25    Beijing
# 1      Bob   30   Shanghai
# 2  Charlie   35  Guangzhou

# 从列表嵌套创建
df = pd.DataFrame(
    [['Alice', 25], ['Bob', 30]],
    columns=['name', 'age']
)
```

---

## 数据读取

### CSV 文件

```python
# 读取
df = pd.read_csv('data.csv')

# 常用参数
df = pd.read_csv(
    'data.csv',
    sep=',',              # 分隔符
    encoding='utf-8',     # 编码
    header=0,             # 表头行号，None 表示无表头
    names=['col1', 'col2'],  # 自定义列名
    index_col='id',       # 索引列
    usecols=['name', 'age'],  # 只读取指定列
    dtype={'age': int},   # 指定列类型
    na_values=['NA', ''],  # 空值标识
    nrows=1000,           # 只读取前 N 行
)

# 写入
df.to_csv('output.csv', index=False)
```

### JSON 文件

```python
# 读取
df = pd.read_json('data.json')

# 常用参数
df = pd.read_json(
    'data.json',
    orient='records',  # JSON 结构：records, columns, index, values
    encoding='utf-8',
)

# 写入
df.to_json('output.json', orient='records', indent=2)
```

### Excel 文件

```bash
pip install openpyxl  # xlsx 支持
```

```python
# 读取
df = pd.read_excel('data.xlsx')

# 读取指定工作表
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 读取所有工作表
dfs = pd.read_excel('data.xlsx', sheet_name=None)  # 返回 dict

# 写入
df.to_excel('output.xlsx', index=False, sheet_name='Result')

# 写入多个工作表
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

### 数据库

```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///database.db')

# 读取
df = pd.read_sql('SELECT * FROM users', engine)
df = pd.read_sql_table('users', engine)
df = pd.read_sql_query('SELECT * FROM users WHERE age > 20', engine)

# 写入
df.to_sql('users', engine, if_exists='replace', index=False)
```

---

## 数据查看

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [5000, 6000, 7000, 5500, 6500]
})

# 基本信息
print(df.shape)      # (5, 3) - 行数, 列数
print(df.columns)    # Index(['name', 'age', 'salary'], dtype='object')
print(df.dtypes)     # 各列数据类型
print(df.info())     # 详细信息

# 查看数据
print(df.head(3))    # 前 3 行
print(df.tail(2))    # 后 2 行
print(df.sample(2))  # 随机 2 行

# 统计摘要
print(df.describe())  # 数值列统计
#              age       salary
# count   5.000000     5.000000
# mean   30.000000  6000.000000
# std     3.807887   790.569415
# min    25.000000  5000.000000
# 25%    28.000000  5500.000000
# 50%    30.000000  6000.000000
# 75%    32.000000  6500.000000
# max    35.000000  7000.000000
```

---

## 数据选择

### 列选择

```python
# 单列（返回 Series）
print(df['name'])

# 多列（返回 DataFrame）
print(df[['name', 'age']])

# 属性访问（仅限列名是有效标识符）
print(df.name)
```

### 行选择

```python
# 按位置（iloc）
print(df.iloc[0])       # 第一行
print(df.iloc[0:3])     # 前三行
print(df.iloc[[0, 2, 4]])  # 第 0、2、4 行

# 按标签（loc）
df.index = ['a', 'b', 'c', 'd', 'e']
print(df.loc['a'])      # 索引为 'a' 的行
print(df.loc['a':'c'])  # 'a' 到 'c'（包含两端）
```

### 行列同时选择

```python
# iloc[行, 列]
print(df.iloc[0, 1])        # 第一行第二列
print(df.iloc[0:3, 0:2])    # 前三行，前两列

# loc[行, 列]
print(df.loc['a', 'name'])
print(df.loc['a':'c', ['name', 'age']])
```

### 条件筛选

```python
# 布尔索引
print(df[df['age'] > 30])

# 多条件（使用 & | ~）
print(df[(df['age'] > 25) & (df['salary'] > 5500)])

# isin
print(df[df['name'].isin(['Alice', 'Bob'])])

# query 方法
print(df.query('age > 25 and salary > 5500'))
```

---

## 数据修改

### 添加列

```python
# 直接赋值
df['bonus'] = df['salary'] * 0.1

# 使用 assign（返回新 DataFrame）
df = df.assign(
    bonus=df['salary'] * 0.1,
    total=lambda x: x['salary'] + x['bonus']
)

# 插入列到指定位置
df.insert(1, 'id', [1, 2, 3, 4, 5])
```

### 修改值

```python
# 按位置修改
df.iloc[0, 1] = 26

# 按标签修改
df.loc['a', 'age'] = 26

# 条件修改
df.loc[df['age'] > 30, 'salary'] = 8000

# replace
df['city'] = df['city'].replace('Beijing', '北京')
```

### 删除

```python
# 删除列
df = df.drop('bonus', axis=1)
df = df.drop(['bonus', 'total'], axis=1)

# 删除行
df = df.drop(0)  # 删除索引为 0 的行
df = df.drop(['a', 'b'])  # 删除索引为 'a', 'b' 的行

# 原地删除
df.drop('bonus', axis=1, inplace=True)
```

---

## 缺失值处理

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, 2, 3, np.nan]
})

# 检查缺失值
print(df.isna())       # 返回布尔 DataFrame
print(df.isna().sum())  # 每列缺失值数量
print(df.isna().any())  # 每列是否有缺失值

# 删除缺失值
df.dropna()            # 删除任何包含 NaN 的行
df.dropna(axis=1)      # 删除任何包含 NaN 的列
df.dropna(how='all')   # 只删除全为 NaN 的行
df.dropna(subset=['A', 'B'])  # 只检查指定列

# 填充缺失值
df.fillna(0)           # 用 0 填充
df.fillna(df.mean())   # 用均值填充
df.fillna(method='ffill')  # 前向填充
df.fillna(method='bfill')  # 后向填充
df['A'].fillna(df['A'].median())  # 用中位数填充
```

---

## 数据类型转换

```python
# 查看类型
print(df.dtypes)

# 转换类型
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)
df['name'] = df['name'].astype('string')

# 转换为日期
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# 转换为分类
df['category'] = df['category'].astype('category')
```

---

## 字符串操作

```python
# str 访问器
df['name'].str.lower()          # 转小写
df['name'].str.upper()          # 转大写
df['name'].str.strip()          # 去除空格
df['name'].str.contains('li')   # 包含判断
df['name'].str.startswith('A')  # 开头判断
df['name'].str.replace('a', 'A')  # 替换
df['name'].str.split(' ')       # 分割
df['name'].str.len()            # 长度

# 正则表达式
df['phone'].str.extract(r'(\d{3})-(\d{4})')  # 提取
df['text'].str.findall(r'\d+')  # 查找所有匹配
```

---

## 与 Pydantic 配合

### DataFrame 转 Pydantic 模型

```python
from pydantic import BaseModel
from typing import List
import pandas as pd

class User(BaseModel):
    name: str
    age: int
    email: str

# DataFrame 转模型列表
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'email': ['alice@example.com', 'bob@example.com']
})

users: List[User] = [User(**row) for row in df.to_dict('records')]
```

### Pydantic 模型转 DataFrame

```python
users = [
    User(name='Alice', age=25, email='alice@example.com'),
    User(name='Bob', age=30, email='bob@example.com'),
]

df = pd.DataFrame([u.model_dump() for u in users])
```

### 数据验证

```python
from pydantic import BaseModel, validator
import pandas as pd

class CleanedUser(BaseModel):
    name: str
    age: int
    email: str

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty')
        return v.strip()

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('age must be positive')
        return v

def validate_dataframe(df: pd.DataFrame) -> List[CleanedUser]:
    """验证并清洗 DataFrame"""
    valid_rows = []
    errors = []

    for idx, row in df.iterrows():
        try:
            user = CleanedUser(**row.to_dict())
            valid_rows.append(user)
        except Exception as e:
            errors.append((idx, str(e)))

    if errors:
        print(f"Validation errors: {errors}")

    return valid_rows
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 链式赋值警告 | `df[df['a'] > 0]['b'] = 1` | 使用 `.loc`: `df.loc[df['a'] > 0, 'b'] = 1` |
| 修改副本 | 切片可能返回视图或副本 | 使用 `.copy()` 明确复制 |
| inplace 陷阱 | `inplace=True` 返回 None | 避免使用 inplace，直接赋值 |
| 大文件内存 | 一次加载整个文件 | 使用 `chunksize` 分块读取 |

---

## 小结

1. **Series**：一维数据，带索引
2. **DataFrame**：二维表格，核心数据结构
3. **数据读取**：支持 CSV、JSON、Excel、数据库
4. **数据选择**：`iloc` 按位置，`loc` 按标签
5. **缺失值**：`dropna()` 删除，`fillna()` 填充
6. **与 Pydantic**：`to_dict('records')` + 模型验证

