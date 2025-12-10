# 🐼 05 - Pandas 数据处理

> Pandas 是数据分析的核心工具，让数据处理变得简单

---

## 目录

1. [Pandas 简介](#1-pandas-简介)
2. [数据结构](#2-数据结构)
3. [数据读写](#3-数据读写)
4. [数据选择](#4-数据选择)
5. [数据清洗](#5-数据清洗)
6. [数据转换](#6-数据转换)
7. [数据聚合](#7-数据聚合)
8. [练习题](#8-练习题)

---

## 1. Pandas 简介

### 1.1 安装和导入

```python
# 安装
# pip install pandas

# 导入
import pandas as pd
import numpy as np

print(pd.__version__)
```

### 1.2 核心数据结构

- **Series**：一维带标签数组
- **DataFrame**：二维带标签表格（最常用）

---

## 2. 数据结构

### 2.1 Series

```python
import pandas as pd
import numpy as np

# 从列表创建
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# 指定索引
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s)
# a    1
# b    2
# c    3

# 从字典创建
d = {'apple': 100, 'banana': 200, 'cherry': 150}
s = pd.Series(d)
print(s)

# 访问
print(s['apple'])     # 100
print(s[['apple', 'banana']])  # 多个
print(s[s > 100])     # 条件筛选

# 属性
print(s.index)   # Index(['apple', 'banana', 'cherry'], dtype='object')
print(s.values)  # [100 200 150]
print(s.dtype)   # int64
```

### 2.2 DataFrame

```python
# 从字典创建
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
}
df = pd.DataFrame(data)
print(df)
#       name  age     city
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 2  Charlie   35  Chicago

# 指定索引
df = pd.DataFrame(data, index=['a', 'b', 'c'])

# 从列表创建
data_list = [
    ['Alice', 25, 'NYC'],
    ['Bob', 30, 'LA'],
    ['Charlie', 35, 'Chicago']
]
df = pd.DataFrame(data_list, columns=['name', 'age', 'city'])

# 从 NumPy 数组创建
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# 基本属性
print(df.shape)      # (3, 3)
print(df.columns)    # Index(['name', 'age', 'city'], dtype='object')
print(df.index)      # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)     # 每列的数据类型
print(df.info())     # 详细信息
print(df.describe()) # 数值列的统计摘要
```

---

## 3. 数据读写

### 3.1 读取 CSV

```python
# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 常用参数
df = pd.read_csv(
    'data.csv',
    sep=',',             # 分隔符
    header=0,            # 标题行（None 表示没有）
    names=['col1', 'col2'],  # 指定列名
    index_col='id',      # 设置索引列
    usecols=['col1', 'col2'],  # 只读取指定列
    dtype={'age': int},  # 指定数据类型
    nrows=100,           # 只读前 100 行
    skiprows=1,          # 跳过前 1 行
    na_values=['NA', 'N/A', ''],  # 视为缺失值的值
    encoding='utf-8'     # 编码
)

# 读取在线 CSV
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)
```

### 3.2 写入 CSV

```python
# 写入 CSV
df.to_csv('output.csv', index=False)

# 常用参数
df.to_csv(
    'output.csv',
    index=False,         # 不写入索引
    columns=['col1', 'col2'],  # 只写入指定列
    header=True,         # 写入列名
    encoding='utf-8'
)
```

### 3.3 其他格式

```python
# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx', index=False)

# JSON
df = pd.read_json('data.json')
df.to_json('output.json')

# SQL（需要数据库连接）
# import sqlite3
# conn = sqlite3.connect('database.db')
# df = pd.read_sql('SELECT * FROM table_name', conn)
# df.to_sql('table_name', conn, if_exists='replace')
```

---

## 4. 数据选择

### 4.1 列选择

```python
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'Chicago', 'NYC'],
    'salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)

# 选择单列（返回 Series）
print(df['name'])
print(df.name)  # 等价写法

# 选择多列（返回 DataFrame）
print(df[['name', 'age']])
```

### 4.2 行选择

```python
# 按索引位置选择（iloc）
print(df.iloc[0])       # 第一行
print(df.iloc[0:2])     # 前两行
print(df.iloc[[0, 2]])  # 第1和第3行
print(df.iloc[0, 1])    # 第1行第2列

# 按标签选择（loc）
df.index = ['a', 'b', 'c', 'd']  # 设置标签索引
print(df.loc['a'])        # 标签为 'a' 的行
print(df.loc['a':'c'])    # 标签从 'a' 到 'c'（包含）
print(df.loc['a', 'name'])  # 指定行和列
```

### 4.3 条件选择

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'Chicago', 'NYC'],
    'salary': [50000, 60000, 70000, 55000]
})

# 单条件
print(df[df['age'] > 28])

# 多条件（& 表示 and，| 表示 or）
print(df[(df['age'] > 25) & (df['city'] == 'NYC')])

# isin() 方法
print(df[df['city'].isin(['NYC', 'LA'])])

# query() 方法（更易读）
print(df.query('age > 25 and city == "NYC"'))

# 字符串方法
print(df[df['name'].str.startswith('A')])
print(df[df['name'].str.contains('li')])
```

---

## 5. 数据清洗

### 5.1 处理缺失值

```python
# 创建带缺失值的数据
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': ['x', 'y', 'z', None]
})

# 检测缺失值
print(df.isnull())       # 每个元素是否为空
print(df.isnull().sum()) # 每列空值数量
print(df.isnull().sum().sum())  # 总空值数量

# 删除缺失值
df.dropna()              # 删除任何包含空值的行
df.dropna(axis=1)        # 删除任何包含空值的列
df.dropna(how='all')     # 只删除全为空的行
df.dropna(thresh=2)      # 至少有 2 个非空值的行

# 填充缺失值
df.fillna(0)             # 用 0 填充
df.fillna({'A': 0, 'B': 99})  # 不同列用不同值
df['A'].fillna(df['A'].mean())  # 用均值填充
df.fillna(method='ffill')  # 用前一个值填充
df.fillna(method='bfill')  # 用后一个值填充
```

### 5.2 处理重复值

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Alice', 'Charlie'],
    'age': [25, 30, 25, 35]
})

# 检测重复
print(df.duplicated())              # 标记重复行
print(df.duplicated().sum())        # 重复行数量
print(df[df.duplicated()])          # 查看重复行

# 删除重复
df.drop_duplicates()                # 删除重复行
df.drop_duplicates(subset=['name']) # 只考虑特定列
df.drop_duplicates(keep='last')     # 保留最后一个
```

### 5.3 数据类型转换

```python
df = pd.DataFrame({
    'A': ['1', '2', '3'],
    'B': ['1.1', '2.2', '3.3'],
    'C': ['2023-01-01', '2023-01-02', '2023-01-03']
})

# 转换数据类型
df['A'] = df['A'].astype(int)
df['B'] = df['B'].astype(float)
df['C'] = pd.to_datetime(df['C'])

# 类别类型（节省内存）
df['category'] = df['A'].astype('category')

print(df.dtypes)
```

### 5.4 字符串处理

```python
df = pd.DataFrame({
    'name': ['  Alice  ', 'bob', 'CHARLIE']
})

# 字符串方法（通过 .str 访问）
df['name_clean'] = df['name'].str.strip()      # 去除空格
df['name_lower'] = df['name'].str.lower()      # 小写
df['name_upper'] = df['name'].str.upper()      # 大写
df['name_title'] = df['name'].str.title()      # 首字母大写
df['name_len'] = df['name'].str.len()          # 长度
df['starts_a'] = df['name'].str.lower().str.startswith('a')

# 替换
df['name'] = df['name'].str.replace('Alice', 'ALICE')

# 分割
df['email'] = ['alice@gmail.com', 'bob@yahoo.com', 'charlie@outlook.com']
df['domain'] = df['email'].str.split('@').str[1]

print(df)
```

---

## 6. 数据转换

### 6.1 添加和修改列

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# 添加新列
df['bonus'] = df['salary'] * 0.1
df['age_group'] = ['young' if x < 30 else 'senior' for x in df['age']]

# 使用 apply
df['salary_k'] = df['salary'].apply(lambda x: x / 1000)

# 使用 map（映射）
city_map = {'Alice': 'NYC', 'Bob': 'LA', 'Charlie': 'Chicago'}
df['city'] = df['name'].map(city_map)

# 使用 assign（链式操作）
df = df.assign(
    total=df['salary'] + df['bonus'],
    tax=df['salary'] * 0.2
)

# 修改列名
df.rename(columns={'salary': 'annual_salary'}, inplace=True)

# 删除列
df.drop(columns=['bonus'], inplace=True)
```

### 6.2 apply 函数

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 对单列应用函数
df['A_squared'] = df['A'].apply(lambda x: x ** 2)

# 对多列应用函数
df['sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)

# 对整个 DataFrame 应用
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

df_normalized = df[['A', 'B']].apply(normalize)
print(df_normalized)
```

### 6.3 排序

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
})

# 按值排序
df.sort_values('age')                    # 升序
df.sort_values('age', ascending=False)   # 降序
df.sort_values(['city', 'age'])          # 多列排序

# 按索引排序
df.sort_index()

# 获取最大/最小的 N 个
df.nlargest(2, 'salary')  # 工资最高的 2 个
df.nsmallest(2, 'age')    # 年龄最小的 2 个
```

---

## 7. 数据聚合

### 7.1 基本聚合

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'department': ['IT', 'HR', 'IT', 'HR', 'IT'],
    'salary': [50000, 60000, 70000, 55000, 65000],
    'age': [25, 30, 35, 28, 32]
})

# 基本统计
print(df['salary'].sum())    # 总和
print(df['salary'].mean())   # 均值
print(df['salary'].median()) # 中位数
print(df['salary'].std())    # 标准差
print(df['salary'].min())    # 最小值
print(df['salary'].max())    # 最大值
print(df['salary'].count())  # 计数

# 多个统计
print(df['salary'].agg(['sum', 'mean', 'std']))

# 描述性统计
print(df.describe())
```

### 7.2 分组聚合（groupby）

```python
# 单列分组
grouped = df.groupby('department')

# 聚合
print(grouped['salary'].mean())
# department
# HR    57500.0
# IT    61666.666667

# 多种聚合
print(grouped['salary'].agg(['mean', 'sum', 'count']))

# 多列聚合
print(grouped.agg({
    'salary': 'mean',
    'age': 'max'
}))

# 自定义聚合
print(grouped.agg({
    'salary': ['mean', 'std'],
    'age': lambda x: x.max() - x.min()
}))

# 多列分组
df['gender'] = ['F', 'M', 'M', 'M', 'F']
print(df.groupby(['department', 'gender'])['salary'].mean())
```

### 7.3 数据透视表

```python
# pivot_table
pivot = df.pivot_table(
    values='salary',
    index='department',
    columns='gender',
    aggfunc='mean'
)
print(pivot)
# gender           F        M
# department
# HR             NaN  57500.0
# IT         57500.0  70000.0

# 多值多聚合
pivot = df.pivot_table(
    values=['salary', 'age'],
    index='department',
    aggfunc={'salary': 'mean', 'age': 'max'}
)
print(pivot)
```

### 7.4 合并数据

```python
# concat: 简单堆叠
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

pd.concat([df1, df2])              # 垂直堆叠
pd.concat([df1, df2], axis=1)      # 水平堆叠
pd.concat([df1, df2], ignore_index=True)  # 重置索引

# merge: 类似 SQL JOIN
left = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'value': [1, 2, 3]
})
right = pd.DataFrame({
    'key': ['A', 'B', 'D'],
    'value': [4, 5, 6]
})

pd.merge(left, right, on='key')              # 内连接
pd.merge(left, right, on='key', how='left')  # 左连接
pd.merge(left, right, on='key', how='right') # 右连接
pd.merge(left, right, on='key', how='outer') # 外连接

# 不同列名连接
pd.merge(left, right, left_on='key1', right_on='key2')
```

---

## 8. 练习题

### 基础练习

1. 创建一个包含学生姓名、年龄、成绩的 DataFrame
2. 筛选出成绩大于 80 的学生
3. 按成绩降序排序
4. 计算平均成绩

### 进阶练习

5. 加载一个 CSV 数据集，进行数据清洗（处理缺失值、重复值）
6. 使用 groupby 进行分组统计

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
import pandas as pd
import numpy as np

# 1. 创建 DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [20, 21, 22, 20, 21],
    'score': [85, 72, 90, 68, 95]
})
print(df)

# 2. 筛选成绩大于 80
high_scorers = df[df['score'] > 80]
print(high_scorers)

# 3. 按成绩降序排序
sorted_df = df.sort_values('score', ascending=False)
print(sorted_df)

# 4. 计算平均成绩
avg_score = df['score'].mean()
print(f"平均成绩: {avg_score}")

# 5. 数据清洗示例
# 创建带缺失值和重复值的数据
df_dirty = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Alice', 'Charlie', np.nan],
    'age': [25, 30, 25, np.nan, 28],
    'score': [85, np.nan, 85, 90, 75]
})

# 查看缺失值
print(df_dirty.isnull().sum())

# 填充缺失值
df_clean = df_dirty.copy()
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())
df_clean['score'] = df_clean['score'].fillna(df_clean['score'].median())
df_clean['name'] = df_clean['name'].fillna('Unknown')

# 删除重复值
df_clean = df_clean.drop_duplicates()
print(df_clean)

# 6. 分组统计
df = pd.DataFrame({
    'department': ['IT', 'HR', 'IT', 'HR', 'IT', 'Finance'],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'salary': [50000, 45000, 60000, 48000, 55000, 52000]
})

# 按部门统计
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'sum', 'count']
})
print(dept_stats)
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [06-Matplotlib可视化.md](./06-Matplotlib可视化.md)

