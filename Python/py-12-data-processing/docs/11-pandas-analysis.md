# Pandas 数据分析

> 分组聚合、数据透视与时序分析

## 排序

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [5000, 6000, 7000, 5500]
})

# 按单列排序
df.sort_values('age')                    # 升序
df.sort_values('age', ascending=False)   # 降序

# 按多列排序
df.sort_values(['age', 'salary'], ascending=[True, False])

# 按索引排序
df.sort_index()
df.sort_index(ascending=False)

# 原地排序
df.sort_values('age', inplace=True)
```

---

## 分组聚合

### 基础分组

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['IT', 'IT', 'HR', 'HR', 'Sales'],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'salary': [5000, 6000, 4500, 4800, 5500],
    'age': [25, 30, 35, 28, 32]
})

# 单列分组
grouped = df.groupby('department')

# 聚合操作
grouped['salary'].mean()     # 各部门平均薪资
grouped['salary'].sum()      # 各部门薪资总和
grouped['salary'].count()    # 各部门人数
grouped['salary'].max()      # 各部门最高薪资
grouped['salary'].min()      # 各部门最低薪资

# 多列分组
df.groupby(['department', 'age'])['salary'].mean()
```

### 多种聚合

```python
# 多个聚合函数
df.groupby('department')['salary'].agg(['mean', 'sum', 'count'])

# 不同列不同聚合
df.groupby('department').agg({
    'salary': ['mean', 'max'],
    'age': 'mean'
})

# 自定义聚合函数
df.groupby('department')['salary'].agg(lambda x: x.max() - x.min())

# 命名聚合（推荐）
df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    headcount=('name', 'count')
)
```

### 分组遍历

```python
for name, group in df.groupby('department'):
    print(f"Department: {name}")
    print(group)
    print()
```

### 分组转换

```python
# transform: 返回与原 DataFrame 同形状的结果
df['salary_pct'] = df.groupby('department')['salary'].transform(
    lambda x: x / x.sum()
)

# 组内标准化
df['salary_zscore'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

### 分组过滤

```python
# filter: 返回满足条件的分组
df.groupby('department').filter(lambda x: x['salary'].mean() > 5000)
```

---

## 数据透视表

### pivot_table

```python
import pandas as pd

df = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180],
    'quantity': [10, 15, 12, 18]
})

# 基础透视
pd.pivot_table(
    df,
    values='sales',
    index='date',
    columns='product'
)
#          A      B
# 2024-01  100    150
# 2024-02  120    180

# 多值透视
pd.pivot_table(
    df,
    values=['sales', 'quantity'],
    index='date',
    columns='product',
    aggfunc='sum'
)

# 多行多列
pd.pivot_table(
    df,
    values='sales',
    index=['date', 'region'],
    columns='product',
    aggfunc='sum',
    fill_value=0,      # 填充空值
    margins=True       # 添加汇总行/列
)
```

### crosstab（交叉表）

```python
# 计数交叉表
pd.crosstab(df['department'], df['level'])

# 带聚合值
pd.crosstab(
    df['department'],
    df['level'],
    values=df['salary'],
    aggfunc='mean'
)
```

---

## 合并与连接

### concat（拼接）

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# 纵向拼接
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)  # 重置索引

# 横向拼接
pd.concat([df1, df2], axis=1)
```

### merge（关联）

```python
users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'user_id': [1, 2, 4],
    'amount': [100, 200, 300]
})

# 内连接（默认）
pd.merge(users, orders, on='user_id')

# 左连接
pd.merge(users, orders, on='user_id', how='left')

# 右连接
pd.merge(users, orders, on='user_id', how='right')

# 外连接
pd.merge(users, orders, on='user_id', how='outer')

# 不同列名
pd.merge(
    users, orders,
    left_on='user_id',
    right_on='uid'
)
```

### join

```python
# 基于索引连接
df1.set_index('key').join(df2.set_index('key'))
```

---

## 时间序列

### 日期类型

```python
import pandas as pd

# 转换为日期
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# 日期范围
dates = pd.date_range('2024-01-01', periods=10, freq='D')
dates = pd.date_range('2024-01-01', '2024-12-31', freq='M')

# 频率
# D: 天, W: 周, M: 月末, MS: 月初, Q: 季末, Y: 年末
# H: 小时, T/min: 分钟, S: 秒
```

### 日期属性

```python
df['date'] = pd.to_datetime(df['date'])

# 提取日期组件
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday      # 0=周一
df['day_name'] = df['date'].dt.day_name()  # Monday
df['quarter'] = df['date'].dt.quarter      # 季度
df['is_month_end'] = df['date'].dt.is_month_end
```

### 时间索引

```python
# 设置时间索引
df = df.set_index('date')

# 按时间切片
df['2024']              # 2024 年所有数据
df['2024-01']           # 2024 年 1 月
df['2024-01-01':'2024-03-31']  # 日期范围

# 重采样
df.resample('M').sum()      # 按月汇总
df.resample('W').mean()     # 按周平均
df.resample('Q').first()    # 按季度取首值

# 移动窗口
df['rolling_mean'] = df['value'].rolling(7).mean()   # 7 日移动平均
df['expanding_sum'] = df['value'].expanding().sum()  # 累计求和
```

### 时间偏移

```python
from pandas.tseries.offsets import Day, Week, MonthEnd

df['date'] + Day(7)         # 加 7 天
df['date'] + MonthEnd(1)    # 到月末
df['date'] - Week(2)        # 减 2 周

# shift
df['prev_value'] = df['value'].shift(1)   # 上一期
df['next_value'] = df['value'].shift(-1)  # 下一期
df['change'] = df['value'] - df['value'].shift(1)  # 环比变化
df['pct_change'] = df['value'].pct_change()  # 环比增长率
```

---

## 统计分析

### 描述性统计

```python
# 数值统计
df['salary'].mean()      # 均值
df['salary'].median()    # 中位数
df['salary'].std()       # 标准差
df['salary'].var()       # 方差
df['salary'].quantile(0.75)  # 75% 分位数

# 计数
df['category'].value_counts()       # 频数
df['category'].value_counts(normalize=True)  # 频率

# 唯一值
df['category'].unique()       # 唯一值数组
df['category'].nunique()      # 唯一值数量
```

### 相关性

```python
# 相关系数矩阵
df[['age', 'salary', 'bonus']].corr()

# 两列相关系数
df['age'].corr(df['salary'])
```

### 分箱

```python
# 等宽分箱
df['age_bin'] = pd.cut(df['age'], bins=3)
df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['青年', '中年', '老年'])

# 等频分箱
df['salary_quantile'] = pd.qcut(df['salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

---

## 实战示例

### 销售数据分析

```python
import pandas as pd

# 模拟数据
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'product': ['A', 'B', 'C'] * 33 + ['A'],
    'region': ['北方', '南方'] * 50,
    'sales': [100 + i * 2 + (i % 7) * 10 for i in range(100)],
    'quantity': [10 + i % 5 for i in range(100)]
})

# 1. 总体概览
print("数据概览:")
print(df.describe())

# 2. 按产品分析
product_analysis = df.groupby('product').agg(
    总销售额=('sales', 'sum'),
    平均销售额=('sales', 'mean'),
    销售次数=('sales', 'count')
).round(2)
print("\n按产品分析:")
print(product_analysis)

# 3. 按地区和产品交叉分析
pivot = pd.pivot_table(
    df,
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    margins=True
)
print("\n地区-产品透视表:")
print(pivot)

# 4. 时间趋势（按周汇总）
df = df.set_index('date')
weekly = df.resample('W')['sales'].sum()
print("\n周销售趋势:")
print(weekly)

# 5. 移动平均
df['sales_ma7'] = df['sales'].rolling(7).mean()
```

---

## 性能优化

### 数据类型优化

```python
# 减少内存占用
df['category'] = df['category'].astype('category')
df['small_int'] = df['small_int'].astype('int8')
df['id'] = df['id'].astype('int32')

# 查看内存使用
print(df.memory_usage(deep=True))
```

### 分块读取

```python
# 大文件分块处理
chunks = pd.read_csv('huge_file.csv', chunksize=100000)

results = []
for chunk in chunks:
    # 处理每个块
    result = chunk.groupby('category')['value'].sum()
    results.append(result)

# 合并结果
final = pd.concat(results).groupby(level=0).sum()
```

### 使用 eval/query

```python
# 大数据集上更快
df.eval('new_col = col1 + col2')
df.query('age > 30 and salary > 5000')
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| groupby 后丢失列 | 非聚合列被排除 | 使用 `as_index=False` 或 `.reset_index()` |
| merge 重复列 | 出现 `_x`, `_y` 后缀 | 指定 `suffixes` 或提前重命名 |
| 时间索引切片 | 字符串格式错误 | 使用 ISO 格式 `'2024-01-01'` |
| resample 无效 | 索引不是时间类型 | 先 `set_index` 时间列 |

---

## 小结

1. **分组聚合**：`groupby` + `agg`，支持多列多函数
2. **透视表**：`pivot_table` 创建交叉分析
3. **合并**：`concat` 拼接，`merge` 关联
4. **时间序列**：`resample` 重采样，`rolling` 滑动窗口
5. **性能**：合适的 dtype，分块处理大文件

