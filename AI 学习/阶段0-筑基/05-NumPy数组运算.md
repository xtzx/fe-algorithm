# 🔢 05 - NumPy 数组运算

> NumPy 是 Python 科学计算的基础，几乎所有 AI 库都依赖它

---

## 目录

1. [NumPy 简介](#1-numpy-简介)
2. [数组创建](#2-数组创建)
3. [数组操作](#3-数组操作)
4. [数学运算](#4-数学运算)
5. [广播机制](#5-广播机制)
6. [常用函数](#6-常用函数)
7. [练习题](#7-练习题)

---

## 1. NumPy 简介

### 1.1 为什么用 NumPy？

```python
import numpy as np
import time

# Python 列表 vs NumPy 数组
size = 1000000

# Python 列表运算
python_list = list(range(size))
start = time.time()
result = [x * 2 for x in python_list]
print(f"Python list: {time.time() - start:.4f}s")

# NumPy 数组运算
numpy_array = np.arange(size)
start = time.time()
result = numpy_array * 2
print(f"NumPy array: {time.time() - start:.4f}s")

# NumPy 通常快 10-100 倍！
```

### 1.2 安装和导入

```python
# 安装
# pip install numpy

# 导入（约定使用 np 作为别名）
import numpy as np

print(np.__version__)
```

---

## 2. 数组创建

### 2.1 从列表创建

```python
import numpy as np

# 一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)        # [1 2 3 4 5]
print(arr1.dtype)  # int64

# 二维数组（矩阵）
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# 指定数据类型
arr_float = np.array([1, 2, 3], dtype=np.float32)
print(arr_float.dtype)  # float32
```

### 2.2 特殊数组

```python
# 全零数组
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 全一数组
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# 单位矩阵
eye = np.eye(3)
print(eye)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 空数组（未初始化，内容随机）
empty = np.empty((2, 2))

# 填充特定值
full = np.full((2, 3), 7)
print(full)
# [[7 7 7]
#  [7 7 7]]
```

### 2.3 序列数组

```python
# arange: 类似 range
arr = np.arange(0, 10, 2)  # 起始, 结束, 步长
print(arr)  # [0 2 4 6 8]

# linspace: 等间隔划分
arr = np.linspace(0, 1, 5)  # 0到1之间5个数
print(arr)  # [0.   0.25 0.5  0.75 1.  ]

# logspace: 对数间隔
arr = np.logspace(0, 3, 4)  # 10^0 到 10^3
print(arr)  # [   1.   10.  100. 1000.]
```

### 2.4 随机数组

```python
np.random.seed(42)  # 设置随机种子，保证可复现

# 均匀分布 [0, 1)
uniform = np.random.rand(3, 4)

# 标准正态分布 (均值0, 标准差1)
normal = np.random.randn(3, 4)

# 指定范围的随机整数
integers = np.random.randint(0, 10, (3, 4))

# 指定范围的均匀分布
uniform_range = np.random.uniform(-1, 1, (3, 4))

# 指定参数的正态分布
normal_params = np.random.normal(loc=5, scale=2, size=(3, 4))

# 随机打乱
arr = np.arange(10)
np.random.shuffle(arr)
print(arr)

# 随机选择
choices = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
print(choices)
```

### 2.5 数组属性

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"形状: {arr.shape}")      # (2, 3)
print(f"维度: {arr.ndim}")       # 2
print(f"元素数量: {arr.size}")   # 6
print(f"数据类型: {arr.dtype}")  # int64
print(f"元素字节数: {arr.itemsize}")  # 8
print(f"总字节数: {arr.nbytes}")      # 48
```

---

## 3. 数组操作

### 3.1 索引和切片

```python
# 一维数组
arr = np.arange(10)
print(arr)        # [0 1 2 3 4 5 6 7 8 9]
print(arr[3])     # 3
print(arr[-1])    # 9
print(arr[2:7])   # [2 3 4 5 6]
print(arr[::2])   # [0 2 4 6 8] 步长为2
print(arr[::-1])  # [9 8 7 6 5 4 3 2 1 0] 反转

# 二维数组
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

print(arr2d[1, 2])      # 6 - 第2行第3列
print(arr2d[0])         # [1 2 3] - 第1行
print(arr2d[:, 1])      # [2 5 8] - 第2列
print(arr2d[0:2, 1:3])  # [[2 3] [5 6]] - 子矩阵
```

### 3.2 布尔索引

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 条件筛选
mask = arr > 3
print(mask)         # [False False False  True  True  True]
print(arr[mask])    # [4 5 6]

# 直接使用条件
print(arr[arr > 3])  # [4 5 6]

# 多条件
print(arr[(arr > 2) & (arr < 5)])  # [3 4]
print(arr[(arr < 2) | (arr > 5)])  # [1 6]

# 赋值
arr[arr > 3] = 0
print(arr)  # [1 2 3 0 0 0]
```

### 3.3 花式索引

```python
arr = np.arange(10, 20)
print(arr)  # [10 11 12 13 14 15 16 17 18 19]

# 使用索引数组
indices = [0, 3, 5, 7]
print(arr[indices])  # [10 13 15 17]

# 二维数组
arr2d = np.arange(12).reshape(3, 4)
print(arr2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 选择特定位置的元素
print(arr2d[[0, 1, 2], [0, 1, 2]])  # [0 5 10] 对角线元素
```

### 3.4 形状变换

```python
arr = np.arange(12)

# reshape: 改变形状
arr_2d = arr.reshape(3, 4)
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1 自动计算
arr_2d = arr.reshape(3, -1)  # 3行，列数自动计算
arr_2d = arr.reshape(-1, 4)  # 列数4，行数自动计算

# flatten: 展平为一维
flat = arr_2d.flatten()
print(flat)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# ravel: 展平（返回视图，更高效）
raveled = arr_2d.ravel()

# transpose: 转置
transposed = arr_2d.T
print(transposed)
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
```

### 3.5 数组拼接和分割

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 拼接
print(np.concatenate([a, b]))  # [1 2 3 4 5 6]
print(np.stack([a, b]))        # [[1 2 3] [4 5 6]] 沿新轴堆叠
print(np.vstack([a, b]))       # [[1 2 3] [4 5 6]] 垂直堆叠
print(np.hstack([a, b]))       # [1 2 3 4 5 6] 水平堆叠

# 二维数组拼接
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(np.vstack([arr1, arr2]))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

print(np.hstack([arr1, arr2]))
# [[1 2 5 6]
#  [3 4 7 8]]

# 分割
arr = np.arange(12)
print(np.split(arr, 3))  # [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
```

---

## 4. 数学运算

### 4.1 元素级运算

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# 基本运算（逐元素）
print(a + b)   # [11 22 33 44]
print(a - b)   # [-9 -18 -27 -36]
print(a * b)   # [10 40 90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]
print(a ** 2)  # [1 4 9 16]

# 与标量运算
print(a + 10)  # [11 12 13 14]
print(a * 2)   # [2 4 6 8]

# 数学函数
print(np.sqrt(a))   # [1.   1.41 1.73 2.  ]
print(np.exp(a))    # [ 2.72  7.39 20.09 54.60]
print(np.log(a))    # [0.   0.69 1.10 1.39]
print(np.sin(a))    # [0.84 0.91 0.14 -0.76]
```

### 4.2 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 元素级乘法（不是矩阵乘法）
print(A * B)
# [[ 5 12]
#  [21 32]]

# 矩阵乘法
print(np.dot(A, B))
# [[19 22]
#  [43 50]]

# 或者使用 @ 运算符（Python 3.5+）
print(A @ B)
# [[19 22]
#  [43 50]]

# 向量点积
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(np.dot(v1, v2))  # 32 = 1*4 + 2*5 + 3*6
```

### 4.3 线性代数

```python
A = np.array([[1, 2], [3, 4]])

# 转置
print(A.T)

# 行列式
print(np.linalg.det(A))  # -2.0

# 逆矩阵
print(np.linalg.inv(A))
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 解线性方程组 Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(f"解: {x}")  # [1. 2.]

# 矩阵范数
print(np.linalg.norm(A))        # Frobenius 范数
print(np.linalg.norm(A, ord=1)) # 1-范数
print(np.linalg.norm(A, ord=2)) # 2-范数（谱范数）
```

---

## 5. 广播机制

### 5.1 广播规则

```
广播让不同形状的数组可以进行运算
规则：
1. 如果维度数不同，在较小数组的形状左边补1
2. 如果某维度大小不同，且其中一个为1，则扩展为较大的那个
3. 如果某维度大小不同且都不为1，则报错
```

### 5.2 广播示例

```python
# 标量和数组
arr = np.array([1, 2, 3])
print(arr + 10)  # [11 12 13]
# 10 被广播为 [10, 10, 10]

# 一维和二维
arr1 = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
arr2 = np.array([10, 20, 30])             # (3,) -> (1, 3) -> (2, 3)
print(arr1 + arr2)
# [[11 22 33]
#  [14 25 36]]

# 列向量和行向量
col = np.array([[1], [2], [3]])  # (3, 1)
row = np.array([10, 20, 30])      # (3,) -> (1, 3)
print(col + row)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]

# 实用：标准化
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
mean = data.mean(axis=0)  # 每列的均值 [4. 5. 6.]
std = data.std(axis=0)    # 每列的标准差
normalized = (data - mean) / std
print(normalized)
```

---

## 6. 常用函数

### 6.1 统计函数

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 基本统计
print(np.sum(arr))        # 21
print(np.mean(arr))       # 3.5
print(np.std(arr))        # 1.707...
print(np.var(arr))        # 2.916...
print(np.min(arr))        # 1
print(np.max(arr))        # 6

# 按轴计算
print(np.sum(arr, axis=0))  # [5 7 9] 每列的和
print(np.sum(arr, axis=1))  # [6 15] 每行的和
print(np.mean(arr, axis=0)) # [2.5 3.5 4.5] 每列的均值

# 累积
print(np.cumsum(arr))       # [ 1  3  6 10 15 21] 累积和
print(np.cumprod(arr))      # [  1   2   6  24 120 720] 累积积

# 位置
print(np.argmax(arr))       # 5 最大值的索引
print(np.argmin(arr))       # 0 最小值的索引
print(np.argsort(arr[0]))   # [0 1 2] 排序后的索引
```

### 6.2 比较和逻辑

```python
arr = np.array([1, 2, 3, 4, 5])

# 比较运算
print(arr > 3)        # [False False False  True  True]
print(arr == 3)       # [False False  True False False]
print(np.greater(arr, 3))  # 等价于 arr > 3

# 逻辑运算
a = np.array([True, True, False])
b = np.array([True, False, False])
print(np.logical_and(a, b))  # [ True False False]
print(np.logical_or(a, b))   # [ True  True False]
print(np.logical_not(a))     # [False False  True]

# 条件选择
print(np.where(arr > 3, 1, 0))  # [0 0 0 1 1]
print(np.where(arr > 3))        # (array([3, 4]),) 满足条件的索引

# 判断
print(np.all(arr > 0))   # True 是否全部满足
print(np.any(arr > 4))   # True 是否有满足的
```

### 6.3 复制和视图

```python
arr = np.array([1, 2, 3, 4, 5])

# 视图（共享内存）
view = arr[1:4]
view[0] = 100
print(arr)  # [  1 100   3   4   5] 原数组也被修改！

# 复制（独立内存）
arr = np.array([1, 2, 3, 4, 5])
copy = arr[1:4].copy()
copy[0] = 100
print(arr)  # [1 2 3 4 5] 原数组不变
```

---

## 7. 练习题

### 基础练习

1. 创建一个 5x5 的单位矩阵
2. 创建一个 10 个元素的数组，包含 0 到 1 之间均匀分布的数
3. 计算两个向量的点积和夹角余弦
4. 对一个二维数组的每一列进行标准化（减均值除标准差）

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
import numpy as np

# 1. 5x5 单位矩阵
identity = np.eye(5)
print(identity)

# 2. 0到1之间的10个均匀数
uniform = np.linspace(0, 1, 10)
print(uniform)

# 3. 向量点积和夹角余弦
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)
cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"点积: {dot_product}")
print(f"夹角余弦: {cos_angle}")

# 4. 列标准化
data = np.array([[1, 200, 3000], [4, 500, 6000], [7, 800, 9000]], dtype=float)
mean = data.mean(axis=0)
std = data.std(axis=0)
normalized = (data - mean) / std
print("标准化后:")
print(normalized)
print(f"每列均值: {normalized.mean(axis=0)}")  # 接近0
print(f"每列标准差: {normalized.std(axis=0)}")  # 接近1
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [06-Pandas数据处理.md](./06-Pandas数据处理.md)

