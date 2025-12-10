# 🔥 02 - PyTorch 基础：Tensor

> Tensor 是 PyTorch 的核心数据结构，类似 NumPy 的 ndarray，但支持 GPU 加速和自动求导

---

## 目录

1. [Tensor 创建](#1-tensor-创建)
2. [Tensor 属性](#2-tensor-属性)
3. [索引与切片](#3-索引与切片)
4. [形状操作](#4-形状操作)
5. [数学运算](#5-数学运算)
6. [设备管理](#6-设备管理)
7. [与 NumPy 互转](#7-与-numpy-互转)
8. [练习题](#8-练习题)

---

## 1. Tensor 创建

### 1.1 从数据创建

```python
import torch
import numpy as np

# 从 Python 列表
t1 = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表: {t1}")

# 从嵌套列表（矩阵）
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"从嵌套列表:\n{t2}")

# 从 NumPy 数组
arr = np.array([1.0, 2.0, 3.0])
t3 = torch.from_numpy(arr)  # 共享内存
t4 = torch.tensor(arr)      # 复制数据
print(f"从 NumPy: {t3}")

# 指定数据类型
t5 = torch.tensor([1, 2, 3], dtype=torch.float32)
t6 = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"float32: {t5.dtype}")
print(f"int64: {t6.dtype}")
```

### 1.2 特殊 Tensor

```python
# 全零
zeros = torch.zeros(3, 4)
print(f"全零:\n{zeros}")

# 全一
ones = torch.ones(3, 4)
print(f"全一:\n{ones}")

# 单位矩阵
eye = torch.eye(3)
print(f"单位矩阵:\n{eye}")

# 填充特定值
full = torch.full((2, 3), fill_value=7)
print(f"填充7:\n{full}")

# 未初始化（随机值，速度快）
empty = torch.empty(2, 3)
print(f"未初始化:\n{empty}")

# 与另一个 Tensor 相同形状
x = torch.randn(2, 3)
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
```

### 1.3 序列和随机

```python
# arange
t = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
print(f"arange: {t}")

# linspace
t = torch.linspace(0, 1, 5)  # 5 个均匀分布的点
print(f"linspace: {t}")

# 随机数
torch.manual_seed(42)  # 设置随机种子

# 均匀分布 [0, 1)
uniform = torch.rand(3, 4)

# 标准正态分布
normal = torch.randn(3, 4)

# 指定范围的随机整数
integers = torch.randint(0, 10, (3, 4))

# 指定范围的均匀分布
uniform_range = torch.empty(3, 4).uniform_(-1, 1)

# 指定参数的正态分布
normal_params = torch.normal(mean=0, std=1, size=(3, 4))

print(f"均匀分布:\n{uniform}")
print(f"正态分布:\n{normal}")
```

---

## 2. Tensor 属性

```python
t = torch.randn(2, 3, 4)

# 基本属性
print(f"形状: {t.shape}")           # torch.Size([2, 3, 4])
print(f"形状: {t.size()}")          # 同上
print(f"维度数: {t.dim()}")         # 3
print(f"元素总数: {t.numel()}")     # 24
print(f"数据类型: {t.dtype}")       # torch.float32
print(f"设备: {t.device}")          # cpu 或 cuda:0

# 检查属性
print(f"是否需要梯度: {t.requires_grad}")
print(f"是否是叶子节点: {t.is_leaf}")
print(f"是否是 CUDA: {t.is_cuda}")

# 常用数据类型
dtypes = [
    torch.float32,   # torch.float, 默认浮点类型
    torch.float64,   # torch.double
    torch.float16,   # torch.half, 混合精度训练
    torch.bfloat16,  # Brain Float 16
    torch.int32,     # torch.int
    torch.int64,     # torch.long, 默认整数类型
    torch.bool       # 布尔类型
]
```

---

## 3. 索引与切片

### 3.1 基本索引

```python
t = torch.arange(12).reshape(3, 4)
print(f"原始 Tensor:\n{t}")
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 单个元素
print(f"t[1, 2] = {t[1, 2]}")  # 6

# 行
print(f"t[0] = {t[0]}")  # [0, 1, 2, 3]

# 列
print(f"t[:, 1] = {t[:, 1]}")  # [1, 5, 9]

# 切片
print(f"t[0:2, 1:3] =\n{t[0:2, 1:3]}")
# [[1, 2],
#  [5, 6]]

# 步长
print(f"t[::2, ::2] =\n{t[::2, ::2]}")
# [[ 0,  2],
#  [ 8, 10]]
```

### 3.2 高级索引

```python
# 布尔索引
t = torch.randn(4, 4)
mask = t > 0
print(f"正数: {t[mask]}")

# 设置值
t[t < 0] = 0  # 将所有负数设为 0
print(t)

# 花式索引
t = torch.arange(12).reshape(3, 4)
indices = torch.tensor([0, 2])
print(f"选择第 0 和 2 行:\n{t[indices]}")

# gather: 按索引收集
t = torch.tensor([[1, 2], [3, 4]])
idx = torch.tensor([[0, 0], [1, 0]])
result = torch.gather(t, dim=1, index=idx)
print(f"gather 结果:\n{result}")
# [[1, 1],
#  [4, 3]]
```

### 3.3 常用索引操作

```python
# where: 条件选择
x = torch.randn(3, 3)
y = torch.ones(3, 3)
result = torch.where(x > 0, x, y)  # x>0 取 x，否则取 y
print(f"where 结果:\n{result}")

# masked_select: 按掩码选择
mask = x > 0
selected = torch.masked_select(x, mask)
print(f"正数元素: {selected}")

# index_select: 按索引选择
t = torch.arange(12).reshape(3, 4)
result = torch.index_select(t, dim=0, index=torch.tensor([0, 2]))
print(f"选择第 0 和 2 行:\n{result}")
```

---

## 4. 形状操作

### 4.1 reshape 和 view

```python
t = torch.arange(12)

# reshape: 改变形状
t1 = t.reshape(3, 4)
t2 = t.reshape(2, 2, 3)
t3 = t.reshape(3, -1)  # -1 自动计算

print(f"reshape(3, 4):\n{t1}")
print(f"reshape(2, 2, 3):\n{t2}")

# view: 和 reshape 类似，但要求内存连续
t4 = t.view(3, 4)

# 什么时候用 reshape vs view？
# view 更快（不复制数据），但要求内存连续
# reshape 更通用，必要时会复制数据
```

### 4.2 squeeze 和 unsqueeze

```python
# squeeze: 去除大小为 1 的维度
t = torch.randn(1, 3, 1, 4)
print(f"原始形状: {t.shape}")  # [1, 3, 1, 4]

t1 = t.squeeze()      # 去除所有大小为 1 的维度
print(f"squeeze(): {t1.shape}")  # [3, 4]

t2 = t.squeeze(0)     # 只去除第 0 维
print(f"squeeze(0): {t2.shape}")  # [3, 1, 4]

# unsqueeze: 增加大小为 1 的维度
t = torch.randn(3, 4)
print(f"原始形状: {t.shape}")  # [3, 4]

t1 = t.unsqueeze(0)   # 在第 0 维增加
print(f"unsqueeze(0): {t1.shape}")  # [1, 3, 4]

t2 = t.unsqueeze(-1)  # 在最后增加
print(f"unsqueeze(-1): {t2.shape}")  # [3, 4, 1]

# 常见用法：给 batch 维度
single_image = torch.randn(3, 224, 224)  # [C, H, W]
batch_image = single_image.unsqueeze(0)  # [1, C, H, W]
```

### 4.3 转置和维度交换

```python
t = torch.randn(2, 3, 4)

# transpose: 交换两个维度
t1 = t.transpose(0, 1)
print(f"transpose(0, 1): {t1.shape}")  # [3, 2, 4]

# permute: 任意重排维度
t2 = t.permute(2, 0, 1)
print(f"permute(2, 0, 1): {t2.shape}")  # [4, 2, 3]

# 2D 矩阵转置
m = torch.randn(3, 4)
mt = m.T  # 或 m.t()
print(f"矩阵转置: {m.shape} -> {mt.shape}")  # [3, 4] -> [4, 3]

# 图像格式转换
# PyTorch: [N, C, H, W]
# TensorFlow/PIL: [N, H, W, C]
img_pytorch = torch.randn(32, 3, 224, 224)
img_tf = img_pytorch.permute(0, 2, 3, 1)
print(f"PyTorch -> TF: {img_pytorch.shape} -> {img_tf.shape}")
```

### 4.4 拼接和分割

```python
# cat: 沿现有维度拼接
a = torch.randn(2, 3)
b = torch.randn(2, 3)

c = torch.cat([a, b], dim=0)  # 沿第 0 维
print(f"cat dim=0: {c.shape}")  # [4, 3]

d = torch.cat([a, b], dim=1)  # 沿第 1 维
print(f"cat dim=1: {d.shape}")  # [2, 6]

# stack: 沿新维度堆叠
e = torch.stack([a, b], dim=0)
print(f"stack dim=0: {e.shape}")  # [2, 2, 3]

# split: 均匀分割
chunks = torch.split(c, split_size_or_sections=2, dim=0)
print(f"split: {[chunk.shape for chunk in chunks]}")  # [[2, 3], [2, 3]]

# chunk: 分成 N 份
chunks = torch.chunk(c, chunks=2, dim=0)
print(f"chunk: {[chunk.shape for chunk in chunks]}")  # [[2, 3], [2, 3]]
```

---

## 5. 数学运算

### 5.1 基本运算

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

# 加减乘除
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # 逐元素乘法
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")  # 幂运算

# 函数形式
print(f"torch.add(a, b) = {torch.add(a, b)}")
print(f"torch.mul(a, b) = {torch.mul(a, b)}")

# 原地操作（节省内存，但会影响梯度计算）
a.add_(1)  # a = a + 1
print(f"原地加法: {a}")
```

### 5.2 矩阵运算

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# 矩阵乘法
C = torch.mm(A, B)       # 2D 矩阵乘法
C = torch.matmul(A, B)   # 通用，支持广播
C = A @ B                # 运算符形式
print(f"矩阵乘法: {A.shape} @ {B.shape} = {C.shape}")  # [2, 4]

# 批量矩阵乘法
batch_A = torch.randn(32, 2, 3)
batch_B = torch.randn(32, 3, 4)
batch_C = torch.bmm(batch_A, batch_B)  # [32, 2, 4]
# 或
batch_C = batch_A @ batch_B

# 向量点积
v1 = torch.tensor([1., 2., 3.])
v2 = torch.tensor([4., 5., 6.])
dot = torch.dot(v1, v2)
print(f"点积: {dot}")  # 32

# 外积
outer = torch.outer(v1, v2)
print(f"外积形状: {outer.shape}")  # [3, 3]
```

### 5.3 规约运算

```python
t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

# 求和
print(f"总和: {t.sum()}")
print(f"按行求和: {t.sum(dim=1)}")  # [6, 15]
print(f"按列求和: {t.sum(dim=0)}")  # [5, 7, 9]

# 均值
print(f"均值: {t.mean()}")
print(f"按行均值: {t.mean(dim=1)}")

# 最值
print(f"最大值: {t.max()}")
print(f"最小值: {t.min()}")

# 最值及其索引
max_val, max_idx = t.max(dim=1)
print(f"每行最大值: {max_val}, 索引: {max_idx}")

# argmax/argmin
print(f"最大值索引: {t.argmax()}")
print(f"每行最大值索引: {t.argmax(dim=1)}")

# 其他
print(f"标准差: {t.std()}")
print(f"方差: {t.var()}")
print(f"累积和: {t.cumsum(dim=1)}")
```

### 5.4 广播机制

```python
# PyTorch 的广播规则与 NumPy 相同
a = torch.randn(3, 4)
b = torch.randn(4)      # 自动扩展为 [3, 4]
c = torch.randn(3, 1)   # 自动扩展为 [3, 4]

print(f"a + b: {(a + b).shape}")  # [3, 4]
print(f"a + c: {(a + c).shape}")  # [3, 4]
print(f"b + c: {(b + c).shape}")  # [3, 4]

# 常见用法：对每行减去均值
x = torch.randn(32, 100)
mean = x.mean(dim=1, keepdim=True)  # [32, 1]
x_centered = x - mean  # 广播相减
```

---

## 6. 设备管理

### 6.1 GPU 基础

```python
# 检查 CUDA
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
```

### 6.2 设备转移

```python
# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建时指定设备
t = torch.randn(3, 4, device=device)
print(f"Tensor 设备: {t.device}")

# 转移设备
t_cpu = torch.randn(3, 4)
t_gpu = t_cpu.to(device)  # 推荐方式
# 或
t_gpu = t_cpu.cuda()      # 直接转到 GPU
t_cpu = t_gpu.cpu()       # 转回 CPU

# 模型转移
model = MyModel()
model = model.to(device)

# 数据转移
for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)
    output = model(x)
```

### 6.3 内存管理

```python
if torch.cuda.is_available():
    # 查看 GPU 内存
    print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # 清空缓存
    torch.cuda.empty_cache()

    # 重置统计信息
    torch.cuda.reset_peak_memory_stats()
```

---

## 7. 与 NumPy 互转

```python
import numpy as np

# Tensor → NumPy
t = torch.randn(3, 4)
arr = t.numpy()  # 共享内存（CPU 上）
arr = t.detach().cpu().numpy()  # 安全方式（GPU 或有梯度时）

# NumPy → Tensor
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)  # 共享内存
t = torch.tensor(arr)      # 复制数据

# 注意：共享内存意味着修改一个会影响另一个
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)
arr[0] = 100
print(t)  # tensor([100.,   2.,   3.])
```

---

## 8. 练习题

### 基础练习

1. 创建一个形状为 [3, 4, 5] 的随机 Tensor，然后 reshape 成 [12, 5]
2. 给定一个形状为 [32, 10] 的 Tensor（表示 32 个样本的 10 分类 logits），找出每个样本的预测类别
3. 实现批量归一化：对形状 [batch, features] 的 Tensor，每个特征减均值除标准差

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
import torch

# 1. reshape
t = torch.randn(3, 4, 5)
t_reshaped = t.reshape(12, 5)
print(f"原始: {t.shape}, reshape后: {t_reshaped.shape}")

# 2. 找预测类别
logits = torch.randn(32, 10)
predictions = logits.argmax(dim=1)
print(f"预测类别形状: {predictions.shape}")  # [32]
print(f"前5个预测: {predictions[:5]}")

# 3. 批量归一化
x = torch.randn(32, 100)
mean = x.mean(dim=0, keepdim=True)  # [1, 100]
std = x.std(dim=0, keepdim=True)    # [1, 100]
x_normalized = (x - mean) / (std + 1e-8)

print(f"归一化后均值: {x_normalized.mean(dim=0).mean():.6f}")  # 约 0
print(f"归一化后标准差: {x_normalized.std(dim=0).mean():.6f}")  # 约 1
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [03-PyTorch基础-autograd.md](./03-PyTorch基础-autograd.md)

