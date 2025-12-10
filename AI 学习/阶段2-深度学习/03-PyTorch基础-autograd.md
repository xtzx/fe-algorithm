# 🔥 03 - PyTorch 基础：autograd 自动求导

> autograd 是 PyTorch 的核心，自动计算梯度，让反向传播变得简单

---

## 目录

1. [自动求导基础](#1-自动求导基础)
2. [计算图](#2-计算图)
3. [梯度计算细节](#3-梯度计算细节)
4. [常见操作](#4-常见操作)
5. [反向传播实战](#5-反向传播实战)
6. [练习题](#6-练习题)

---

## 1. 自动求导基础

### 1.1 requires_grad

```python
import torch

# 需要计算梯度的 Tensor
x = torch.tensor([1., 2., 3.], requires_grad=True)
print(f"requires_grad: {x.requires_grad}")

# 默认不需要梯度
y = torch.tensor([4., 5., 6.])
print(f"默认 requires_grad: {y.requires_grad}")

# 修改 requires_grad
y.requires_grad_(True)  # 原地修改
print(f"修改后: {y.requires_grad}")

# 创建时指定
z = torch.randn(3, requires_grad=True)
```

### 1.2 简单求导示例

```python
# 计算 y = x^2 的导数 dy/dx = 2x
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2

print(f"x = {x}")
print(f"y = x^2 = {y}")

# 对标量反向传播
y_sum = y.sum()  # 需要是标量才能直接 backward
y_sum.backward()

print(f"dy/dx = 2x = {x.grad}")  # [2., 4., 6.]

# 验证：在 x=2 处，dy/dx = 2*2 = 4 ✓
```

### 1.3 链式法则

```python
# y = x^2, z = y^3
# dz/dx = dz/dy * dy/dx = 3y^2 * 2x = 3(x^2)^2 * 2x = 6x^5

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y ** 3

z.backward()

print(f"x = {x}")
print(f"y = x^2 = {y}")
print(f"z = y^3 = {z}")
print(f"dz/dx = {x.grad}")  # 6 * 2^5 = 192

# 手动验证
print(f"手动计算: 6 * x^5 = {6 * (2.0 ** 5)}")  # 192.0 ✓
```

---

## 2. 计算图

### 2.1 动态计算图

```
PyTorch 使用动态计算图（Define-by-Run）：
- 每次前向传播都会构建新的计算图
- 反向传播后图会被释放（除非 retain_graph=True）
- 允许使用 Python 控制流（if/for）
```

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# 动态计算图：可以使用 Python 控制流
if x > 0:
    y = x ** 2
else:
    y = -x ** 2

y.backward()
print(f"梯度: {x.grad}")  # 4.0

# 条件不同，计算图也不同
x = torch.tensor(-2.0, requires_grad=True)
if x > 0:
    y = x ** 2
else:
    y = -x ** 2

y.backward()
print(f"梯度: {x.grad}")  # 4.0 (因为 y = -x^2, dy/dx = -2x = -2*(-2) = 4)
```

### 2.2 叶子节点

```python
# 叶子节点：由用户创建且 requires_grad=True 的 Tensor
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
z = y + 1

print(f"x is_leaf: {x.is_leaf}")  # True - 叶子节点
print(f"y is_leaf: {y.is_leaf}")  # False - 由运算产生
print(f"z is_leaf: {z.is_leaf}")  # False

# 只有叶子节点的梯度会被保留
z.sum().backward()
print(f"x.grad: {x.grad}")  # 保留
print(f"y.grad: {y.grad}")  # None，中间节点梯度不保留

# 如果需要中间节点的梯度，使用 retain_grad()
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.retain_grad()  # 保留 y 的梯度
z = y + 1
z.sum().backward()
print(f"y.grad (retain): {y.grad}")  # 现在有值了
```

### 2.3 grad_fn

```python
# 每个 Tensor 记录了产生它的操作
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
z = y.sum()

print(f"x.grad_fn: {x.grad_fn}")  # None - 叶子节点
print(f"y.grad_fn: {y.grad_fn}")  # <PowBackward0>
print(f"z.grad_fn: {z.grad_fn}")  # <SumBackward0>

# grad_fn 构成了计算图的反向链接
```

---

## 3. 梯度计算细节

### 3.1 梯度累积

```python
# 梯度是累积的，不会自动清零
x = torch.tensor([1., 2., 3.], requires_grad=True)

# 第一次
y = (x ** 2).sum()
y.backward()
print(f"第一次: {x.grad}")  # [2, 4, 6]

# 第二次（梯度累积）
y = (x ** 2).sum()
y.backward()
print(f"第二次（累积）: {x.grad}")  # [4, 8, 12]

# 清零梯度
x.grad.zero_()  # 或 x.grad = None
y = (x ** 2).sum()
y.backward()
print(f"清零后: {x.grad}")  # [2, 4, 6]

# 在训练循环中，务必清零梯度！
# optimizer.zero_grad()
```

### 3.2 非标量反向传播

```python
# backward() 默认只能对标量调用
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2  # y 是向量

# 直接 backward 会报错
# y.backward()  # RuntimeError

# 方法 1：先求和变成标量
y.sum().backward()
print(f"y.sum().backward(): {x.grad}")

# 方法 2：传入 gradient 参数
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
gradient = torch.tensor([1., 1., 1.])  # 外部梯度
y.backward(gradient)
print(f"y.backward(gradient): {x.grad}")

# gradient 的作用是指定 "外部对 y 的梯度"
# 实际计算的是 (dy/dx) * gradient
```

### 3.3 停止梯度

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)

# 方法 1：with torch.no_grad()
with torch.no_grad():
    y = x ** 2
print(f"no_grad 内: y.requires_grad = {y.requires_grad}")  # False

# 方法 2：detach()
y = x ** 2
y_detached = y.detach()  # 返回一个不需要梯度的副本
print(f"detach: y_detached.requires_grad = {y_detached.requires_grad}")  # False

# 常见用途：
# 1. 推理时不需要梯度
# 2. 冻结部分网络
# 3. 计算指标时
```

### 3.4 保留计算图

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
z = y.sum()

# 第一次 backward 后图会被释放
z.backward(retain_graph=True)  # 保留图
print(f"第一次: {x.grad}")

x.grad.zero_()
z.backward()  # 可以再次 backward
print(f"第二次: {x.grad}")

# 不保留的话第二次会报错
# RuntimeError: Trying to backward through the graph a second time
```

---

## 4. 常见操作

### 4.1 冻结参数

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 冻结前两层
for param in model[0].parameters():
    param.requires_grad = False

# 检查
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

# 只优化未冻结的参数
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
```

### 4.2 检查梯度

```python
x = torch.randn(3, requires_grad=True)
y = x.sum()
y.backward()

# 检查梯度
print(f"梯度: {x.grad}")
print(f"梯度是否存在: {x.grad is not None}")
print(f"梯度形状: {x.grad.shape}")

# 检查梯度是否有问题
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: grad_norm = {grad_norm:.6f}")
            if torch.isnan(grad_norm):
                print(f"  警告：发现 NaN 梯度！")
            if grad_norm > 100:
                print(f"  警告：梯度可能爆炸！")
```

### 4.3 梯度裁剪

```python
import torch.nn.utils as utils

model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters())

# 训练步骤
x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

output = model(x)
loss = nn.CrossEntropyLoss()(output, y)

optimizer.zero_grad()
loss.backward()

# 梯度裁剪（防止梯度爆炸）
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 或按值裁剪
utils.clip_grad_value_(model.parameters(), clip_value=0.5)

optimizer.step()
```

---

## 5. 反向传播实战

### 5.1 手写线性回归

```python
import torch
import matplotlib.pyplot as plt

# 生成数据
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.3  # y = 3x + 2 + noise

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 超参数
lr = 0.1
epochs = 100
losses = []

# 训练
for epoch in range(epochs):
    # 前向传播
    y_pred = X * w + b

    # 计算损失
    loss = ((y_pred - y) ** 2).mean()
    losses.append(loss.item())

    # 反向传播
    loss.backward()

    # 更新参数（手动梯度下降）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n最终: w = {w.item():.4f} (真实: 3), b = {b.item():.4f} (真实: 2)")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(X.numpy(), y.numpy(), alpha=0.5, label='Data')
axes[0].plot(X.numpy(), (X * w + b).detach().numpy(), 'r-', label='Fitted')
axes[0].legend()
axes[0].set_title('Linear Regression')

axes[1].plot(losses)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss')

plt.tight_layout()
plt.show()
```

### 5.2 用 nn.Module 重写

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 数据
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.3

# 模型
model = nn.Linear(1, 1)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练
for epoch in range(100):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# 查看学到的参数
print(f"\n学到的参数:")
print(f"  w = {model.weight.item():.4f} (真实: 3)")
print(f"  b = {model.bias.item():.4f} (真实: 2)")
```

### 5.3 完整训练模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ========== 准备数据 ==========
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 定义模型 ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ========== 损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== 训练循环 ==========
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # 训练模式
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        # 数据转移到设备
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

print("\n训练完成！")
```

---

## 6. 练习题

### 基础练习

1. 计算 f(x) = x³ + 2x² - 5x + 3 在 x=2 处的导数
2. 用 autograd 验证链式法则：f(g(x)) 的导数
3. 实现一个简单的神经网络训练循环

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
import torch

# 1. 计算导数
x = torch.tensor(2.0, requires_grad=True)
f = x**3 + 2*x**2 - 5*x + 3
f.backward()

print(f"f(x) = x³ + 2x² - 5x + 3")
print(f"f(2) = {f.item()}")
print(f"f'(x) = 3x² + 4x - 5")
print(f"f'(2) = {x.grad.item()}")  # 3*4 + 4*2 - 5 = 15

# 验证
manual = 3 * (2**2) + 4 * 2 - 5
print(f"手动计算: {manual}")

# 2. 链式法则验证
# f(x) = sin(x), g(x) = x^2
# h(x) = f(g(x)) = sin(x^2)
# h'(x) = f'(g(x)) * g'(x) = cos(x^2) * 2x

x = torch.tensor(1.0, requires_grad=True)
h = torch.sin(x ** 2)
h.backward()

print(f"\nh(x) = sin(x²)")
print(f"h'(x) = cos(x²) * 2x")
print(f"h'(1) = {x.grad.item():.6f}")

# 验证
import math
manual = math.cos(1**2) * 2 * 1
print(f"手动计算: {manual:.6f}")

# 3. 简单神经网络训练
import torch.nn as nn

# 数据
X = torch.randn(100, 5)
y = (X.sum(dim=1) > 0).long()  # 简单二分类

# 模型
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(50):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        acc = (output.argmax(1) == y).float().mean()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Acc = {acc.item():.2%}")
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [04-nn.Module与模型构建.md](./04-nn.Module与模型构建.md)

