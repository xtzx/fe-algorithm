# 🖼️ 06 - 卷积神经网络 CNN

> CNN 是处理图像的利器，通过卷积操作提取空间特征

---

## 目录

1. [卷积操作](#1-卷积操作)
2. [池化层](#2-池化层)
3. [构建 CNN](#3-构建-cnn)
4. [经典架构演进](#4-经典架构演进)
5. [实战：MNIST 分类](#5-实战mnist-分类)
6. [练习题](#6-练习题)

---

## 1. 卷积操作

### 1.1 卷积的直观理解

```
卷积核在输入上滑动，计算加权和

输入图像 (5x5)          卷积核 (3x3)        输出 (3x3)
┌─────────────┐         ┌───────┐           ┌───────┐
│ 1 2 3 4 5   │         │ 1 0 1 │           │ ? ? ? │
│ 2 3 4 5 6   │    *    │ 0 1 0 │     =     │ ? ? ? │
│ 3 4 5 6 7   │         │ 1 0 1 │           │ ? ? ? │
│ 4 5 6 7 8   │         └───────┘           └───────┘
│ 5 6 7 8 9   │
└─────────────┘

卷积核学习检测特定模式（边缘、纹理、形状等）
```

### 1.2 Conv2d 基础

```python
import torch
import torch.nn as nn

# 卷积层
conv = nn.Conv2d(
    in_channels=3,    # 输入通道数（RGB=3）
    out_channels=64,  # 输出通道数（卷积核数量）
    kernel_size=3,    # 卷积核大小
    stride=1,         # 步长
    padding=1,        # 填充
    bias=True         # 是否有偏置
)

# 输入：[batch, channels, height, width]
x = torch.randn(32, 3, 224, 224)
y = conv(x)
print(f"输入: {x.shape} → 输出: {y.shape}")
# [32, 3, 224, 224] → [32, 64, 224, 224]

# 查看参数
print(f"权重形状: {conv.weight.shape}")  # [out_c, in_c, kH, kW] = [64, 3, 3, 3]
print(f"偏置形状: {conv.bias.shape}")    # [64]
```

### 1.3 输出尺寸计算

```python
# 公式：output_size = (input_size + 2*padding - kernel_size) / stride + 1

def calc_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size + 2*padding - kernel_size) // stride + 1

# 示例
print(calc_output_size(224, kernel_size=3, stride=1, padding=1))  # 224
print(calc_output_size(224, kernel_size=3, stride=2, padding=1))  # 112
print(calc_output_size(224, kernel_size=7, stride=2, padding=3))  # 112

# 常用配置
# 保持尺寸不变：kernel=3, stride=1, padding=1
# 减半尺寸：kernel=3, stride=2, padding=1
# 减半尺寸：kernel=2, stride=2, padding=0 (池化常用)
```

### 1.4 不同卷积类型

```python
# 标准卷积
conv_standard = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# 深度可分离卷积（MobileNet）
# 1. 深度卷积：每个通道单独卷积
conv_depthwise = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
# 2. 逐点卷积：1x1 卷积融合通道
conv_pointwise = nn.Conv2d(64, 128, kernel_size=1)

# 分组卷积
conv_grouped = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)

# 1x1 卷积（通道变换）
conv_1x1 = nn.Conv2d(64, 32, kernel_size=1)

# 空洞卷积（扩大感受野）
conv_dilated = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

# 转置卷积（上采样）
conv_transpose = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# 输入 [B, 64, 14, 14] → 输出 [B, 32, 28, 28]
```

### 1.5 可视化卷积核

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 训练后的卷积核
conv = nn.Conv2d(3, 16, kernel_size=3)

# 可视化第一层卷积核
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < 16:
        # 取出一个卷积核，将 3 通道转为灰度显示
        kernel = conv.weight[i].detach().cpu()
        # 简单处理：取均值
        kernel_gray = kernel.mean(dim=0)
        ax.imshow(kernel_gray, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Kernel {i}')
plt.tight_layout()
plt.show()
```

---

## 2. 池化层

### 2.1 最大池化

```python
# 最大池化：取区域内最大值
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(1, 64, 28, 28)
y = max_pool(x)
print(f"MaxPool: {x.shape} → {y.shape}")  # [1, 64, 28, 28] → [1, 64, 14, 14]

# 可视化
x = torch.tensor([[[[1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.]]]])

pool = nn.MaxPool2d(2, 2)
y = pool(x)
print(f"输入:\n{x[0, 0]}")
print(f"MaxPool 输出:\n{y[0, 0]}")
# [[6, 8],
#  [14, 16]]
```

### 2.2 平均池化

```python
# 平均池化：取区域内平均值
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

x = torch.randn(1, 64, 28, 28)
y = avg_pool(x)
print(f"AvgPool: {x.shape} → {y.shape}")  # [1, 64, 14, 14]
```

### 2.3 全局池化

```python
# 全局平均池化（GAP）：把每个通道压缩成一个数
gap = nn.AdaptiveAvgPool2d(1)

x = torch.randn(32, 512, 7, 7)
y = gap(x)
print(f"GAP: {x.shape} → {y.shape}")  # [32, 512, 7, 7] → [32, 512, 1, 1]

# 等价于
y = x.mean(dim=(2, 3), keepdim=True)

# 自适应池化：输出指定大小
adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 输出固定为 7x7
x = torch.randn(32, 512, 14, 14)
y = adaptive_pool(x)
print(f"Adaptive: {x.shape} → {y.shape}")  # [32, 512, 7, 7]
```

---

## 3. 构建 CNN

### 3.1 基本 CNN 结构

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 特征提取器
        self.features = nn.Sequential(
            # 卷积块 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 → 112

            # 卷积块 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 → 56

            # 卷积块 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 → 28

            # 卷积块 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 测试
model = SimpleCNN(num_classes=10)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(f"输出形状: {y.shape}")  # [2, 10]

# 参数统计
total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total:,}")
```

### 3.2 卷积块封装

```python
def conv_block(in_channels, out_channels, pool=True):
    """卷积块：Conv -> BN -> ReLU (-> Pool)"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, pool=True)   # /2
        self.conv2 = conv_block(64, 128, pool=True)           # /2
        self.conv3 = conv_block(128, 256, pool=True)          # /2
        self.conv4 = conv_block(256, 512, pool=True)          # /2

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

---

## 4. 经典架构演进

### 4.1 LeNet (1998)

```python
class LeNet(nn.Module):
    """第一个成功的 CNN，用于手写数字识别"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)    # 28→24
        self.pool = nn.AvgPool2d(2, 2)     # 24→12
        self.conv2 = nn.Conv2d(6, 16, 5)   # 12→8
        # pool: 8→4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 4.2 VGG (2014)

```python
class VGG16(nn.Module):
    """
    核心思想：使用多个小卷积核（3x3）代替大卷积核
    两个 3x3 卷积的感受野等于一个 5x5，但参数更少
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 4.3 使用预训练模型

```python
import torchvision.models as models

# 加载预训练 VGG16
vgg = models.vgg16(pretrained=True)

# 替换最后一层适应新任务
vgg.classifier[6] = nn.Linear(4096, 10)

# 冻结特征提取器
for param in vgg.features.parameters():
    param.requires_grad = False

# 只训练分类器
optimizer = torch.optim.Adam(vgg.classifier.parameters(), lr=0.001)
```

---

## 5. 实战：MNIST 分类

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========== 数据准备 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ========== 模型 ==========
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 28→26
        x = torch.relu(self.conv2(x))  # 26→24
        x = nn.functional.max_pool2d(x, 2)  # 24→12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # 64*12*12 = 9216
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ========== 训练 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return correct / len(loader.dataset)

# 训练循环
for epoch in range(1, 11):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    test_acc = test(model, test_loader, device)
    print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}')

# 预期：10 epoch 后 Test Acc > 99%
```

---

## 6. 练习题

### 基础练习

1. 计算：输入 32x32，kernel=5, stride=2, padding=2 后的输出尺寸
2. 实现一个 CNN 用于 CIFAR-10 分类
3. 尝试用深度可分离卷积替换标准卷积，比较参数量

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 输出尺寸计算
# (32 + 2*2 - 5) / 2 + 1 = 31/2 + 1 = 15.5 → 15
print("输出尺寸:", (32 + 4 - 5) // 2 + 1)  # 16


# 2. CIFAR-10 CNN
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32→16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16→8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8→4

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 3. 深度可分离卷积对比
# 标准卷积参数：in_c * out_c * k * k = 64 * 128 * 3 * 3 = 73,728
conv_standard = nn.Conv2d(64, 128, 3, padding=1)
print(f"标准卷积参数: {sum(p.numel() for p in conv_standard.parameters())}")

# 深度可分离卷积参数：in_c * k * k + in_c * out_c = 64*9 + 64*128 = 8,768
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, k, padding=k//2, groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

conv_dw = DepthwiseSeparable(64, 128)
print(f"深度可分离参数: {sum(p.numel() for p in conv_dw.parameters())}")
# 参数减少约 8 倍！
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [07-ResNet与ViT.md](./07-ResNet与ViT.md)

