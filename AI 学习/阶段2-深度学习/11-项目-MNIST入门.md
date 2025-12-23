# 🔢 11 - 项目：MNIST 手写数字识别

> 深度学习的 "Hello World"，入门首选项目

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据准备](#2-数据准备)
3. [MLP 模型](#3-mlp-模型)
4. [CNN 模型](#4-cnn-模型)
5. [训练与评估](#5-训练与评估)
6. [结果分析](#6-结果分析)
7. [扩展任务](#7-扩展任务)

---

## 1. 项目概述

### 1.1 任务说明

```
数据集：MNIST 手写数字
├── 训练集：60,000 张 28x28 灰度图像
├── 测试集：10,000 张
├── 类别：0-9 共 10 个数字
└── 难度：⭐（入门级）

目标：识别手写数字图像属于哪个类别

方案：
1. MLP（全连接网络）
2. 简单 CNN
```

### 1.2 为什么选择 MNIST

| 特点 | 说明 |
|------|------|
| 简单 | 图像小（28x28），类别少（10 类） |
| 快速 | CPU 也能快速训练 |
| 经典 | 深度学习入门必做项目 |
| 易调试 | 容易达到 99% 准确率 |

---

## 2. 数据准备

### 2.1 完整代码

```python
"""
MNIST 手写数字识别项目
目标：熟悉 PyTorch 完整训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 2. 数据准备
# ============================================================
print("\n" + "=" * 60)
print("1. 数据准备")
print("=" * 60)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 下载并加载数据
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"图像尺寸: {train_dataset[0][0].shape}")
print(f"类别数: {len(train_dataset.classes)}")

# 可视化样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flatten()):
    image, label = train_dataset[idx]
    ax.imshow(image.squeeze().numpy(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.suptitle('MNIST Samples', fontsize=14)
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150)
plt.show()

# 类别分布
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
plt.figure(figsize=(10, 5))
plt.hist(labels, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Digit')
plt.ylabel('Count')
plt.title('Class Distribution in Training Set')
plt.xticks(range(10))
plt.grid(True, alpha=0.3)
plt.savefig('mnist_distribution.png', dpi=150)
plt.show()
```

---

## 3. MLP 模型

### 3.1 模型定义

```python
# ============================================================
# 3. MLP 模型
# ============================================================
print("\n" + "=" * 60)
print("2. MLP 模型")
print("=" * 60)

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)  # [B, 1, 28, 28] -> [B, 784]
        return self.layers(x)

# 创建模型
mlp_model = MLP()
print(mlp_model)

# 参数统计
total_params = sum(p.numel() for p in mlp_model.parameters())
print(f"\nMLP 参数量: {total_params:,}")

# 测试前向传播
sample_input = torch.randn(2, 1, 28, 28)
sample_output = mlp_model(sample_input)
print(f"输入形状: {sample_input.shape}")
print(f"输出形状: {sample_output.shape}")
```

---

## 4. CNN 模型

### 4.1 模型定义

```python
# ============================================================
# 4. CNN 模型
# ============================================================
print("\n" + "=" * 60)
print("3. CNN 模型")
print("=" * 60)

class SimpleCNN(nn.Module):
    """简单的 CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积：1 -> 32 通道
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            # 第二层卷积：32 -> 64 通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 创建模型
cnn_model = SimpleCNN()
print(cnn_model)

# 参数统计
total_params_cnn = sum(p.numel() for p in cnn_model.parameters())
print(f"\nCNN 参数量: {total_params_cnn:,}")

# 测试前向传播
sample_output_cnn = cnn_model(sample_input)
print(f"输入形状: {sample_input.shape}")
print(f"输出形状: {sample_output_cnn.shape}")
```

---

## 5. 训练与评估

### 5.1 训练函数

```python
# ============================================================
# 5. 训练与评估
# ============================================================
print("\n" + "=" * 60)
print("4. 训练与评估")
print("=" * 60)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, num_epochs, device, model_name="model"):
    """完整训练流程"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    print(f"\n开始训练 {model_name}...")
    print("-" * 50)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }
```

### 5.2 训练两个模型

```python
# 训练 MLP
mlp_model = MLP().to(device)
mlp_results = train_model(mlp_model, train_loader, test_loader, num_epochs=10, device=device, model_name="MLP")

# 训练 CNN
cnn_model = SimpleCNN().to(device)
cnn_results = train_model(cnn_model, train_loader, test_loader, num_epochs=10, device=device, model_name="CNN")
```

---

## 6. 结果分析

### 6.1 训练曲线

```python
# ============================================================
# 6. 结果分析
# ============================================================
print("\n" + "=" * 60)
print("5. 结果分析")
print("=" * 60)

# 绘制训练曲线
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MLP Loss
axes[0, 0].plot(mlp_results['train_losses'], label='Train', linewidth=2)
axes[0, 0].plot(mlp_results['test_losses'], label='Test', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('MLP - Loss Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# MLP Accuracy
axes[0, 1].plot(mlp_results['train_accs'], label='Train', linewidth=2)
axes[0, 1].plot(mlp_results['test_accs'], label='Test', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('MLP - Accuracy Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)

# CNN Loss
axes[1, 0].plot(cnn_results['train_losses'], label='Train', linewidth=2)
axes[1, 0].plot(cnn_results['test_losses'], label='Test', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('CNN - Loss Curves')
axes[1, 0].legend()
axes[1, 0].grid(True)

# CNN Accuracy
axes[1, 1].plot(cnn_results['train_accs'], label='Train', linewidth=2)
axes[1, 1].plot(cnn_results['test_accs'], label='Test', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('CNN - Accuracy Curves')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('mnist_training_curves.png', dpi=150)
plt.show()

# 模型对比
print("\n模型对比:")
print("-" * 40)
print(f"MLP 最终测试准确率: {mlp_results['test_accs'][-1]:.4f}")
print(f"CNN 最终测试准确率: {cnn_results['test_accs'][-1]:.4f}")
```

### 6.2 混淆矩阵

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(model, test_loader, device, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()

    print(f"\n{title} - 分类报告:")
    print(classification_report(all_targets, all_preds))

# 绘制 CNN 混淆矩阵
plot_confusion_matrix(cnn_model, test_loader, device, "CNN Confusion Matrix")
```

### 6.3 错误样本分析

```python
def analyze_errors(model, test_loader, device, num_samples=10):
    """分析错误预测的样本"""
    model.eval()
    errors = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            # 找出错误预测
            wrong_mask = pred != target
            if wrong_mask.any():
                wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
                for idx in wrong_indices:
                    errors.append({
                        'image': data[idx].cpu(),
                        'true_label': target[idx].item(),
                        'pred_label': pred[idx].item(),
                        'confidence': F.softmax(output[idx], dim=0).max().item()
                    })
                    if len(errors) >= num_samples:
                        break
            if len(errors) >= num_samples:
                break

    # 可视化错误样本
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(errors):
            err = errors[idx]
            ax.imshow(err['image'].squeeze().numpy(), cmap='gray')
            ax.set_title(f"True: {err['true_label']}, Pred: {err['pred_label']}\n"
                        f"Conf: {err['confidence']:.2f}", fontsize=10)
        ax.axis('off')

    plt.suptitle('Misclassified Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig('mnist_errors.png', dpi=150)
    plt.show()

# 分析错误样本
analyze_errors(cnn_model, test_loader, device)
```

### 6.4 特征可视化

```python
def visualize_conv_features(model, test_loader, device):
    """可视化卷积层的特征图"""
    model.eval()

    # 获取一张图像
    data, target = next(iter(test_loader))
    image = data[0:1].to(device)
    label = target[0].item()

    # 提取特征
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # 注册钩子
    model.conv_layers[0].register_forward_hook(hook_fn('conv1'))
    model.conv_layers[4].register_forward_hook(hook_fn('conv2'))

    # 前向传播
    with torch.no_grad():
        _ = model(image)

    # 可视化
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))

    # 原图
    axes[0, 0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title(f'Original (Label: {label})')
    axes[0, 0].axis('off')
    for i in range(1, 8):
        axes[0, i].axis('off')

    # Conv1 特征图
    conv1_feat = activations['conv1'].squeeze().cpu().numpy()
    for i in range(8):
        axes[1, i].imshow(conv1_feat[i], cmap='viridis')
        axes[1, i].set_title(f'Conv1-{i}', fontsize=8)
        axes[1, i].axis('off')

    # Conv2 特征图
    conv2_feat = activations['conv2'].squeeze().cpu().numpy()
    for i in range(8):
        axes[2, i].imshow(conv2_feat[i], cmap='viridis')
        axes[2, i].set_title(f'Conv2-{i}', fontsize=8)
        axes[2, i].axis('off')

    plt.suptitle('CNN Feature Maps', fontsize=14)
    plt.tight_layout()
    plt.savefig('mnist_features.png', dpi=150)
    plt.show()

visualize_conv_features(cnn_model, test_loader, device)
```

---

## 7. 扩展任务

### 7.1 模型保存与加载

```python
# 保存模型
torch.save(cnn_model.state_dict(), 'mnist_cnn_model.pth')
print("模型已保存到 mnist_cnn_model.pth")

# 加载模型
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('mnist_cnn_model.pth'))
loaded_model.to(device)
loaded_model.eval()

# 验证
test_loss, test_acc = evaluate(loaded_model, test_loader, nn.CrossEntropyLoss(), device)
print(f"加载模型的测试准确率: {test_acc:.4f}")
```

### 7.2 预测新图像

```python
def predict_digit(model, image, device):
    """预测单张图像"""
    model.eval()

    # 预处理
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    # 归一化
    image = (image - 0.1307) / 0.3081

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    return pred, confidence, probs.squeeze().cpu().numpy()

# 测试
sample_image = test_dataset[42][0]
pred, conf, probs = predict_digit(cnn_model, sample_image, device)

print(f"预测结果: {pred}")
print(f"置信度: {conf:.4f}")
print(f"各类别概率: {probs.round(3)}")

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
plt.title(f'Prediction: {pred} (Confidence: {conf:.2%})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(10), probs)
plt.xlabel('Digit')
plt.ylabel('Probability')
plt.title('Class Probabilities')
plt.xticks(range(10))

plt.tight_layout()
plt.show()
```

### 7.3 进阶挑战

```python
"""
进阶任务清单：

1. 数据增强
   - 随机旋转（±15度）
   - 随机平移
   - 弹性变形

2. 更深的网络
   - 增加卷积层
   - 使用残差连接
   - 尝试 3x3 卷积堆叠

3. 正则化技巧
   - 增加 Dropout
   - 使用 Label Smoothing
   - 添加 weight_decay

4. 学习率调度
   - StepLR
   - CosineAnnealingLR
   - OneCycleLR

5. 目标：达到 99.5%+ 准确率
"""

# 进阶 CNN 示例
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 训练进阶模型
advanced_model = AdvancedCNN().to(device)
print(f"进阶 CNN 参数量: {sum(p.numel() for p in advanced_model.parameters()):,}")
```

---

## 项目总结

```
🎯 本项目完成的任务：

1. ✅ 加载 MNIST 数据集
2. ✅ 实现 MLP 和 CNN 两种模型
3. ✅ 完成训练和评估流程
4. ✅ 可视化训练曲线和混淆矩阵
5. ✅ 分析错误样本
6. ✅ 可视化 CNN 特征图
7. ✅ 模型保存与加载

📊 典型结果：
- MLP：~97-98% 准确率
- 简单 CNN：~99% 准确率
- 进阶 CNN：~99.5% 准确率

📝 学到的知识点：
- PyTorch 完整训练流程
- MLP vs CNN 的区别
- BatchNorm 和 Dropout 的使用
- 模型评估和错误分析
```

---

## ➡️ 下一步

完成本入门项目后，继续挑战 [12-项目-CIFAR10图像分类.md](./12-项目-CIFAR10图像分类.md)

