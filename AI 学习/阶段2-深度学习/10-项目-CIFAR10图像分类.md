# 🖼️ 项目：CIFAR-10 图像分类

> 完整的图像分类项目：从数据加载到模型训练、评估和可视化

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据准备](#2-数据准备)
3. [自定义 CNN](#3-自定义-cnn)
4. [迁移学习](#4-迁移学习)
5. [训练与评估](#5-训练与评估)
6. [结果可视化](#6-结果可视化)
7. [优化方向](#7-优化方向)

---

## 1. 项目概述

### 1.1 任务说明

```
数据集：CIFAR-10
- 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 训练集：50,000 张 32x32 彩色图像
- 测试集：10,000 张

目标：训练模型准确分类这些图像

方案：
1. 自定义小型 CNN
2. 预训练 ResNet 迁移学习
```

### 1.2 项目结构

```
cifar10_project/
├── data/              # 数据目录
├── models/            # 模型定义
├── utils/             # 工具函数
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本
└── checkpoints/       # 保存的模型
```

---

## 2. 数据准备

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ========== 数据增强 ==========
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# ========== 加载数据 ==========
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

# 划分训练集和验证集
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 注意：验证集应该用 test_transform
# 这里简化处理，实际项目中建议用 Subset 重新封装

# ========== DataLoader ==========
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# ========== 可视化样本 ==========
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img, title=None):
    """显示图像（反归一化）"""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)

# 显示一批样本
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    imshow(images[i], title=classes[labels[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")
print(f"测试集: {len(test_dataset)}")
```

---

## 3. 自定义 CNN

```python
class CIFAR10Net(nn.Module):
    """自定义 CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积块 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Dropout(0.25)
        )

        # 卷积块 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            nn.Dropout(0.25)
        )

        # 卷积块 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
            nn.Dropout(0.25)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# 创建模型
model = CIFAR10Net()

# 查看模型结构
print(model)

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 测试前向传播
x = torch.randn(2, 3, 32, 32)
y = model(x)
print(f"输出形状: {y.shape}")  # [2, 10]
```

---

## 4. 迁移学习

```python
import torchvision.models as models

def create_resnet_model(num_classes=10, pretrained=True):
    """创建预训练 ResNet 模型"""

    # 加载预训练 ResNet-18
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # 修改第一个卷积层（适应 32x32 输入）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除 maxpool

    # 修改最后一层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def create_vit_model(num_classes=10):
    """创建 Vision Transformer（需要调整输入大小）"""
    # ViT 通常需要更大的输入（224x224）
    # 这里需要在数据预处理中 resize

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model

# 创建 ResNet 模型
resnet_model = create_resnet_model(num_classes=10, pretrained=True)

# 冻结早期层（可选）
def freeze_early_layers(model, num_layers_to_freeze=6):
    """冻结前几层"""
    layers = list(model.children())
    for layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    # 打印可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# freeze_early_layers(resnet_model, 6)
```

---

## 5. 训练与评估

```python
import time
from tqdm import tqdm

# ========== 设置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 选择模型
model = CIFAR10Net()  # 或 create_resnet_model()
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ========== 训练函数 ==========
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total*labels.size(0):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), correct / total

# ========== 训练循环 ==========
num_epochs = 50
best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\n开始训练，共 {num_epochs} 个 epoch")
print("=" * 60)

for epoch in range(num_epochs):
    start_time = time.time()

    # 训练
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # 验证
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # 更新学习率
    scheduler.step()

    # 记录
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ 保存最佳模型，验证准确率: {val_acc:.4f}")

    elapsed = time.time() - start_time

    print(f"Epoch {epoch+1:3d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")

print("=" * 60)
print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")

# ========== 测试 ==========
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"测试集准确率: {test_acc:.4f}")
```

---

## 6. 结果可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ========== 训练曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲线
axes[0].plot(train_losses, label='Train', linewidth=2)
axes[0].plot(val_losses, label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy 曲线
axes[1].plot(train_accs, label='Train', linewidth=2)
axes[1].plot(val_accs, label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

# ========== 混淆矩阵 ==========
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

predictions, labels = get_predictions(model, test_loader, device)

# 混淆矩阵
cm = confusion_matrix(labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# 分类报告
print("\n分类报告:")
print(classification_report(labels, predictions, target_names=classes))

# ========== 可视化预测结果 ==========
def visualize_predictions(model, loader, device, num_images=16):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:num_images], labels[:num_images]

    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        # 反归一化显示图像
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        pred_label = classes[preds[i]]
        true_label = classes[labels[i]]
        confidence = probs[i][preds[i]].item()

        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2f}',
                     color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()

visualize_predictions(model, test_loader, device)

# ========== 错误分析 ==========
def analyze_errors(model, loader, device, num_errors=16):
    """分析分类错误的样本"""
    model.eval()
    errors = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    errors.append({
                        'image': images[i],
                        'true': labels[i].item(),
                        'pred': preds[i].item(),
                        'confidence': probs[i][preds[i]].item()
                    })

                    if len(errors) >= num_errors:
                        break

            if len(errors) >= num_errors:
                break

    # 可视化错误样本
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(errors):
            error = errors[i]
            img = error['image'].numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.set_title(f"True: {classes[error['true']]}\n"
                        f"Pred: {classes[error['pred']]}\n"
                        f"Conf: {error['confidence']:.2f}",
                        color='red', fontsize=10)
        ax.axis('off')

    plt.suptitle('Classification Errors', fontsize=14)
    plt.tight_layout()
    plt.savefig('errors.png', dpi=150)
    plt.show()

analyze_errors(model, test_loader, device)
```

---

## 7. 优化方向

### 7.1 提升准确率

```python
# 1. 更强的数据增强
advanced_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5)
])

# 2. Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 3. MixUp
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 4. 更深的模型
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 5. 集成学习
def ensemble_predict(models, x):
    outputs = [model(x) for model in models]
    return torch.stack(outputs).mean(dim=0)
```

### 7.2 期望效果

```
模型                    CIFAR-10 测试准确率
--------------------------------------------
自定义小 CNN            ~85%
ResNet-18 (预训练)      ~92%
ResNet-50 (预训练)      ~94%
+ 高级数据增强          ~95%
+ MixUp/CutMix         ~96%
```

### 7.3 完整训练脚本

```python
#!/usr/bin/env python3
"""CIFAR-10 训练脚本"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # 模型
    if args.model == 'custom':
        model = CIFAR10Net()
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)

    model = model.to(device)

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = correct / total
        print(f'Epoch {epoch+1}/{args.epochs}: Test Acc = {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['custom', 'resnet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
```

---

## ➡️ 下一步

完成图像分类项目后，继续学习 [11-项目-文本情感分析.md](./11-项目-文本情感分析.md)

