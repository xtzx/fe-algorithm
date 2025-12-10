# 🏗️ 07 - ResNet 与 Vision Transformer

> 残差连接让网络可以更深，ViT 证明图像也能用 Transformer

---

## 目录

1. [残差连接原理](#1-残差连接原理)
2. [ResNet 实现](#2-resnet-实现)
3. [Vision Transformer (ViT)](#3-vision-transformer-vit)
4. [迁移学习](#4-迁移学习)
5. [练习题](#5-练习题)

---

## 1. 残差连接原理

### 1.1 为什么需要残差连接？

```
问题：网络越深，训练越难
- 梯度消失：梯度在反向传播中逐层衰减
- 退化问题：更深的网络反而准确率更低

直觉：如果新增的层是"多余的"，网络至少应该能学到恒等映射
但实际上，让网络学习 H(x) = x 很难

残差连接的解决方案：
- 让网络学习残差 F(x) = H(x) - x
- 输出变成 H(x) = F(x) + x
- 学习"恒等"就是让 F(x) = 0，这更容易！
```

### 1.2 残差块

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet 基本残差块（用于 ResNet-18/34）"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # 第一个卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 用于调整残差维度

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 如果维度不匹配，需要调整
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接！
        out = self.relu(out)

        return out

# 测试
block = BasicBlock(64, 64)
x = torch.randn(2, 64, 56, 56)
y = block(x)
print(f"BasicBlock: {x.shape} → {y.shape}")
```

### 1.3 瓶颈块

```python
class Bottleneck(nn.Module):
    """ResNet 瓶颈块（用于 ResNet-50/101/152）

    使用 1x1 卷积降维和升维，减少计算量
    结构：1x1(降维) → 3x3 → 1x1(升维)
    """
    expansion = 4  # 输出通道 = out_channels * expansion

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # 1x1 卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 测试
downsample = nn.Sequential(
    nn.Conv2d(64, 256, 1, bias=False),
    nn.BatchNorm2d(256)
)
block = Bottleneck(64, 64, downsample=downsample)
x = torch.randn(2, 64, 56, 56)
y = block(x)
print(f"Bottleneck: {x.shape} → {y.shape}")  # [2, 64, 56, 56] → [2, 256, 56, 56]
```

---

## 2. ResNet 实现

### 2.1 完整 ResNet

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_channels = 64

        # Stem: 初始卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 Stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化
        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None

        # 如果维度不匹配，创建 downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块可能需要 downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # 剩余块
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.conv1(x)   # 224 → 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 112 → 56

        # 4 Stages
        x = self.layer1(x)  # 56 → 56
        x = self.layer2(x)  # 56 → 28
        x = self.layer3(x)  # 28 → 14
        x = self.layer4(x)  # 14 → 7

        # 分类
        x = self.avgpool(x) # 7 → 1
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 不同深度的 ResNet
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

# 测试
model = resnet18(num_classes=10)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(f"ResNet-18 输出: {y.shape}")  # [2, 10]

total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")  # ~11M
```

### 2.2 使用预训练 ResNet

```python
import torchvision.models as models

# 加载预训练权重
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# 或旧版写法
# resnet = models.resnet50(pretrained=True)

# 修改最后一层
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# 冻结除最后一层外的所有层
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.fc.parameters():
    param.requires_grad = True

# 查看可训练参数
trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"可训练参数: {trainable:,}")
```

---

## 3. Vision Transformer (ViT)

### 3.1 ViT 核心思想

```
传统：CNN 用卷积提取局部特征
ViT：把图像分成 patch，当作序列用 Transformer 处理

1. 图像切分为 patch
   224x224 图像 → 14x14 个 16x16 的 patch → 196 个 patch

2. 每个 patch 展平并映射到 embedding
   16x16x3 = 768 → Linear → D 维向量

3. 加上位置编码和 [CLS] token

4. 送入 Transformer Encoder

5. 用 [CLS] token 的输出做分类
```

### 3.2 Patch Embedding

```python
class PatchEmbedding(nn.Module):
    """将图像分成 patch 并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 用卷积实现 patch 分割 + 线性映射
        self.proj = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] → [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # [B, embed_dim, num_patches_h, num_patches_w] → [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x

# 测试
patch_embed = PatchEmbedding()
x = torch.randn(2, 3, 224, 224)
patches = patch_embed(x)
print(f"Patch Embedding: {x.shape} → {patches.shape}")
# [2, 3, 224, 224] → [2, 196, 768]
```

### 3.3 简化版 ViT

```python
class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)

        # 取 [CLS] token 的输出做分类
        x = self.norm(x[:, 0])  # [B, embed_dim]
        x = self.head(x)        # [B, num_classes]

        return x

# 测试
model = ViT(
    img_size=224,
    patch_size=16,
    num_classes=10,
    embed_dim=384,  # ViT-Small
    num_heads=6,
    num_layers=6,
)

x = torch.randn(2, 3, 224, 224)
y = model(x)
print(f"ViT 输出: {y.shape}")  # [2, 10]

total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")
```

### 3.4 使用预训练 ViT

```python
import torchvision.models as models

# 加载预训练 ViT
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# 修改分类头
vit.heads.head = nn.Linear(vit.heads.head.in_features, 10)

# 测试
x = torch.randn(2, 3, 224, 224)
y = vit(x)
print(f"预训练 ViT 输出: {y.shape}")

# 或使用 timm 库（更多模型选择）
# pip install timm
import timm

# 列出可用的 ViT 模型
# print(timm.list_models('vit*'))

# 加载预训练模型
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
```

---

## 4. 迁移学习

### 4.1 迁移学习策略

```python
def create_transfer_model(model_name='resnet50', num_classes=10, freeze_backbone=True):
    """创建迁移学习模型"""

    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.parameters():
                param.requires_grad = True

    return model

# 使用
model = create_transfer_model('resnet50', num_classes=10, freeze_backbone=True)
```

### 4.2 微调策略

```python
class FineTuner:
    """分阶段微调"""

    def __init__(self, model, num_stages=3):
        self.model = model
        self.num_stages = num_stages

        # 获取所有参数组
        if hasattr(model, 'layer1'):  # ResNet
            self.param_groups = [
                model.conv1, model.bn1,
                model.layer1, model.layer2,
                model.layer3, model.layer4,
                model.fc
            ]
        else:  # ViT
            # 简化处理
            self.param_groups = [model]

    def unfreeze_stage(self, stage):
        """解冻特定阶段的参数"""
        # 从后往前解冻
        start_idx = len(self.param_groups) - stage - 1
        start_idx = max(0, start_idx)

        for i, group in enumerate(self.param_groups):
            for param in group.parameters():
                param.requires_grad = (i >= start_idx)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Stage {stage}: 可训练参数 {trainable:,}")

# 使用示例
# Stage 0: 只训练分类头
# Stage 1: 训练最后几层 + 分类头
# Stage 2: 训练全部
```

### 4.3 完整迁移学习流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ========== 数据准备 ==========
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 使用 CIFAR-10 演示
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR10('./data', train=False, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ========== 模型 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)

# 冻结 backbone
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

model = model.to(device)

# ========== 训练 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in loader:
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

    return running_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total

# 训练
for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)
    print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')
```

---

## 5. 练习题

### 基础练习

1. 手写一个 BasicBlock，理解残差连接
2. 用预训练 ResNet 做 CIFAR-10 分类，对比从头训练
3. 理解 ViT 的 patch embedding 过程

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 手写 BasicBlock
class MyBasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # 保存输入

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity  # 残差连接
        out = torch.relu(out)

        return out

# 验证残差连接
block = MyBasicBlock(64)
x = torch.randn(2, 64, 32, 32)
y = block(x)
print(f"输入输出形状: {x.shape} → {y.shape}")


# 2. 预训练 vs 从头训练对比
# 预训练模型通常在少量数据上就能达到很高准确率
# 从头训练需要更多数据和 epoch


# 3. Patch Embedding 理解
def manual_patch_embed(img, patch_size=16, embed_dim=768):
    """手动实现 patch embedding"""
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0

    # 分割成 patch
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # 重排: [B, C, H, W] → [B, num_patches, patch_size*patch_size*C]
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, num_h, num_w, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [B, num_h, num_w, C, patch_size, patch_size]
    patches = patches.view(B, num_patches_h * num_patches_w, -1)
    # [B, num_patches, C*patch_size*patch_size]

    # 线性投影
    proj = nn.Linear(C * patch_size * patch_size, embed_dim)
    embedded = proj(patches)

    return embedded

img = torch.randn(2, 3, 224, 224)
embedded = manual_patch_embed(img)
print(f"手动 Patch Embedding: {embedded.shape}")  # [2, 196, 768]
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [08-循环神经网络RNN.md](./08-循环神经网络RNN.md)

