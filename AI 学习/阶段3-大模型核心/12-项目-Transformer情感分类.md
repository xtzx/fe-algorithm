# 🎭 12 - 项目：Transformer 情感分类

> 入门级项目：从零实现一个 Transformer 分类器，理解核心组件

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据准备](#2-数据准备)
3. [从零实现 Transformer](#3-从零实现-transformer)
4. [训练与评估](#4-训练与评估)
5. [结果分析](#5-结果分析)
6. [扩展任务](#6-扩展任务)

---

## 1. 项目概述

### 1.1 任务说明

```
任务：文本情感分类（正面/负面）
数据集：IMDB 电影评论
难度：⭐⭐（入门级）

目标：
1. 从零实现 Transformer Encoder
2. 理解 Self-Attention 的实际应用
3. 完成完整的训练和评估流程

这是进入 BERT/GPT 前的必备练习！
```

### 1.2 项目结构

```
transformer_sentiment/
├── model.py          # Transformer 实现
├── dataset.py        # 数据处理
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
└── checkpoints/      # 模型保存
```

---

## 2. 数据准备

```python
"""
Transformer 情感分类项目
从零实现 Transformer Encoder，完成文本分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 2. 数据准备
# ============================================================
print("\n" + "=" * 60)
print("1. 数据准备")
print("=" * 60)

# 使用简单的示例数据（实际项目可使用 IMDB）
positive_samples = [
    "I love this movie it is great",
    "This film is wonderful and amazing",
    "Excellent performance by the actors",
    "A masterpiece of cinema",
    "Highly recommended great story",
    "The best movie I have ever seen",
    "Absolutely brilliant and moving",
    "Perfect in every way loved it",
    "Outstanding film with great acting",
    "A beautiful and touching story",
] * 100  # 扩展数据

negative_samples = [
    "This movie is terrible and boring",
    "Worst film I have ever watched",
    "Complete waste of time and money",
    "Awful acting and bad script",
    "I hated every minute of it",
    "Disappointing and poorly made",
    "Do not watch this garbage",
    "Terrible plot and bad acting",
    "A total disaster of a movie",
    "Boring and uninteresting story",
] * 100

# 合并数据
texts = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

# 打乱数据
indices = np.random.permutation(len(texts))
texts = [texts[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"总样本数: {len(texts)}")
print(f"正面样本: {sum(labels)}")
print(f"负面样本: {len(labels) - sum(labels)}")

# ============================================================
# 构建词表
# ============================================================
def tokenize(text):
    """简单分词"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

# 构建词表
word_freq = Counter()
for text in texts:
    word_freq.update(tokenize(text))

# 特殊 token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"

# 创建词表
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2}
for word, _ in word_freq.most_common(10000):
    if word not in vocab:
        vocab[word] = len(vocab)

print(f"词表大小: {len(vocab)}")

# ============================================================
# 数据集类
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 分词
        tokens = [CLS_TOKEN] + tokenize(text)[:self.max_len - 1]

        # 转换为 ID
        token_ids = [self.vocab.get(t, self.vocab[UNK_TOKEN]) for t in tokens]

        # 填充
        padding_len = self.max_len - len(token_ids)
        token_ids = token_ids + [self.vocab[PAD_TOKEN]] * padding_len

        # 注意力掩码
        attention_mask = [1] * (self.max_len - padding_len) + [0] * padding_len

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 划分数据集
train_size = int(0.8 * len(texts))
train_texts, val_texts = texts[:train_size], texts[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

train_dataset = SentimentDataset(train_texts, train_labels, vocab)
val_dataset = SentimentDataset(val_texts, val_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")

# 查看一个样本
sample = train_dataset[0]
print(f"\n样本示例:")
print(f"  input_ids: {sample['input_ids'][:10]}...")
print(f"  attention_mask: {sample['attention_mask'][:10]}...")
print(f"  label: {sample['label']}")
```

---

## 3. 从零实现 Transformer

### 3.1 位置编码

```python
# ============================================================
# 3. 从零实现 Transformer
# ============================================================
print("\n" + "=" * 60)
print("2. 从零实现 Transformer")
print("=" * 60)

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]
```

### 3.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape

        # 计算 Q, K, V
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        out = torch.matmul(attn_weights, V)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out, attn_weights
```

### 3.3 Feed Forward Network

```python
class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # 使用 GELU 激活
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 3.4 Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder 层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Self-Attention + 残差连接
        attn_out, attn_weights = self.attn(self.norm1(x), attention_mask)
        x = x + self.dropout1(attn_out)

        # FFN + 残差连接
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x, attn_weights
```

### 3.5 完整的 Transformer 分类器

```python
class TransformerClassifier(nn.Module):
    """Transformer 文本分类器"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=256, num_classes=2, max_len=128, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 分类头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # 词嵌入
        x = self.embedding(input_ids)  # [B, T, D]
        x = x * (self.d_model ** 0.5)  # 缩放

        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer 层
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask)
            all_attn_weights.append(attn_weights)

        x = self.norm(x)

        # 取 [CLS] token 的表示
        cls_output = x[:, 0, :]  # [B, D]

        # 分类
        logits = self.classifier(cls_output)  # [B, num_classes]

        return logits, all_attn_weights

# 创建模型
model = TransformerClassifier(
    vocab_size=len(vocab),
    d_model=128,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    num_classes=2,
    max_len=32,
    dropout=0.1
).to(device)

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")
print(model)

# 测试前向传播
sample_batch = next(iter(train_loader))
sample_input = sample_batch['input_ids'].to(device)
sample_mask = sample_batch['attention_mask'].to(device)
sample_output, _ = model(sample_input, sample_mask)
print(f"\n输入形状: {sample_input.shape}")
print(f"输出形状: {sample_output.shape}")
```

---

## 4. 训练与评估

```python
# ============================================================
# 4. 训练与评估
# ============================================================
print("\n" + "=" * 60)
print("3. 训练与评估")
print("=" * 60)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), correct / total


# 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 训练循环
num_epochs = 20
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\n开始训练...")
print("-" * 50)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:2d}/{num_epochs}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

print(f"\n最终验证准确率: {val_accs[-1]:.4f}")
```

---

## 5. 结果分析

### 5.1 训练曲线

```python
# ============================================================
# 5. 结果分析
# ============================================================
print("\n" + "=" * 60)
print("4. 结果分析")
print("=" * 60)

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲线
axes[0].plot(train_losses, label='Train', linewidth=2)
axes[0].plot(val_losses, label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True)

# Accuracy 曲线
axes[1].plot(train_accs, label='Train', linewidth=2)
axes[1].plot(val_accs, label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('transformer_training_curves.png', dpi=150)
plt.show()
```

### 5.2 注意力可视化

```python
def visualize_attention(model, text, vocab, device):
    """可视化注意力权重"""
    model.eval()

    # 分词和编码
    tokens = [CLS_TOKEN] + tokenize(text)[:31]
    token_ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    padding_len = 32 - len(token_ids)
    token_ids = token_ids + [vocab[PAD_TOKEN]] * padding_len
    attention_mask = [1] * (32 - padding_len) + [0] * padding_len

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    # 前向传播
    with torch.no_grad():
        logits, attn_weights = model(input_ids, mask)

    # 预测结果
    pred = logits.argmax(dim=1).item()
    prob = F.softmax(logits, dim=1)[0, pred].item()
    sentiment = "正面" if pred == 1 else "负面"

    print(f"文本: {text}")
    print(f"预测: {sentiment} (置信度: {prob:.2%})")

    # 可视化最后一层第一个头的注意力
    attn = attn_weights[-1][0, 0].cpu().numpy()  # [T, T]

    # 只显示有效 token
    valid_len = len(tokens)
    attn = attn[:valid_len, :valid_len]

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(valid_len), tokens, rotation=45, ha='right')
    plt.yticks(range(valid_len), tokens)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title(f'Attention Weights (Pred: {sentiment})')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()

# 可视化示例
visualize_attention(model, "I love this amazing movie", vocab, device)
visualize_attention(model, "This movie is terrible", vocab, device)
```

### 5.3 预测测试

```python
def predict(model, text, vocab, device):
    """预测单条文本"""
    model.eval()

    tokens = [CLS_TOKEN] + tokenize(text)[:31]
    token_ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    padding_len = 32 - len(token_ids)
    token_ids = token_ids + [vocab[PAD_TOKEN]] * padding_len
    attention_mask = [1] * (32 - padding_len) + [0] * padding_len

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, _ = model(input_ids, mask)
        probs = F.softmax(logits, dim=1)

    pred = logits.argmax(dim=1).item()
    confidence = probs[0, pred].item()

    return "正面" if pred == 1 else "负面", confidence

# 测试
test_texts = [
    "This is the best movie ever!",
    "Absolutely terrible, waste of time",
    "Pretty good, I enjoyed it",
    "Not great but not bad either",
    "Boring and disappointing film",
]

print("\n预测测试:")
print("-" * 50)
for text in test_texts:
    sentiment, conf = predict(model, text, vocab, device)
    print(f"[{sentiment}] ({conf:.2%}) {text}")
```

---

## 6. 扩展任务

### 6.1 保存和加载模型

```python
# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'config': {
        'vocab_size': len(vocab),
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'num_classes': 2,
        'max_len': 32,
    }
}, 'transformer_sentiment.pth')

print("模型已保存！")

# 加载模型
checkpoint = torch.load('transformer_sentiment.pth')
loaded_model = TransformerClassifier(**checkpoint['config']).to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_vocab = checkpoint['vocab']

print("模型已加载！")
```

### 6.2 进阶挑战

```python
"""
进阶任务清单：

1. 使用真实 IMDB 数据集
   from datasets import load_dataset
   dataset = load_dataset("imdb")

2. 增加模型容量
   - 更多层（n_layers=4）
   - 更大维度（d_model=256）
   - 更多注意力头（n_heads=8）

3. 添加位置嵌入（可学习）
   self.pos_embedding = nn.Embedding(max_len, d_model)

4. 实现更多池化策略
   - Mean Pooling
   - Max Pooling
   - Attention Pooling

5. 添加预训练词嵌入
   - 使用 GloVe 或 FastText
"""
```

---

## 项目总结

```
🎯 本项目完成的任务：

1. ✅ 从零实现 Multi-Head Attention
2. ✅ 从零实现 Transformer Encoder
3. ✅ 完成文本情感分类任务
4. ✅ 可视化注意力权重
5. ✅ 模型保存和加载

📊 典型结果：
- 验证准确率：~90-95%
- 训练时间：几分钟（CPU）

📝 学到的知识点：
- Self-Attention 的计算过程
- Multi-Head Attention 的实现
- Position Encoding 的作用
- LayerNorm + 残差连接
- [CLS] token 用于分类

🔗 与 BERT 的联系：
BERT = 预训练的 Transformer Encoder
本项目 = 从零训练的简化版 Transformer Encoder
```

---

## ➡️ 下一步

完成本入门项目后，继续挑战 [13-项目-BERT文本分类.md](./13-项目-BERT文本分类.md)

