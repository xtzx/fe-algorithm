# 🏛 01 - Transformer 架构

> Transformer 是现代 NLP 和 LLM 的基石，理解它是学习大模型的第一步

---

## 目录

1. [Transformer 全局视图](#1-transformer-全局视图)
2. [Self-Attention](#2-self-attention)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [Feed Forward Network](#4-feed-forward-network)
5. [残差连接与 LayerNorm](#5-残差连接与-layernorm)
6. [Transformer Block](#6-transformer-block)
7. [代码实现](#7-代码实现)
8. [练习题](#8-练习题)

---

## 1. Transformer 全局视图

### 1.1 为什么需要 Transformer？

```
RNN 的问题：
1. 顺序处理，无法并行 → 训练慢
2. 长距离依赖难以学习 → 梯度消失
3. 信息需要"传递"很多步 → 信息丢失

Transformer 的解决方案：
1. Self-Attention 并行计算所有位置
2. 每个位置直接"看"到所有其他位置
3. 不需要信息传递，直接全局交互
```

### 1.2 原始 Transformer 架构

```
原始 Transformer（2017）是 Encoder-Decoder 结构：

        ┌─────────────────┐
输入 →  │    Encoder      │  → 编码表示
        │  (N 个 Block)   │
        └────────┬────────┘
                 │
                 ↓
        ┌─────────────────┐
输出 ←  │    Decoder      │  ← 目标序列
        │  (N 个 Block)   │
        └─────────────────┘

用途：机器翻译（输入英文 → 输出中文）
```

### 1.3 现代变体

```
Encoder-only (BERT)：
- 只有 Encoder
- 双向注意力（看前看后）
- 用途：文本理解、分类、NER

Decoder-only (GPT/LLaMA)：
- 只有 Decoder
- 单向注意力（只看前面）+ 因果掩码
- 用途：文本生成
- 现代 LLM 的主流选择！

Encoder-Decoder (T5)：
- 完整结构
- 用途：翻译、摘要、问答
```

---

## 2. Self-Attention

### 2.1 核心直觉

```
Self-Attention 的本质：
让序列中的每个位置"关注"其他所有位置，学习它们之间的关系

输入序列："The cat sat on the mat"
         [0]  [1] [2] [3] [4]  [5]

对于 "sat"（位置 2）：
- 它需要知道"谁在坐"→ 关注 "cat"
- 它需要知道"坐在哪"→ 关注 "mat"
- 这种关系通过注意力权重学习

输出：每个位置的表示都融合了相关上下文
```

### 2.2 Q、K、V 的直觉

```
三个向量的角色：

Query (Q) = "我在找什么？"
Key (K)   = "我有什么可以被找到？"
Value (V) = "如果被找到，返回什么内容？"

类比：图书馆查询
- Query：你想找的书的特征（科幻、最新的）
- Key：每本书的标签（分类、出版日期）
- Value：书的实际内容

过程：
1. 用 Query 和所有 Key 计算相似度（注意力分数）
2. 相似度高的 Key 对应的 Value 权重大
3. 加权求和得到输出
```

### 2.3 数学公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

分步解释：
1. QK^T：计算 Query 和所有 Key 的相似度
   - 形状：[seq_len, d_k] × [d_k, seq_len] = [seq_len, seq_len]

2. / √d_k：缩放因子，防止点积过大导致 softmax 饱和
   - d_k 是 Key 的维度

3. softmax：转换为概率分布（权重和为 1）

4. × V：用注意力权重加权 Value
   - 形状：[seq_len, seq_len] × [seq_len, d_v] = [seq_len, d_v]
```

### 2.4 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Args:
        Q: [batch, seq_len, d_k]
        K: [batch, seq_len, d_k]
        V: [batch, seq_len, d_v]
        mask: [batch, 1, seq_len] 或 [batch, seq_len, seq_len]

    Returns:
        output: [batch, seq_len, d_v]
        attention_weights: [batch, seq_len, seq_len]
    """
    d_k = Q.size(-1)

    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, seq_len, seq_len]

    # 2. 应用掩码（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# 测试
batch_size = 2
seq_len = 4
d_k = d_v = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, attn = scaled_dot_product_attention(Q, K, V)
print(f"输入 Q: {Q.shape}")
print(f"输出: {output.shape}")
print(f"注意力权重: {attn.shape}")
print(f"注意力权重（第一个样本）:\n{attn[0]}")
```

### 2.5 可视化注意力

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """可视化注意力权重"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.detach().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Self-Attention Weights')
    plt.show()

# 示例
tokens = ["The", "cat", "sat", "on"]
attn_example = torch.softmax(torch.randn(4, 4), dim=-1)
visualize_attention(attn_example, tokens)
```

---

## 3. Multi-Head Attention

### 3.1 为什么需要多头？

```
单头注意力的局限：
- 只能学习一种注意力模式
- 比如只能关注"语法关系"或"语义关系"

多头注意力：
- 并行运行多个注意力头
- 每个头学习不同的模式
- 拼接后投影，融合多种信息

类比：多个专家
- Head 1：专注语法依赖（主谓关系）
- Head 2：专注指代关系（代词指向谁）
- Head 3：专注语义相似性
```

### 3.2 多头注意力结构

```
输入 X
    │
    ├─→ Linear(W_Q^1) ─→ Q^1 ─┐
    ├─→ Linear(W_K^1) ─→ K^1 ─┼─→ Head^1 ─┐
    ├─→ Linear(W_V^1) ─→ V^1 ─┘           │
    │                                      │
    ├─→ Linear(W_Q^2) ─→ Q^2 ─┐            │
    ├─→ Linear(W_K^2) ─→ K^2 ─┼─→ Head^2 ─┼─→ Concat ─→ Linear(W_O) ─→ 输出
    ├─→ Linear(W_V^2) ─→ V^2 ─┘            │
    │                                      │
    └─→ ... (更多头) ...                   ┘

维度变化（假设 d_model=512, num_heads=8）：
- 输入：[batch, seq_len, 512]
- 每个头：d_k = d_v = 512 / 8 = 64
- 每个头输出：[batch, seq_len, 64]
- 拼接后：[batch, seq_len, 512]
- 线性变换后：[batch, seq_len, 512]
```

### 3.3 代码实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 拆分成多头
        # [batch, seq_len, d_model] → [batch, seq_len, num_heads, d_k]
        # → [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 扩展 mask 维度以匹配多头
            mask = mask.unsqueeze(1)  # [batch, 1, ...]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. 加权求和
        context = torch.matmul(attention_weights, V)
        # [batch, num_heads, seq_len, d_k]

        # 5. 合并多头
        # [batch, num_heads, seq_len, d_k] → [batch, seq_len, num_heads, d_k]
        # → [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 6. 输出投影
        output = self.W_o(context)

        return output, attention_weights

# 测试
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)

output, attn = mha(x, x, x)  # Self-Attention: Q=K=V=X
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print(f"注意力权重: {attn.shape}")  # [batch, num_heads, seq_len, seq_len]
```

---

## 4. Feed Forward Network

### 4.1 FFN 的作用

```
注意力层：学习位置之间的关系（交互）
FFN 层：对每个位置独立进行非线性变换（特征提取）

结构：两层 MLP + 激活函数
FFN(x) = Linear_2(Activation(Linear_1(x)))

维度变化：
d_model → d_ff → d_model
512 → 2048 → 512（通常 d_ff = 4 * d_model）
```

### 4.2 代码实现

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = self.linear1(x)      # [batch, seq_len, d_ff]
        x = F.gelu(x)            # 现代 Transformer 常用 GELU
        x = self.dropout(x)
        x = self.linear2(x)      # [batch, seq_len, d_model]
        return x

# 测试
ffn = FeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 10, 512)
y = ffn(x)
print(f"FFN: {x.shape} → {y.shape}")
```

### 4.3 现代变体：SwiGLU

```python
class SwiGLU(nn.Module):
    """LLaMA 等现代模型使用的 FFN 变体"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # SwiGLU 需要两个上投影
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # 门控
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

---

## 5. 残差连接与 LayerNorm

### 5.1 残差连接

```
作用：
1. 缓解梯度消失
2. 让网络可以更深
3. 让模型可以学习"什么都不做"

output = x + Sublayer(x)
```

### 5.2 LayerNorm

```python
# LayerNorm：在特征维度上归一化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 实际使用 PyTorch 内置的
layer_norm = nn.LayerNorm(512)
```

### 5.3 Pre-LN vs Post-LN

```
Post-LN（原始 Transformer）：
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm

Pre-LN（现代常用）：
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add

Pre-LN 的优势：
- 训练更稳定
- 不需要 Warmup
- 现代 LLM（GPT-3、LLaMA）都用 Pre-LN
```

```python
# Pre-LN 实现
class PreNormBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # Pre-LN: Norm before sublayer
        attn_out, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + attn_out  # 残差连接

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out  # 残差连接

        return x
```

---

## 6. Transformer Block

### 6.1 完整的 Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block (Pre-LN)"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention with residual
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.self_attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_out)

        # FFN with residual
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)

        return x, attn_weights
```

### 6.2 完整的 Decoder Block

```python
class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block (Pre-LN)"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-Attention（用于 Encoder-Decoder 结构）
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        normed_x = self.norm1(x)
        attn_out, _ = self.self_attention(normed_x, normed_x, normed_x, tgt_mask)
        x = x + self.dropout(attn_out)

        # Cross-Attention（如果有 encoder 输出）
        if encoder_output is not None:
            normed_x = self.norm2(x)
            cross_out, _ = self.cross_attention(
                normed_x, encoder_output, encoder_output, src_mask
            )
            x = x + self.dropout(cross_out)

        # FFN
        normed_x = self.norm3(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)

        return x
```

---

## 7. 代码实现

### 7.1 完整的小型 Transformer

```python
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    """简化版 Transformer（Encoder-only，类似 BERT）"""
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        num_classes=2  # 分类任务
    ):
        super().__init__()

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        # 位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # 词嵌入 + 位置嵌入
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Transformer 层
        for layer in self.layers:
            x, _ = layer(x, mask)

        x = self.norm(x)

        # 取 [CLS] 位置（假设是第一个 token）做分类
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)

        return logits

# 测试
model = Transformer(
    vocab_size=10000,
    d_model=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024,
    num_classes=2
)

x = torch.randint(0, 10000, (2, 128))  # [batch, seq_len]
output = model(x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")
```

---

## 8. 练习题

### 基础练习

1. 手动计算一个 4x4 的注意力矩阵（给定 Q、K、V）
2. 实现带 causal mask 的注意力（只能看前面的 token）
3. 修改 FFN 为 SwiGLU

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 手动计算注意力
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
K = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [1., 1.]])
V = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])

d_k = Q.size(-1)
scores = torch.matmul(Q, K.T) / math.sqrt(d_k)
print(f"注意力分数:\n{scores}")

weights = F.softmax(scores, dim=-1)
print(f"注意力权重:\n{weights}")

output = torch.matmul(weights, V)
print(f"输出:\n{output}")


# 2. Causal Mask
def create_causal_mask(seq_len):
    """创建因果掩码（下三角矩阵）"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True 表示可以看，False 表示不能看

mask = create_causal_mask(4)
print(f"因果掩码:\n{mask}")

# 使用
scores = torch.randn(4, 4)
scores = scores.masked_fill(~mask, float('-inf'))
weights = F.softmax(scores, dim=-1)
print(f"带因果掩码的权重:\n{weights}")
# 每行只有对角线及左边有值，其他都是 0


# 3. SwiGLU FFN
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# 测试
swiglu = SwiGLU(512, 2048)
x = torch.randn(2, 10, 512)
y = swiglu(x)
print(f"SwiGLU: {x.shape} -> {y.shape}")
```

</details>

---

## 📖 关键总结

```
Transformer =
  词嵌入 + 位置嵌入
  + N × (Multi-Head Attention + FFN + 残差 + LayerNorm)

Self-Attention:
  1. Q、K、V 分别投影
  2. QK^T / √d_k 计算注意力分数
  3. Softmax 归一化
  4. 加权求和 V

Multi-Head:
  - 并行多个注意力头
  - 每个头学习不同模式
  - 拼接后投影
```

---

## ➡️ 下一步

学完本节后，继续学习 [02-注意力机制详解.md](./02-注意力机制详解.md)

