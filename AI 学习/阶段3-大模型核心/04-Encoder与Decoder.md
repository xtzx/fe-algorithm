# 🔄 04 - Encoder 与 Decoder

> 理解 BERT、GPT、T5 三种架构的区别和适用场景

---

## 目录

1. [三种架构概述](#1-三种架构概述)
2. [Encoder-only (BERT)](#2-encoder-only-bert)
3. [Decoder-only (GPT)](#3-decoder-only-gpt)
4. [Encoder-Decoder (T5)](#4-encoder-decoder-t5)
5. [架构选择指南](#5-架构选择指南)
6. [练习题](#6-练习题)

---

## 1. 三种架构概述

### 1.1 架构对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer 架构变体                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Encoder-only (BERT)     Decoder-only (GPT)    Encoder-Decoder (T5) │
│  ┌───────────────┐       ┌───────────────┐     ┌─────┬─────────┐   │
│  │               │       │               │     │     │         │   │
│  │   Encoder     │       │   Decoder     │     │ Enc │   Dec   │   │
│  │   (双向)      │       │   (单向)      │     │     │         │   │
│  │               │       │               │     └──┬──┴────┬────┘   │
│  └───────────────┘       └───────────────┘        │       │        │
│                                                   └───────┘        │
│  注意力：全局双向         注意力：因果（只看前面）  Encoder双向+Decoder单向│
│                                                                     │
│  代表：BERT, RoBERTa     代表：GPT, LLaMA        代表：T5, BART      │
│        DeBERTa                 Qwen, Mistral          mT5           │
│                                                                     │
│  用途：理解任务           用途：生成任务          用途：序列到序列    │
│       分类、NER、问答           文本生成、对话          翻译、摘要    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 注意力模式对比

```python
import torch
import matplotlib.pyplot as plt

def visualize_attention_patterns():
    seq_len = 6

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Encoder (BERT): 全局双向注意力
    encoder_mask = torch.ones(seq_len, seq_len)
    axes[0].imshow(encoder_mask, cmap='Blues')
    axes[0].set_title('Encoder-only (BERT)\n双向：每个位置看所有位置')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    # Decoder (GPT): 因果注意力
    decoder_mask = torch.tril(torch.ones(seq_len, seq_len))
    axes[1].imshow(decoder_mask, cmap='Blues')
    axes[1].set_title('Decoder-only (GPT)\n单向：只能看前面的位置')

    # Encoder-Decoder: 混合
    # 简化显示
    enc_dec_mask = torch.ones(seq_len, seq_len)
    enc_dec_mask[3:, :3] = 0.5  # 表示 cross-attention
    axes[2].imshow(enc_dec_mask, cmap='Blues')
    axes[2].set_title('Encoder-Decoder (T5)\nEncoder双向 + Decoder单向 + Cross')

    plt.tight_layout()
    plt.show()

visualize_attention_patterns()
```

---

## 2. Encoder-only (BERT)

### 2.1 BERT 架构

```
BERT = Bidirectional Encoder Representations from Transformers

输入：[CLS] token1 token2 ... tokenN [SEP]
     ↓
Embedding (token + position + segment)
     ↓
┌─────────────────────────────────────┐
│  Transformer Encoder Block × 12     │
│  (Self-Attention + FFN)             │
│  (双向注意力：每个位置看所有位置)    │
└─────────────────────────────────────┘
     ↓
输出：每个位置的上下文表示

[CLS] 位置的输出用于句子级任务
其他位置的输出用于 token 级任务
```

### 2.2 预训练任务

```python
# BERT 的预训练任务

# 1. Masked Language Modeling (MLM)
# 随机遮盖 15% 的 token，让模型预测
"""
输入: The [MASK] sat on the mat.
目标: cat

为什么不能用 GPT 的方式？
因为 BERT 需要双向上下文，不能只看前面
"""

# 2. Next Sentence Prediction (NSP)
# 预测两个句子是否连续
"""
输入: [CLS] Sentence A [SEP] Sentence B [SEP]
输出: IsNext / NotNext

后来发现 NSP 用处不大，RoBERTa 去掉了它
"""
```

### 2.3 BERT 代码示例

```python
import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_segments=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        if segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)

        return self.dropout(self.norm(x))

class BertModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooler = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None, segment_ids=None):
        x = self.embedding(input_ids, segment_ids)

        # attention_mask: 1 表示有效，0 表示 padding
        if attention_mask is not None:
            attention_mask = (attention_mask == 0)  # PyTorch 用 True 表示忽略

        x = self.encoder(x, src_key_padding_mask=attention_mask)

        # [CLS] token 的输出
        cls_output = x[:, 0]
        pooled = torch.tanh(self.pooler(cls_output))

        return x, pooled  # sequence_output, pooled_output

# 分类任务
class BertForClassification(nn.Module):
    def __init__(self, bert, num_classes):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        _, pooled = self.bert(input_ids, attention_mask)
        return self.classifier(self.dropout(pooled))
```

### 2.4 BERT 的适用场景

```
✅ 文本分类（情感分析、主题分类）
✅ 命名实体识别（NER）
✅ 问答（抽取式）
✅ 句子相似度
✅ 文本蕴含

❌ 文本生成（不适合，因为是双向的）
❌ 开放式对话
```

---

## 3. Decoder-only (GPT)

### 3.1 GPT 架构

```
GPT = Generative Pre-trained Transformer

输入：token1 token2 ... tokenN
     ↓
Embedding (token + position)
     ↓
┌─────────────────────────────────────┐
│  Transformer Decoder Block × N      │
│  (Causal Self-Attention + FFN)      │
│  (单向注意力：只能看前面的 token)    │
└─────────────────────────────────────┘
     ↓
输出：下一个 token 的预测

自回归生成：
预测 token1 → 预测 token2（基于 token1）→ 预测 token3（基于 1,2）→ ...
```

### 3.2 预训练任务

```python
# GPT 的预训练任务：Next Token Prediction

"""
输入: The cat sat on
目标: the (预测下一个 token)

输入: The cat sat on the
目标: mat

这就是为什么叫"语言模型"
P(sentence) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...
"""

# 损失函数：Cross Entropy Loss
# 预测每个位置的下一个 token
```

### 3.3 GPT 代码示例

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_len=1024):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 因果掩码
        self.register_buffer('causal_mask', None)

    def _get_causal_mask(self, seq_len, device):
        # 上三角为 True（被忽略），下三角为 False（保留）
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        positions = torch.arange(seq_len, device=device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 因果掩码
        causal_mask = self._get_causal_mask(seq_len, device)

        # Transformer
        x = self.transformer(x, mask=causal_mask)
        x = self.ln_f(x)

        # 预测下一个 token
        logits = self.lm_head(x)

        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """自回归生成"""
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(input_ids)

            # 取最后一个位置的预测
            next_token_logits = logits[:, -1, :] / temperature

            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# 训练
def train_gpt(model, dataloader, optimizer, device):
    model.train()

    for input_ids in dataloader:
        input_ids = input_ids.to(device)

        # 输入和目标错位一个位置
        x = input_ids[:, :-1]  # 输入
        y = input_ids[:, 1:]   # 目标（下一个 token）

        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.4 现代 LLM 架构（LLaMA 风格）

```python
class LLaMABlock(nn.Module):
    """LLaMA 风格的 Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()

        # Pre-LN
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # 注意力（带 RoPE）
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)

        # SwiGLU FFN
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, freqs_cis=None):
        # 残差 + Pre-LN
        x = x + self.attention(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x

class RMSNorm(nn.Module):
    """RMS Normalization（LLaMA 使用，比 LayerNorm 更高效）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

---

## 4. Encoder-Decoder (T5)

### 4.1 T5 架构

```
T5 = Text-to-Text Transfer Transformer

把所有任务统一为 text-to-text 格式：

翻译：   "translate English to German: The cat sat on the mat."
         → "Die Katze saß auf der Matte."

摘要：   "summarize: [长文本]"
         → "短摘要"

分类：   "sentiment: This movie is great!"
         → "positive"

问答：   "question: What is AI? context: AI is..."
         → "AI is artificial intelligence"
```

### 4.2 Encoder-Decoder 结构

```python
class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048):
        super().__init__()

        # 共享词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Encoder
        src_emb = self.embedding(src_ids)
        encoder_output = self.encoder(src_emb, src_key_padding_mask=src_mask)

        # Decoder
        tgt_emb = self.embedding(tgt_ids)
        # Decoder 需要因果掩码
        tgt_causal_mask = self._generate_causal_mask(tgt_ids.size(1), tgt_ids.device)

        decoder_output = self.decoder(
            tgt_emb,
            encoder_output,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )

        logits = self.lm_head(decoder_output)
        return logits

    def _generate_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
```

### 4.3 T5 的适用场景

```
✅ 机器翻译
✅ 文本摘要
✅ 问答（生成式）
✅ 文本改写
✅ 数据增强

特点：
- 灵活：可以处理各种 seq2seq 任务
- 但参数量通常比 Decoder-only 大（因为有两套）
```

---

## 5. 架构选择指南

### 5.1 决策树

```
你的任务是什么？
│
├─→ 理解/分类任务？
│   └─→ 用 Encoder-only (BERT)
│
├─→ 生成任务？
│   ├─→ 开放生成（聊天、写作）？
│   │   └─→ 用 Decoder-only (GPT/LLaMA)
│   │
│   └─→ 条件生成（翻译、摘要）？
│       ├─→ 输入输出长度差异大？
│       │   └─→ 用 Encoder-Decoder (T5)
│       │
│       └─→ 长度相近/通用需求？
│           └─→ Decoder-only 也可以（指令微调）
│
└─→ 不确定？
    └─→ 现代趋势：Decoder-only + 指令微调
        （一个模型搞定所有任务）
```

### 5.2 现代趋势

```
2023-2024 的趋势：

1. Decoder-only 成为主流
   - GPT-4, Claude, LLaMA, Qwen 都是 Decoder-only
   - 通过指令微调，一个模型可以做所有任务

2. 规模效应
   - 足够大的 Decoder-only 模型可以涌现各种能力
   - 不需要专门设计架构

3. 特定任务仍有价值
   - 小规模场景：BERT 仍然高效
   - 翻译/摘要：T5 系列仍然强

4. 多模态
   - 视觉编码器 + LLM 解码器（类似 Encoder-Decoder）
```

---

## 6. 练习题

### 基础练习

1. 用 Hugging Face 加载 BERT 和 GPT-2，观察它们的注意力模式
2. 实现一个简单的 seq2seq 模型（Encoder-Decoder）
3. 比较 BERT 和 GPT 在文本分类任务上的效果

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
from transformers import BertModel, GPT2Model, BertTokenizer, GPT2Tokenizer

# 1. 加载模型并观察注意力
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = GPT2Model.from_pretrained('gpt2', output_attentions=True)

text = "The quick brown fox jumps over the lazy dog."

# BERT
inputs_bert = tokenizer_bert(text, return_tensors='pt')
outputs_bert = model_bert(**inputs_bert)
attn_bert = outputs_bert.attentions[0][0]  # 第一层，第一个样本

# GPT-2
inputs_gpt = tokenizer_gpt(text, return_tensors='pt')
outputs_gpt = model_gpt(**inputs_gpt)
attn_gpt = outputs_gpt.attentions[0][0]

print(f"BERT 注意力形状: {attn_bert.shape}")  # [num_heads, seq_len, seq_len]
print(f"GPT-2 注意力形状: {attn_gpt.shape}")

# 观察：GPT-2 的注意力矩阵是下三角（因果掩码）
# BERT 的注意力矩阵是全的（双向）


# 2. 简单 seq2seq
class SimpleSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # Encode
        src_emb = self.embedding(src)
        _, (hidden, cell) = self.encoder(src_emb)

        # Decode
        tgt_emb = self.embedding(tgt)
        outputs, _ = self.decoder(tgt_emb, (hidden, cell))

        return self.fc(outputs)

# 测试
model = SimpleSeq2Seq(vocab_size=1000)
src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))
out = model(src, tgt)
print(f"Seq2Seq 输出: {out.shape}")  # [2, 8, 1000]


# 3. BERT vs GPT 分类对比（伪代码）
"""
from transformers import BertForSequenceClassification, GPT2ForSequenceClassification

# BERT 分类：天然适合，使用 [CLS] token
bert_clf = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# GPT 分类：需要使用最后一个 token
gpt_clf = GPT2ForSequenceClassification.from_pretrained('gpt2')
gpt_clf.config.pad_token_id = tokenizer_gpt.eos_token_id

# 通常 BERT 在小数据集分类任务上效果更好
# 因为它是双向的，能更好地理解上下文
"""
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [05-LLM优化技术.md](./05-LLM优化技术.md)

