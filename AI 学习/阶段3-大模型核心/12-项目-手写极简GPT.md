# 🔨 项目：手写极简 GPT

> 从零实现一个简化版的 GPT 模型，理解 Transformer 核心原理

---

## 项目目标

```
实现一个简化版 GPT：
- 手写完整的 Transformer Decoder
- 在小数据集上训练字符级语言模型
- 能够生成连贯的文本

技术要点：
- Multi-Head Self-Attention
- Causal Mask
- Positional Encoding
- Layer Normalization
- 自回归生成
```

---

## 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# ========== 配置 ==========
class GPTConfig:
    """模型配置"""
    vocab_size: int = 256       # 字符级词表大小
    block_size: int = 128       # 上下文长度
    n_embd: int = 256           # embedding 维度
    n_head: int = 8             # 注意力头数
    n_layer: int = 6            # Transformer 层数
    dropout: float = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ========== 组件实现 ==========

class CausalSelfAttention(nn.Module):
    """因果自注意力（带 mask 的 Multi-Head Attention）"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V 投影（一次性计算）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果 mask（下三角矩阵）
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # 重塑为多头形式: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 应用因果 mask
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax + Dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = attn @ v

        # 合并多头: (B, n_head, T, head_dim) -> (B, T, n_head * head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block = Attention + MLP (Pre-LN 结构)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LN -> Attn -> Residual -> LN -> MLP -> Residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """极简 GPT 模型"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position Embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language Model Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定（Embedding 和 LM Head 共享权重）
        self.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

        # 统计参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices for loss computation
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss (if targets provided)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过最大 {self.config.block_size}"

        # Token + Position Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        tok_emb = self.wte(idx)      # (B, T, n_embd)
        pos_emb = self.wpe(pos)      # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None):
        """
        自回归生成

        Args:
            idx: (B, T) 起始 token
            max_new_tokens: 生成的最大 token 数
            temperature: 采样温度
            top_k: Top-K 采样
        """
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ========== 数据准备 ==========
class CharDataset(torch.utils.data.Dataset):
    """字符级数据集"""

    def __init__(self, text: str, block_size: int):
        # 字符到索引的映射
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # 编码文本
        self.data = [self.stoi[ch] for ch in text]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text: str) -> list:
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: list) -> str:
        return ''.join([self.itos[i] for i in indices])


# ========== 训练 ==========
def train(model, dataset, config, num_epochs=10, batch_size=32, lr=3e-4):
    """训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")

        # 生成示例
        generate_sample(model, dataset, device)

    return model


def generate_sample(model, dataset, device, prompt="The ", max_tokens=200):
    """生成示例文本"""
    model.eval()

    # 编码 prompt
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)

    # 生成
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=40)

    # 解码
    generated = dataset.decode(output[0].tolist())
    print(f"\n生成文本:\n{generated}\n{'='*50}")

    model.train()


# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载文本数据（使用莎士比亚数据集）
    import urllib.request

    # 下载数据
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print("下载数据...")
        urllib.request.urlretrieve(url, 'shakespeare.txt')
        with open('shakespeare.txt', 'r') as f:
            text = f.read()

    print(f"文本长度: {len(text)}")
    print(f"前 500 字符:\n{text[:500]}")

    # 创建数据集
    block_size = 128
    dataset = CharDataset(text, block_size)
    print(f"词表大小: {dataset.vocab_size}")
    print(f"训练样本数: {len(dataset)}")

    # 创建模型
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=0.1
    )

    model = MiniGPT(config)

    # 训练
    model = train(
        model, dataset, config,
        num_epochs=5,
        batch_size=64,
        lr=3e-4
    )

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'vocab': dataset.stoi
    }, 'mini_gpt.pth')

    print("训练完成！模型已保存")
```

---

## 核心组件详解

### 1. Causal Self-Attention

```python
"""
因果注意力的关键：下三角 mask

假设序列长度 T=4:
mask = [
    [1, 0, 0, 0],   # token 0 只能看自己
    [1, 1, 0, 0],   # token 1 可以看 0, 1
    [1, 1, 1, 0],   # token 2 可以看 0, 1, 2
    [1, 1, 1, 1],   # token 3 可以看所有
]

attn_scores = Q @ K^T  # (T, T)
attn_scores = attn_scores.masked_fill(mask == 0, -inf)
attn_probs = softmax(attn_scores)  # -inf 变成 0
"""

# 可视化注意力
def visualize_attention(model, text, dataset, device):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt

    idx = torch.tensor([dataset.encode(text)], device=device)

    # 获取注意力权重（需要修改 forward 返回）
    # 这里简化处理

    with torch.no_grad():
        B, T, C = idx.size()[0], idx.size()[1], model.config.n_embd
        # ... 提取注意力权重 ...
```

### 2. Pre-LN vs Post-LN

```python
# Post-LN (原始 Transformer)
x = x + attn(x)
x = ln(x)
x = x + mlp(x)
x = ln(x)

# Pre-LN (GPT-2/3 使用，更稳定)
x = x + attn(ln(x))
x = x + mlp(ln(x))
```

### 3. 生成策略

```python
# Greedy（贪婪）
next_token = logits.argmax(dim=-1)

# Top-K
top_k_logits, top_k_indices = logits.topk(k)
probs = softmax(top_k_logits)
next_token = top_k_indices[multinomial(probs, 1)]

# Top-P (Nucleus)
sorted_logits, sorted_indices = logits.sort(descending=True)
cumsum_probs = softmax(sorted_logits).cumsum(dim=-1)
mask = cumsum_probs > p
sorted_logits[mask] = -inf
probs = softmax(sorted_logits)
next_token = sorted_indices[multinomial(probs, 1)]

# Temperature
logits = logits / temperature  # T > 1 更随机，T < 1 更确定
```

---

## 扩展阅读

```
参考项目：
1. nanoGPT: https://github.com/karpathy/nanoGPT
2. minGPT: https://github.com/karpathy/minGPT
3. llm.c: https://github.com/karpathy/llm.c

视频教程：
- "Let's build GPT" by Andrej Karpathy
  https://www.youtube.com/watch?v=kCc8FmEb1nY
```

---

## 练习

1. **添加 RoPE 位置编码**：替换固定位置编码
2. **实现 GQA**：减少 KV 头数量
3. **添加 KV Cache**：加速推理
4. **实现 Flash Attention**：优化内存

---

## ➡️ 下一步

继续 [13-自测清单.md](./13-自测清单.md)

