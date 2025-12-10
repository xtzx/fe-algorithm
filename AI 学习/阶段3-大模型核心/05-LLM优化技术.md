# ⚡ 05 - LLM 优化技术

> KV Cache、Flash Attention、GQA、MoE 是现代 LLM 的关键优化技术

---

## 目录

1. [KV Cache](#1-kv-cache)
2. [Flash Attention](#2-flash-attention)
3. [GQA 与 MQA](#3-gqa-与-mqa)
4. [MoE 混合专家](#4-moe-混合专家)
5. [其他优化技术](#5-其他优化技术)
6. [练习题](#6-练习题)

---

## 1. KV Cache

### 1.1 问题：重复计算

```
自回归生成时，每生成一个 token 都需要重新计算整个序列的注意力

Step 1: "The" → 计算 Q,K,V for "The"
Step 2: "The cat" → 重新计算 Q,K,V for "The" 和 "cat"
Step 3: "The cat sat" → 重新计算 Q,K,V for "The", "cat", "sat"
...

问题：
- 时间复杂度：O(n²) 每步
- 计算冗余：之前 token 的 K,V 每次都重算

生成 n 个 token 的总复杂度：O(n³)！
```

### 1.2 KV Cache 解决方案

```
核心思想：缓存已计算的 K 和 V，只计算新 token 的 K,V

Step 1: "The"
  - 计算 K1,V1 → 缓存
  - Q1 和 K1 计算注意力

Step 2: "The cat"
  - 从缓存取 K1,V1
  - 只计算新的 K2,V2 → 追加到缓存
  - Q2 和 [K1,K2] 计算注意力

Step 3: "The cat sat"
  - 从缓存取 K1,V1,K2,V2
  - 只计算新的 K3,V3 → 追加到缓存
  - Q3 和 [K1,K2,K3] 计算注意力

优化后：每步 O(n)，总复杂度 O(n²)
```

### 1.3 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttentionWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads, max_len=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        x: [batch, seq_len, d_model] - 当前输入
        kv_cache: (cached_k, cached_v) - 之前的 K,V 缓存
        use_cache: 是否使用和更新缓存
        """
        batch_size, seq_len, _ = x.shape

        # 计算当前输入的 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # [batch, num_heads, seq_len, d_k]

        # 使用 KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # 拼接缓存的 K,V 和当前的 K,V
            K = torch.cat([cached_k, K], dim=2)
            V = torch.cat([cached_v, V], dim=2)

        # 更新缓存
        new_cache = (K, V) if use_cache else None

        # 计算注意力
        # Q: [batch, heads, 1, d_k] (生成时通常 seq_len=1)
        # K: [batch, heads, total_len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码（只对当前位置有效的部分）
        # 生成时 Q 只有一个 token，不需要复杂的因果掩码
        if Q.size(2) > 1:
            causal_mask = torch.triu(
                torch.ones(Q.size(2), K.size(2), device=x.device),
                diagonal=K.size(2) - Q.size(2) + 1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # 重塑
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, new_cache

# 使用示例
attn = CausalSelfAttentionWithKVCache(d_model=256, num_heads=4)

# 预填充阶段（处理 prompt）
prompt = torch.randn(1, 10, 256)  # 10 个 token 的 prompt
output, kv_cache = attn(prompt, kv_cache=None, use_cache=True)
print(f"Prompt 输出: {output.shape}")
print(f"缓存 K: {kv_cache[0].shape}")  # [1, 4, 10, 64]

# 生成阶段（一次生成一个 token）
for i in range(5):
    new_token = torch.randn(1, 1, 256)  # 新生成的 token
    output, kv_cache = attn(new_token, kv_cache=kv_cache, use_cache=True)
    print(f"Step {i+1}: 输出 {output.shape}, 缓存长度 {kv_cache[0].shape[2]}")
```

### 1.4 KV Cache 的内存占用

```python
def calculate_kv_cache_memory(
    batch_size,
    seq_len,
    num_layers,
    num_heads,
    d_k,
    dtype=torch.float16
):
    """计算 KV Cache 的内存占用"""
    # 每层需要存储 K 和 V
    # K: [batch, num_heads, seq_len, d_k]
    # V: [batch, num_heads, seq_len, d_k]

    bytes_per_element = 2 if dtype == torch.float16 else 4

    # 每层的 KV 缓存大小
    kv_per_layer = batch_size * num_heads * seq_len * d_k * 2  # *2 for K and V

    # 总缓存大小
    total_elements = kv_per_layer * num_layers
    total_bytes = total_elements * bytes_per_element

    return total_bytes / (1024 ** 3)  # GB

# LLaMA-7B 参数
memory_gb = calculate_kv_cache_memory(
    batch_size=1,
    seq_len=4096,
    num_layers=32,
    num_heads=32,
    d_k=128,
    dtype=torch.float16
)
print(f"LLaMA-7B KV Cache (4K context): {memory_gb:.2f} GB")

# 长上下文的挑战
memory_gb_32k = calculate_kv_cache_memory(
    batch_size=1,
    seq_len=32768,
    num_layers=32,
    num_heads=32,
    d_k=128,
    dtype=torch.float16
)
print(f"LLaMA-7B KV Cache (32K context): {memory_gb_32k:.2f} GB")
```

---

## 2. Flash Attention

### 2.1 标准注意力的内存问题

```
标准注意力计算：
1. S = QK^T      → 需要存储 [seq_len, seq_len] 矩阵
2. P = softmax(S) → 同样大小
3. O = PV        → 最终输出

问题：中间矩阵的显存占用是 O(n²)
- 当 seq_len = 8192 时，注意力矩阵占用 256MB（FP32）
- 需要大量 GPU-HBM 带宽

这是标准注意力的瓶颈：内存带宽受限，而非计算受限
```

### 2.2 Flash Attention 核心思想

```
核心思想：分块计算 + 重计算

1. 将 Q, K, V 分成小块（blocks）
2. 逐块计算注意力，不存储完整的注意力矩阵
3. 使用 online softmax 技巧，保持数值稳定

优势：
- 显存占用从 O(n²) 降到 O(n)
- 减少 GPU-HBM 带宽需求
- 速度更快（尤其是长序列）

关键技术：
- Tiling：分块计算
- Recomputation：反向传播时重新计算，而不是存储
- Online Softmax：增量计算 softmax
```

### 2.3 使用 Flash Attention

```python
# Flash Attention 通常通过优化的 CUDA 内核提供
# PyTorch 2.0+ 内置了 scaled_dot_product_attention

import torch
import torch.nn.functional as F

# 方法 1：PyTorch 2.0+ 自动使用 Flash Attention
q = torch.randn(2, 8, 1024, 64, device='cuda')  # [batch, heads, seq, d_k]
k = torch.randn(2, 8, 1024, 64, device='cuda')
v = torch.randn(2, 8, 1024, 64, device='cuda')

# 自动选择最优实现（包括 Flash Attention）
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print(f"Flash Attention 输出: {output.shape}")

# 方法 2：使用 flash_attn 库
# pip install flash-attn
"""
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
"""

# 性能对比
import time

def benchmark_attention(q, k, v, method='standard'):
    if method == 'standard':
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
    else:  # flash
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return output

# 预热
for _ in range(10):
    _ = benchmark_attention(q, k, v, 'flash')
torch.cuda.synchronize()

# 测试
start = time.time()
for _ in range(100):
    _ = benchmark_attention(q, k, v, 'flash')
torch.cuda.synchronize()
flash_time = time.time() - start

print(f"Flash Attention 时间: {flash_time:.4f}s")
```

---

## 3. GQA 与 MQA

### 3.1 问题：KV Cache 太大

```
标准 Multi-Head Attention (MHA):
- 每个头都有独立的 K, V
- KV Cache 大小：2 * num_heads * d_k * seq_len * num_layers

LLaMA-70B 的 KV Cache (4K context):
- num_heads=64, d_k=128, layers=80
- 约 40GB 仅用于缓存！

解决方案：减少 K, V 的头数
```

### 3.2 MQA（Multi-Query Attention）

```
MQA：所有 Query 头共享一套 K, V

标准 MHA:
  Q: [batch, heads, seq, d_k]  → 每个头独立
  K: [batch, heads, seq, d_k]  → 每个头独立
  V: [batch, heads, seq, d_k]  → 每个头独立

MQA:
  Q: [batch, heads, seq, d_k]  → 每个头独立
  K: [batch, 1, seq, d_k]      → 所有头共享！
  V: [batch, 1, seq, d_k]      → 所有头共享！

优势：KV Cache 减少 num_heads 倍
缺点：模型质量可能下降
```

### 3.3 GQA（Grouped-Query Attention）

```
GQA：折中方案，将 Q 头分成 G 组，每组共享一套 K, V

例：8 个 Q 头，2 组
  Q: [batch, 8, seq, d_k]   → 8 个头
  K: [batch, 2, seq, d_k]   → 2 组，每组 4 个 Q 头共享
  V: [batch, 2, seq, d_k]   → 同上

LLaMA 2 70B 使用 GQA:
- 64 个 Q 头，8 组 K/V 头
- KV Cache 减少 8 倍
```

### 3.4 代码实现

```python
class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA)"""
    def __init__(self, d_model, num_q_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须能被 num_kv_heads 整除"

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads  # 每组有多少 Q 头
        self.d_k = d_model // num_q_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)  # 更少的 K 头
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)  # 更少的 V 头
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑
        Q = Q.view(batch_size, seq_len, self.num_q_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # KV Cache
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)
        new_cache = (K, V) if use_cache else None

        # 扩展 K, V 以匹配 Q 的头数
        # [batch, num_kv_heads, seq, d_k] → [batch, num_q_heads, seq, d_k]
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码
        if Q.size(2) > 1:
            causal_mask = torch.triu(
                torch.ones(Q.size(2), K.size(2), device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)

        # 输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output), new_cache

# 对比参数量和缓存大小
d_model = 4096
seq_len = 4096
batch_size = 1

# MHA: 32 Q 头，32 KV 头
mha = nn.MultiheadAttention(d_model, num_heads=32, batch_first=True)

# GQA: 32 Q 头，8 KV 头
gqa = GroupedQueryAttention(d_model, num_q_heads=32, num_kv_heads=8)

print(f"MHA 参数量: {sum(p.numel() for p in mha.parameters()):,}")
print(f"GQA 参数量: {sum(p.numel() for p in gqa.parameters()):,}")

# KV Cache 大小对比
mha_cache_size = batch_size * 32 * seq_len * (d_model // 32) * 2  # K + V
gqa_cache_size = batch_size * 8 * seq_len * (d_model // 32) * 2
print(f"MHA KV Cache: {mha_cache_size * 2 / 1e6:.2f} MB (FP16)")
print(f"GQA KV Cache: {gqa_cache_size * 2 / 1e6:.2f} MB (FP16)")
print(f"缓存减少: {mha_cache_size / gqa_cache_size:.1f}x")
```

---

## 4. MoE 混合专家

### 4.1 核心思想

```
问题：模型越大效果越好，但计算成本也越高

MoE 的解决方案：
- 有很多"专家"（Expert），每个是一个 FFN
- 每次只激活少数专家处理当前 token
- 参数量大，但实际计算量小

例：Mixtral 8x7B
- 8 个专家，每个 7B 参数
- 总参数量 ~47B
- 每次只激活 2 个专家
- 实际推理成本约 14B
```

### 4.2 结构

```
标准 Transformer:
  Input → Attention → FFN → Output

MoE Transformer:
  Input → Attention → Router → [Expert 1]
                             [Expert 2]  → 加权合并 → Output
                             [Expert 3]
                             ...
                             [Expert N]

Router（门控网络）：
- 决定每个 token 激活哪些专家
- 输出每个专家的权重
- Top-K 选择（通常 K=2）
```

### 4.3 代码实现

```python
class Expert(nn.Module):
    """单个专家（就是一个 FFN）"""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))))

class MoELayer(nn.Module):
    """Mixture of Experts 层"""
    def __init__(self, d_model, d_ff, num_experts, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 专家们
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # 门控网络（Router）
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # 展平为 [batch * seq_len, d_model]
        x_flat = x.view(-1, d_model)

        # 门控：计算每个 token 对每个专家的权重
        gate_logits = self.gate(x_flat)  # [batch*seq, num_experts]

        # Top-K 选择
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [batch*seq, top_k]

        # 计算输出
        # 简化实现：遍历每个专家
        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # 找出选择了这个专家的 token
            mask = (top_k_indices == i).any(dim=-1)  # [batch*seq]

            if mask.sum() == 0:
                continue

            # 获取这些 token 对应的权重
            expert_weight = torch.zeros(x_flat.size(0), device=x.device)
            for k in range(self.top_k):
                expert_weight += (top_k_indices[:, k] == i).float() * top_k_weights[:, k]

            # 计算专家输出并加权
            expert_output = expert(x_flat)
            output += expert_weight.unsqueeze(-1) * expert_output

        return output.view(batch_size, seq_len, d_model)

# 使用示例
moe = MoELayer(d_model=256, d_ff=1024, num_experts=8, top_k=2)
x = torch.randn(2, 10, 256)
y = moe(x)
print(f"MoE 输出: {y.shape}")

# 参数量对比
dense_ffn = Expert(256, 1024)
print(f"单个 FFN 参数: {sum(p.numel() for p in dense_ffn.parameters()):,}")
print(f"MoE (8 experts) 参数: {sum(p.numel() for p in moe.parameters()):,}")
print("但实际计算量只是 2 个 FFN！")
```

### 4.4 MoE 的挑战

```
1. 负载均衡
   - 某些专家可能被过度使用，其他专家闲置
   - 解决：添加 auxiliary loss 鼓励均衡

2. 通信开销
   - 分布式训练时，token 需要发送到不同 GPU 上的专家
   - 解决：Expert Parallelism

3. 训练不稳定
   - 门控网络的训练可能不稳定
   - 解决：容量因子、噪声门控
```

---

## 5. 其他优化技术

### 5.1 量化

```python
# 模型量化：减少参数精度，降低内存和计算

# 常见量化级别
# FP32: 32位浮点（完整精度）
# FP16/BF16: 16位浮点（训练常用）
# INT8: 8位整数（推理）
# INT4: 4位整数（激进量化）

# 使用 bitsandbytes 进行 INT8 量化
"""
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
"""

# 使用 GGUF/GGML 格式进行 INT4 量化（llama.cpp）
"""
# 命令行
./quantize model.bin model_q4.bin q4_0
"""
```

### 5.2 推测解码（Speculative Decoding）

```python
"""
推测解码：用小模型快速生成草稿，大模型验证

1. 小模型（Draft Model）快速生成 K 个 token
2. 大模型一次性验证这 K 个 token
3. 接受正确的，拒绝错误的

优势：
- 如果草稿正确率高，可以大幅加速
- 输出质量与大模型完全一致（保证）

伪代码：
draft_tokens = draft_model.generate(prompt, num_tokens=K)
verified = target_model.verify(prompt + draft_tokens)
accept_count = find_first_mismatch(draft_tokens, verified)
"""
```

### 5.3 连续批处理（Continuous Batching）

```
问题：不同请求的生成长度不同，短请求要等长请求

传统批处理：
  Request A: [################]
  Request B: [####____________]  ← 浪费
  Request C: [########________]  ← 浪费

连续批处理（vLLM）：
  当 B 完成时，立即填入新请求 D
  GPU 利用率大幅提升
```

---

## 6. 练习题

### 基础练习

1. 实现带 KV Cache 的注意力，并验证输出一致性
2. 计算不同模型配置的 KV Cache 大小
3. 理解 GQA 如何减少内存占用

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. KV Cache 一致性验证
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.d_k = d_model

    def forward_full(self, x):
        """不使用缓存，计算完整序列"""
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 因果掩码
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(x.device), float('-inf'))
        return torch.matmul(F.softmax(scores, dim=-1), V)

    def forward_cached(self, x, kv_cache=None):
        """使用 KV Cache 逐步生成"""
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=1)
            V = torch.cat([kv_cache[1], V], dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return torch.matmul(F.softmax(scores, dim=-1), V), (K, V)

# 验证
torch.manual_seed(42)
attn = SimpleAttention(64)
x = torch.randn(1, 5, 64)

# 完整计算
out_full = attn.forward_full(x)

# 逐步计算
kv_cache = None
outputs = []
for i in range(5):
    out_i, kv_cache = attn.forward_cached(x[:, i:i+1], kv_cache)
    outputs.append(out_i)

out_cached = torch.cat(outputs, dim=1)

print(f"完整计算: {out_full.shape}")
print(f"缓存计算: {out_cached.shape}")
print(f"差异: {(out_full - out_cached).abs().max():.6f}")  # 应该接近 0


# 2. KV Cache 大小计算
def kv_cache_size_gb(model_config):
    """计算不同模型的 KV Cache 大小"""
    b = model_config['batch_size']
    s = model_config['seq_len']
    l = model_config['num_layers']
    h = model_config['num_kv_heads']
    d = model_config['d_k']

    # 2 for K and V, 2 for FP16
    size_bytes = b * s * l * h * d * 2 * 2
    return size_bytes / (1024 ** 3)

models = {
    'LLaMA-7B': {'batch_size': 1, 'seq_len': 4096, 'num_layers': 32, 'num_kv_heads': 32, 'd_k': 128},
    'LLaMA-70B (GQA)': {'batch_size': 1, 'seq_len': 4096, 'num_layers': 80, 'num_kv_heads': 8, 'd_k': 128},
    'Mixtral-8x7B': {'batch_size': 1, 'seq_len': 32768, 'num_layers': 32, 'num_kv_heads': 8, 'd_k': 128},
}

for name, config in models.items():
    print(f"{name}: {kv_cache_size_gb(config):.2f} GB")
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [06-Tokenization.md](./06-Tokenization.md)

