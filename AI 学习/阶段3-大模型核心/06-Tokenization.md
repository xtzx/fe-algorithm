# 🔤 06 - Tokenization

> Tokenization 是 LLM 处理文本的第一步，理解它对于使用和优化模型至关重要

---

## 目录

1. [什么是 Tokenization](#1-什么是-tokenization)
2. [BPE 算法](#2-bpe-算法)
3. [其他分词算法](#3-其他分词算法)
4. [Hugging Face Tokenizers](#4-hugging-face-tokenizers)
5. [特殊 Token](#5-特殊-token)
6. [实践注意事项](#6-实践注意事项)
7. [练习题](#7-练习题)

---

## 1. 什么是 Tokenization

### 1.1 为什么需要 Tokenization

```
计算机不能直接理解文字，需要转换为数字

简单方法：
1. 字符级：'H','e','l','l','o' → [1,2,3,3,4]
   - 词表小，但序列长

2. 词级：'Hello', 'World' → [1234, 5678]
   - 序列短，但词表大
   - 未见词（OOV）问题

子词 Tokenization（现代方法）：
- 常见词保持完整
- 罕见词拆成子词
- 平衡词表大小和序列长度
```

### 1.2 Token vs 字符 vs 词

```python
text = "Hello, ChatGPT! 你好，世界！"

# 字符级
chars = list(text)
print(f"字符数: {len(chars)}")  # 20

# 词级（简单空格分割）
words = text.split()
print(f"词数: {len(words)}")  # 3

# 子词级（GPT Tokenizer）
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.tokenize(text)
print(f"Token 数: {len(tokens)}")  # 约 10-15
print(f"Tokens: {tokens}")

# 中文：每个字通常是 1-2 个 token
# 英文：常见词 1 个 token，罕见词多个 token
```

### 1.3 Token 与成本

```
API 定价通常按 token 计费：
- GPT-4: ~$0.03 / 1K input tokens
- Claude: ~$0.003 / 1K input tokens

经验法则：
- 英文：1 token ≈ 4 字符 ≈ 0.75 词
- 中文：1 token ≈ 1-2 汉字

"The quick brown fox" = 4 tokens
"敏捷的棕色狐狸跳过" ≈ 8-10 tokens
```

---

## 2. BPE 算法

### 2.1 BPE 原理

```
BPE（Byte Pair Encoding）：迭代合并最常见的字符对

初始：词表是所有单字符
迭代：
  1. 统计所有相邻字符对的频率
  2. 合并频率最高的字符对为新 token
  3. 更新词表
  4. 重复直到达到目标词表大小

示例：
原始文本："aaabdaaabac"

Step 1: 最常见对 "aa" → 合并为 "Z"
  → "ZabdZabac" (Z=aa)

Step 2: 最常见对 "Za" → 合并为 "Y"
  → "YbdYbac" (Y=Za=aaa)

Step 3: ...继续
```

### 2.2 BPE 实现

```python
from collections import Counter, defaultdict

def get_pair_stats(vocab):
    """统计相邻字符对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """合并最常见的字符对"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq

    return new_vocab

def train_bpe(text, num_merges):
    """训练 BPE"""
    # 初始化：每个字符加空格，词尾加 </w>
    words = text.split()
    vocab = Counter()
    for word in words:
        word_with_space = ' '.join(list(word)) + ' </w>'
        vocab[word_with_space] += 1

    merges = []

    for i in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

        print(f"Merge {i+1}: {best_pair} → {''.join(best_pair)}")

    return vocab, merges

# 示例
text = "low lower lowest low lower new newer newest"
vocab, merges = train_bpe(text, num_merges=10)

print("\n最终词表:")
for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:10]:
    print(f"  {word}: {freq}")
```

### 2.3 BPE 分词

```python
def bpe_tokenize(text, merges):
    """使用学到的 BPE 规则分词"""
    words = text.split()
    tokens = []

    for word in words:
        # 初始化为字符
        word_tokens = list(word) + ['</w>']

        # 应用合并规则
        while True:
            # 找到可以合并的对
            pairs = [(word_tokens[i], word_tokens[i+1])
                     for i in range(len(word_tokens) - 1)]

            # 找到在 merges 中排名最高的对
            best_pair = None
            best_idx = float('inf')
            for pair in pairs:
                if pair in merges:
                    idx = merges.index(pair)
                    if idx < best_idx:
                        best_idx = idx
                        best_pair = pair

            if best_pair is None:
                break

            # 合并
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and \
                   (word_tokens[i], word_tokens[i+1]) == best_pair:
                    new_tokens.append(''.join(best_pair))
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_tokens

        tokens.extend(word_tokens)

    return tokens

# 测试
tokens = bpe_tokenize("lowest", merges)
print(f"'lowest' → {tokens}")
```

---

## 3. 其他分词算法

### 3.1 WordPiece（BERT）

```
WordPiece：类似 BPE，但使用不同的合并策略

BPE：合并频率最高的对
WordPiece：合并能最大化语言模型似然的对

特点：
- 子词用 ## 前缀表示（除了词首）
- "unbelievable" → ["un", "##believ", "##able"]
```

### 3.2 Unigram（T5、ALBERT）

```
Unigram：从大词表开始，逐步删减

1. 初始化：包含所有可能子串的大词表
2. 计算每个子词的重要性（基于语言模型）
3. 删除不重要的子词
4. 重复直到达到目标大小

优点：可以输出多种分词结果的概率
```

### 3.3 SentencePiece

```python
# SentencePiece：语言无关的分词器
# 不依赖预分词（空格分割），直接在原始文本上训练

# 安装
# pip install sentencepiece

import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='data.txt',
    model_prefix='mymodel',
    vocab_size=8000,
    model_type='bpe'  # 或 'unigram'
)

# 加载
sp = spm.SentencePieceProcessor()
sp.load('mymodel.model')

# 分词
text = "Hello, world! 你好世界"
tokens = sp.encode_as_pieces(text)
print(f"Tokens: {tokens}")

ids = sp.encode_as_ids(text)
print(f"IDs: {ids}")

# 解码
decoded = sp.decode_pieces(tokens)
print(f"Decoded: {decoded}")
```

### 3.4 算法对比

| 算法 | 合并策略 | 使用模型 |
|------|---------|---------|
| BPE | 频率最高 | GPT, LLaMA |
| WordPiece | 似然最大 | BERT |
| Unigram | 语言模型 | T5, ALBERT |
| SentencePiece | BPE/Unigram | 多语言模型 |

---

## 4. Hugging Face Tokenizers

### 4.1 基本使用

```python
from transformers import AutoTokenizer

# 加载预训练 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 分词
text = "Hello, how are you doing today?"

# 方法 1：基础分词
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# 方法 2：完整编码（推荐）
encoding = tokenizer(text)
print(f"Input IDs: {encoding['input_ids']}")
print(f"Attention Mask: {encoding['attention_mask']}")

# 方法 3：编码并填充到固定长度
encoding = tokenizer(
    text,
    padding='max_length',
    max_length=20,
    truncation=True,
    return_tensors='pt'
)
print(f"Padded IDs: {encoding['input_ids']}")

# 解码
decoded = tokenizer.decode(encoding['input_ids'][0])
print(f"Decoded: {decoded}")
```

### 4.2 批量处理

```python
texts = [
    "Hello, world!",
    "This is a longer sentence that will need different handling.",
    "Short."
]

# 批量编码（自动填充到最长序列）
batch_encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors='pt'
)

print(f"Batch shape: {batch_encoding['input_ids'].shape}")
print(f"Attention masks:\n{batch_encoding['attention_mask']}")
```

### 4.3 不同模型的 Tokenizer

```python
# GPT-2 Tokenizer（BPE）
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(gpt2_tokenizer.tokenize("Hello, world!"))
# ['Hello', ',', 'Ġworld', '!']  # Ġ 表示前面有空格

# BERT Tokenizer（WordPiece）
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(bert_tokenizer.tokenize("unbelievable"))
# ['un', '##believable'] 或 ['un', '##believ', '##able']

# LLaMA Tokenizer（SentencePiece）
# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 中文 Tokenizer
chinese_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
print(chinese_tokenizer.tokenize("你好世界"))
# ['你', '好', '世', '界']
```

### 4.4 Tokenizer 的词表

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 词表大小
print(f"词表大小: {len(tokenizer)}")  # 50257

# 查看词表
vocab = tokenizer.get_vocab()
print(f"前 10 个 token: {list(vocab.items())[:10]}")

# Token ID 转换
token = "hello"
token_id = tokenizer.convert_tokens_to_ids(token)
print(f"'{token}' → ID: {token_id}")

# ID 转 Token
token_back = tokenizer.convert_ids_to_tokens(token_id)
print(f"ID {token_id} → '{token_back}'")

# 添加新 token
num_added = tokenizer.add_tokens(['[CUSTOM]', '[SPECIAL]'])
print(f"添加了 {num_added} 个新 token")
print(f"新词表大小: {len(tokenizer)}")
```

---

## 5. 特殊 Token

### 5.1 常见特殊 Token

```python
from transformers import AutoTokenizer

# BERT
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"BERT 特殊 token:")
print(f"  [PAD]: {bert_tok.pad_token} (ID: {bert_tok.pad_token_id})")
print(f"  [UNK]: {bert_tok.unk_token} (ID: {bert_tok.unk_token_id})")
print(f"  [CLS]: {bert_tok.cls_token} (ID: {bert_tok.cls_token_id})")
print(f"  [SEP]: {bert_tok.sep_token} (ID: {bert_tok.sep_token_id})")
print(f"  [MASK]: {bert_tok.mask_token} (ID: {bert_tok.mask_token_id})")

# GPT-2
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
print(f"\nGPT-2 特殊 token:")
print(f"  BOS: {gpt2_tok.bos_token}")  # 可能为 None
print(f"  EOS: {gpt2_tok.eos_token} (ID: {gpt2_tok.eos_token_id})")
```

### 5.2 特殊 Token 的作用

```
[PAD] - 填充 token，用于对齐不同长度的序列
[UNK] - 未知 token，用于词表外的词
[CLS] - 句子开始 token，其输出用于句子级任务（BERT）
[SEP] - 分隔 token，用于分隔句子对
[MASK] - 掩码 token，用于 MLM 训练
[BOS] - 序列开始 token（Begin of Sequence）
[EOS] - 序列结束 token（End of Sequence）

LLM 中的特殊 token：
<|im_start|> - 消息开始
<|im_end|> - 消息结束
<|system|>, <|user|>, <|assistant|> - 角色标记
```

### 5.3 Chat Template

```python
from transformers import AutoTokenizer

# 使用 chat template
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# 构建对话
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you?"},
    {"role": "user", "content": "What's the weather like?"}
]

# 应用 chat template（如果支持）
try:
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Formatted:\n{formatted}")
except:
    print("This tokenizer doesn't have a chat template")

# 手动构建（通用方法）
def format_chat(messages, system_prompt="You are a helpful assistant."):
    formatted = f"<|system|>\n{system_prompt}<|end|>\n"
    for msg in messages:
        role = msg['role']
        content = msg['content']
        formatted += f"<|{role}|>\n{content}<|end|>\n"
    formatted += "<|assistant|>\n"
    return formatted
```

---

## 6. 实践注意事项

### 6.1 上下文窗口

```python
# 检查上下文窗口限制
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"GPT-2 最大长度: {tokenizer.model_max_length}")  # 1024

# 处理超长文本
long_text = "..." * 2000  # 很长的文本

encoding = tokenizer(
    long_text,
    truncation=True,
    max_length=512,
    return_overflowing_tokens=True,  # 返回溢出的 token
    stride=50  # 滑动窗口重叠
)

print(f"分块数: {len(encoding['input_ids'])}")
```

### 6.2 Tokenization 影响

```python
# Token 数量影响生成质量和成本

def analyze_tokenization(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)

    print(f"\n{model_name}:")
    print(f"  文本长度: {len(text)} 字符")
    print(f"  Token 数: {len(tokens)}")
    print(f"  压缩率: {len(text) / len(tokens):.2f} 字符/token")
    print(f"  Tokens: {tokens[:10]}...")

text_en = "The quick brown fox jumps over the lazy dog."
text_zh = "敏捷的棕色狐狸跳过了懒惰的狗。"

for text in [text_en, text_zh]:
    print(f"\n{'='*50}")
    print(f"Text: {text}")
    analyze_tokenization(text, "gpt2")
    # analyze_tokenization(text, "bert-base-chinese")
```

### 6.3 处理特殊情况

```python
# 处理代码
code = """
def hello():
    print("Hello, World!")
"""

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize(code)
print(f"代码 tokens: {tokens}")
# 注意：缩进和换行也会被 tokenize

# 处理 URL
url = "https://www.example.com/path?param=value"
tokens = tokenizer.tokenize(url)
print(f"URL tokens: {tokens}")
# URL 通常会被拆成很多 token

# 处理数字
numbers = "The year is 2024 and pi is 3.14159"
tokens = tokenizer.tokenize(numbers)
print(f"数字 tokens: {tokens}")
# 数字可能每位一个 token
```

---

## 7. 练习题

### 基础练习

1. 手动实现 BPE 训练和分词
2. 比较不同模型的 Tokenizer 在中英文上的表现
3. 估算一段文本的 API 成本

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. BPE 实现见上文

# 2. Tokenizer 对比
from transformers import AutoTokenizer

def compare_tokenizers(text):
    models = [
        ("GPT-2", "gpt2"),
        ("BERT", "bert-base-uncased"),
        ("BERT-Chinese", "bert-base-chinese"),
    ]

    print(f"Text: {text}")
    print("-" * 50)

    for name, model_name in models:
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            tokens = tok.tokenize(text)
            ids = tok.encode(text)
            print(f"{name}:")
            print(f"  Tokens: {len(tokens)}")
            print(f"  {tokens[:10]}...")
        except:
            print(f"{name}: 加载失败")
        print()

# 英文测试
compare_tokenizers("Hello, how are you doing today?")

# 中文测试
compare_tokenizers("你好，今天天气怎么样？")


# 3. API 成本估算
def estimate_cost(text, model="gpt-4"):
    """估算 API 调用成本"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 近似
    num_tokens = len(tokenizer.encode(text))

    # 价格（美元/1K tokens）
    prices = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }

    price = prices.get(model, {"input": 0.01, "output": 0.01})

    # 假设输出和输入一样长
    input_cost = num_tokens / 1000 * price["input"]
    output_cost = num_tokens / 1000 * price["output"]

    print(f"文本: {text[:50]}...")
    print(f"Token 数: {num_tokens}")
    print(f"模型: {model}")
    print(f"  输入成本: ${input_cost:.4f}")
    print(f"  输出成本: ${output_cost:.4f}")
    print(f"  总成本: ${input_cost + output_cost:.4f}")

# 测试
long_text = "这是一段很长的文本..." * 100
estimate_cost(long_text, "gpt-4")
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [07-HuggingFace生态.md](./07-HuggingFace生态.md)

