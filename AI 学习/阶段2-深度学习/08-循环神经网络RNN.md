# 📝 08 - 循环神经网络 RNN

> RNN 处理序列数据，LSTM/GRU 解决长期依赖问题

---

## 目录

1. [RNN 基础](#1-rnn-基础)
2. [LSTM](#2-lstm)
3. [GRU](#3-gru)
4. [双向 RNN](#4-双向-rnn)
5. [实战：情感分类](#5-实战情感分类)
6. [练习题](#6-练习题)

---

## 1. RNN 基础

### 1.1 序列数据与 RNN

```
序列数据特点：
- 有先后顺序（文本、时间序列、音频等）
- 不同位置之间有依赖关系

RNN 核心思想：
- 处理每个时间步时，考虑"记忆"（隐藏状态）
- 隐藏状态传递历史信息

h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
y_t = W_hy * h_t + b_y
```

### 1.2 PyTorch RNN

```python
import torch
import torch.nn as nn

# RNN 层
rnn = nn.RNN(
    input_size=10,    # 输入特征维度
    hidden_size=20,   # 隐藏状态维度
    num_layers=2,     # RNN 层数
    batch_first=True, # 输入格式 [batch, seq, feature]
    dropout=0.1,      # 层间 dropout（num_layers > 1 时生效）
    bidirectional=False
)

# 输入：[batch, seq_len, input_size]
x = torch.randn(32, 15, 10)  # 32 个样本，序列长度 15，特征维度 10

# 可选：初始隐藏状态 [num_layers, batch, hidden_size]
h0 = torch.zeros(2, 32, 20)

# 前向传播
output, h_n = rnn(x, h0)
# output: 每个时间步的输出 [batch, seq_len, hidden_size]
# h_n: 最后时间步的隐藏状态 [num_layers, batch, hidden_size]

print(f"Output 形状: {output.shape}")  # [32, 15, 20]
print(f"Hidden 形状: {h_n.shape}")     # [2, 32, 20]
```

### 1.3 手写 RNN Cell

```python
class SimpleRNNCell(nn.Module):
    """单个 RNN 单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入到隐藏
        self.i2h = nn.Linear(input_size, hidden_size)
        # 隐藏到隐藏
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # x: [batch, input_size]
        # h_prev: [batch, hidden_size]
        h_new = torch.tanh(self.i2h(x) + self.h2h(h_prev))
        return h_new

class SimpleRNN(nn.Module):
    """手写 RNN"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = SimpleRNNCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        # x: [batch, seq_len, input_size]
        batch_size, seq_len, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        h = h0

        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)

        # [batch, seq_len, hidden_size]
        outputs = torch.stack(outputs, dim=1)

        return outputs, h

# 测试
rnn = SimpleRNN(10, 20)
x = torch.randn(32, 15, 10)
output, h_n = rnn(x)
print(f"手写 RNN Output: {output.shape}")  # [32, 15, 20]
```

---

## 2. LSTM

### 2.1 LSTM 结构

```
LSTM 解决 RNN 的梯度消失问题，通过"门"机制控制信息流

三个门 + 细胞状态：
- 遗忘门 (f)：决定丢弃多少旧信息
- 输入门 (i)：决定添加多少新信息
- 输出门 (o)：决定输出多少信息
- 细胞状态 (c)：长期记忆

f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
c_t = f_t * c_{t-1} + i_t * c̃_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)
```

### 2.2 PyTorch LSTM

```python
# LSTM 层
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=False
)

x = torch.randn(32, 15, 10)

# 初始状态：(h0, c0)
h0 = torch.zeros(2, 32, 20)  # hidden state
c0 = torch.zeros(2, 32, 20)  # cell state

output, (h_n, c_n) = lstm(x, (h0, c0))

print(f"Output: {output.shape}")  # [32, 15, 20]
print(f"h_n: {h_n.shape}")        # [2, 32, 20]
print(f"c_n: {c_n.shape}")        # [2, 32, 20]
```

### 2.3 手写 LSTM Cell

```python
class LSTMCell(nn.Module):
    """手写 LSTM 单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 合并计算所有门（效率更高）
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state

        # 拼接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=1)

        # 计算所有门
        gates = self.gates(combined)

        # 分割成四个门
        i, f, g, o = gates.chunk(4, dim=1)

        # 激活
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)     # 候选细胞状态
        o = torch.sigmoid(o)  # 输出门

        # 更新细胞状态
        c = f * c_prev + i * g

        # 更新隐藏状态
        h = o * torch.tanh(c)

        return h, c

# 测试
cell = LSTMCell(10, 20)
x = torch.randn(32, 10)
h = torch.zeros(32, 20)
c = torch.zeros(32, 20)
h_new, c_new = cell(x, (h, c))
print(f"LSTM Cell: h={h_new.shape}, c={c_new.shape}")
```

---

## 3. GRU

### 3.1 GRU 结构

```
GRU 简化了 LSTM，只有两个门，没有细胞状态

- 重置门 (r)：控制忽略多少历史信息
- 更新门 (z)：控制保留多少历史信息

z_t = σ(W_z · [h_{t-1}, x_t])
r_t = σ(W_r · [h_{t-1}, x_t])
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

### 3.2 PyTorch GRU

```python
# GRU 层
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.1,
    bidirectional=False
)

x = torch.randn(32, 15, 10)
h0 = torch.zeros(2, 32, 20)

output, h_n = gru(x, h0)

print(f"Output: {output.shape}")  # [32, 15, 20]
print(f"h_n: {h_n.shape}")        # [2, 32, 20]
```

### 3.3 LSTM vs GRU

```
LSTM:
- 3 个门 + 细胞状态
- 参数更多
- 在长序列上可能表现更好

GRU:
- 2 个门，没有细胞状态
- 参数更少，训练更快
- 在较短序列上效果相当

实践建议：
- 先尝试 GRU（更快）
- 如果效果不好，再尝试 LSTM
- 现代 NLP 多用 Transformer
```

---

## 4. 双向 RNN

### 4.1 双向 RNN 原理

```
单向：只看前文
双向：同时看前文和后文

"I love [MASK] learning"
前向：根据 "I love" 预测
后向：根据 "learning" 预测
双向：结合两个方向的信息
```

### 4.2 PyTorch 双向 RNN

```python
# 双向 LSTM
bilstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # 关键参数
)

x = torch.randn(32, 15, 10)

# 双向时，num_directions = 2
# h0: [num_layers * num_directions, batch, hidden_size]
h0 = torch.zeros(4, 32, 20)  # 2 * 2 = 4
c0 = torch.zeros(4, 32, 20)

output, (h_n, c_n) = bilstm(x, (h0, c0))

# output: [batch, seq_len, hidden_size * num_directions]
print(f"Output: {output.shape}")  # [32, 15, 40]
# h_n: [num_layers * num_directions, batch, hidden_size]
print(f"h_n: {h_n.shape}")        # [4, 32, 20]

# 分离前向和后向
# output 在最后一维上拼接：[forward, backward]
forward_output = output[:, :, :20]
backward_output = output[:, :, 20:]

# h_n 交替排列：[layer0_forward, layer0_backward, layer1_forward, layer1_backward]
```

### 4.3 获取句子表示

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为双向

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        output, (h_n, c_n) = self.lstm(embedded)
        # output: [batch, seq_len, hidden*2]
        # h_n: [num_layers*2, batch, hidden]

        # 方法 1：取最后时间步的输出
        # last_output = output[:, -1, :]  # [batch, hidden*2]

        # 方法 2：拼接前向和后向的最终隐藏状态（更常用）
        # h_n[-2]: 最后一层前向的最终状态
        # h_n[-1]: 最后一层后向的最终状态
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden*2]

        # 方法 3：对所有时间步求平均（Mean Pooling）
        # mean_output = output.mean(dim=1)  # [batch, hidden*2]

        out = self.fc(h_concat)  # [batch, num_classes]
        return out

# 测试
model = BiLSTMClassifier(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=64,
    num_classes=2
)

x = torch.randint(0, 10000, (32, 50))  # [batch, seq_len]
y = model(x)
print(f"分类输出: {y.shape}")  # [32, 2]
```

---

## 5. 实战：情感分类

### 5.1 数据准备

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 示例数据
texts = [
    "I love this movie it is great",
    "This film is terrible and boring",
    "Amazing performance by the actors",
    "Worst movie ever do not watch",
    "Highly recommended excellent film",
    "Disappointing and waste of time",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# 构建词表
def build_vocab(texts, min_freq=1):
    word_freq = Counter()
    for text in texts:
        word_freq.update(text.lower().split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts)
print(f"词表大小: {len(vocab)}")

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower().split()

        # 转换为 ID
        ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in text]

        # 截断或填充
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.vocab['<PAD>']] * (self.max_len - len(ids))

        return torch.tensor(ids), torch.tensor(self.labels[idx])

dataset = TextDataset(texts, labels, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### 5.2 模型定义

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.5, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)

        output, (h_n, c_n) = self.lstm(embedded)

        # 拼接双向最终隐藏状态
        h_concat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h_concat = self.dropout(h_concat)

        out = self.fc(h_concat)
        return out

# 创建模型
model = SentimentLSTM(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_dim=32,
    num_classes=2,
    num_layers=1,
    dropout=0.3
)

print(model)
```

### 5.3 训练与评估

```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    loss, acc = train_epoch(model, loader, criterion, optimizer, device)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}')

# 推理
model.eval()
test_text = "this movie is great"
test_ids = [vocab.get(w, vocab['<UNK>']) for w in test_text.split()]
test_ids = test_ids + [vocab['<PAD>']] * (32 - len(test_ids))
test_tensor = torch.tensor([test_ids]).to(device)

with torch.no_grad():
    output = model(test_tensor)
    pred = output.argmax(1).item()
    print(f"'{test_text}' -> {'Positive' if pred == 1 else 'Negative'}")
```

### 5.4 处理变长序列

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def collate_fn(batch):
    """处理变长序列的 collate 函数"""
    texts, labels = zip(*batch)

    # 获取实际长度
    lengths = torch.tensor([len(t) for t in texts])

    # 填充
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return texts_padded, labels, lengths

class LSTMWithPacking(nn.Module):
    """使用 packing 处理变长序列"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Pack（去掉填充，提高效率）
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                       batch_first=True, enforce_sorted=False)

        output, (h_n, c_n) = self.lstm(packed)

        # Unpack（可选，如果需要所有时间步的输出）
        # output, _ = pad_packed_sequence(output, batch_first=True)

        h_concat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h_concat)
```

---

## 6. 练习题

### 基础练习

1. 手写 GRU Cell
2. 用 LSTM 做时间序列预测（如正弦波）
3. 比较 RNN、LSTM、GRU 在同一任务上的表现

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 手写 GRU Cell
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 重置门和更新门
        self.gate = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        # 候选隐藏状态
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)

        # 计算门
        gates = torch.sigmoid(self.gate(combined))
        r, z = gates.chunk(2, dim=1)  # 重置门、更新门

        # 计算候选隐藏状态
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.candidate(combined_reset))

        # 更新隐藏状态
        h_new = (1 - z) * h_prev + z * h_candidate

        return h_new

# 测试
cell = GRUCell(10, 20)
x = torch.randn(32, 10)
h = torch.zeros(32, 20)
h_new = cell(x, h)
print(f"GRU Cell: {h_new.shape}")


# 2. 时间序列预测
import numpy as np

# 生成正弦波数据
t = np.linspace(0, 100, 1000)
data = np.sin(t)

# 准备数据
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), \
           torch.tensor(y, dtype=torch.float32)

seq_len = 50
X, y = create_sequences(data, seq_len)
print(f"X: {X.shape}, y: {y.shape}")

# 简单的 LSTM 预测器
class TimeSeriesLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = TimeSeriesLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    pred = model(X[:100])
    loss = criterion(pred.squeeze(), y[:100])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}: Loss={loss.item():.6f}')
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [09-训练技巧与可视化.md](./09-训练技巧与可视化.md)

