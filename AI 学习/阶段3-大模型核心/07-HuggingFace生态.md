# 🤗 07 - Hugging Face 生态

> Hugging Face 是 NLP/LLM 的核心工具库，掌握它能大幅提升开发效率

---

## 目录

1. [Hugging Face 概览](#1-hugging-face-概览)
2. [Transformers 库](#2-transformers-库)
3. [Datasets 库](#3-datasets-库)
4. [Model Hub](#4-model-hub)
5. [Pipeline 快速推理](#5-pipeline-快速推理)
6. [模型微调](#6-模型微调)
7. [练习题](#7-练习题)

---

## 1. Hugging Face 概览

### 1.1 核心组件

```
Hugging Face 生态：

1. transformers - 预训练模型库
   - 加载各种 Transformer 模型
   - BERT, GPT, LLaMA, T5, Whisper...

2. datasets - 数据集库
   - 加载公开数据集
   - 高效数据处理

3. Model Hub - 模型仓库
   - 存储/分享模型
   - 超过 50 万个模型

4. tokenizers - 高性能分词器
   - Rust 实现，速度快

5. accelerate - 分布式训练
   - 简化多 GPU 训练

6. PEFT - 高效微调
   - LoRA, Prefix Tuning
```

### 1.2 安装

```bash
# 核心库
pip install transformers datasets tokenizers

# 可选
pip install accelerate  # 加速/分布式
pip install peft       # 高效微调
pip install evaluate   # 评估指标
pip install sentencepiece  # 某些模型需要
```

---

## 2. Transformers 库

### 2.1 Auto 类（推荐方式）

```python
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoConfig
)

# 自动根据模型名加载正确的类
model_name = "bert-base-uncased"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型（基础版，不带任务头）
model = AutoModel.from_pretrained(model_name)

# 加载分类模型
classifier = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 加载生成模型
generator = AutoModelForCausalLM.from_pretrained("gpt2")

# 加载 MLM 模型
mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
```

### 2.2 模型推理

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 加载
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 准备输入
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
print(f"Inputs: {inputs}")

# 推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# 输出
print(f"Last hidden state: {outputs.last_hidden_state.shape}")
# [1, seq_len, 768]

print(f"Pooler output: {outputs.pooler_output.shape}")
# [1, 768] - [CLS] token 的输出

# 获取词向量
word_embeddings = outputs.last_hidden_state
print(f"第一个 token 的向量: {word_embeddings[0, 0, :5]}")
```

### 2.3 生成文本

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 准备输入
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_return_sequences=1,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# 解码
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### 2.4 生成参数详解

```python
# 生成策略参数
generation_config = {
    # 基础参数
    "max_new_tokens": 100,       # 最大生成 token 数
    "min_new_tokens": 10,        # 最小生成 token 数

    # 采样参数
    "do_sample": True,           # 是否采样（False=贪婪）
    "temperature": 0.7,          # 温度（越高越随机）
    "top_k": 50,                 # Top-K 采样
    "top_p": 0.9,                # Top-P (Nucleus) 采样

    # 重复控制
    "repetition_penalty": 1.1,   # 重复惩罚
    "no_repeat_ngram_size": 3,   # 禁止重复的 n-gram 大小

    # 束搜索
    "num_beams": 5,              # 束宽度
    "num_return_sequences": 3,   # 返回序列数

    # 停止条件
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

outputs = model.generate(**inputs, **generation_config)
```

---

## 3. Datasets 库

### 3.1 加载数据集

```python
from datasets import load_dataset

# 加载内置数据集
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 查看样本
print(dataset['train'][0])

# 加载数据集的子集
dataset = load_dataset("imdb", split="train[:1000]")

# 从 CSV 加载
# dataset = load_dataset("csv", data_files="data.csv")

# 从 JSON 加载
# dataset = load_dataset("json", data_files="data.json")

# 加载中文数据集
# dataset = load_dataset("nlpcc/c3")
```

### 3.2 数据处理

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载数据
dataset = load_dataset("imdb", split="train[:1000]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义处理函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# 应用处理（批量处理，速度快）
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # 删除原始文本列
)

print(tokenized_dataset[0].keys())
# dict_keys(['label', 'input_ids', 'token_type_ids', 'attention_mask'])

# 设置格式为 PyTorch
tokenized_dataset.set_format("torch")
```

### 3.3 数据集操作

```python
# 过滤
positive_reviews = dataset.filter(lambda x: x["label"] == 1)

# 排序
sorted_dataset = dataset.sort("label")

# 打乱
shuffled = dataset.shuffle(seed=42)

# 选择列
text_only = dataset.select_columns(["text"])

# 切分
train_test = dataset.train_test_split(test_size=0.2)
print(train_test)
# DatasetDict({
#     train: Dataset({...})
#     test: Dataset({...})
# })

# 合并
from datasets import concatenate_datasets
combined = concatenate_datasets([dataset1, dataset2])
```

---

## 4. Model Hub

### 4.1 浏览和搜索

```python
from huggingface_hub import HfApi, list_models

# 搜索模型
api = HfApi()
models = api.list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=5
)

for model in models:
    print(f"{model.id}: {model.downloads} downloads")

# 搜索特定模型
# models = api.list_models(search="bert-base")
```

### 4.2 下载和使用

```python
from transformers import AutoModel, AutoTokenizer

# 从 Hub 加载（自动下载并缓存）
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 指定缓存目录
model = AutoModel.from_pretrained(
    model_name,
    cache_dir="./model_cache"
)

# 加载特定版本
model = AutoModel.from_pretrained(
    model_name,
    revision="v1.0"  # 或 commit hash
)

# 信任远程代码（某些模型需要）
model = AutoModel.from_pretrained(
    "some-model",
    trust_remote_code=True
)
```

### 4.3 上传模型

```python
from huggingface_hub import HfApi, login

# 登录
login()  # 或设置 HF_TOKEN 环境变量

# 上传模型
api = HfApi()

# 方法 1：使用 push_to_hub
model.push_to_hub("my-username/my-model-name")
tokenizer.push_to_hub("my-username/my-model-name")

# 方法 2：使用 API
api.upload_folder(
    folder_path="./my-model",
    repo_id="my-username/my-model-name",
    repo_type="model"
)
```

---

## 5. Pipeline 快速推理

### 5.1 各种 Pipeline

```python
from transformers import pipeline

# 文本分类
classifier = pipeline("text-classification")
result = classifier("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# 情感分析（指定模型）
sentiment = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
print(sentiment("This product is amazing!"))

# 命名实体识别
ner = pipeline("ner", grouped_entities=True)
print(ner("My name is John and I work at Google in New York."))

# 问答
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is the capital of France."
)
print(result)

# 文本生成
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time", max_length=50))

# 填空
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
print(fill_mask("The capital of France is [MASK]."))

# 摘要
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = "..." # 长文本
print(summarizer(text, max_length=130, min_length=30))

# 翻译
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("Hello, how are you?"))

# 零样本分类
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a tutorial about NLP",
    candidate_labels=["technology", "sports", "politics"]
)
print(result)
```

### 5.2 批量处理

```python
# Pipeline 支持批量处理
classifier = pipeline("text-classification", device=0)  # GPU

texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
]

# 批量推理
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text}: {result}")

# 使用 Dataset
from datasets import load_dataset
dataset = load_dataset("imdb", split="test[:100]")
results = classifier(dataset["text"])
```

---

## 6. 模型微调

### 6.1 使用 Trainer

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 加载数据和模型
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # 演示用小数据
    eval_dataset=tokenized_datasets["test"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练
trainer.train()

# 评估
results = trainer.evaluate()
print(results)

# 保存
trainer.save_model("./my_fine_tuned_model")
```

### 6.2 手动训练循环

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
from tqdm import tqdm

# 准备数据
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 创建 DataLoader
train_dataloader = DataLoader(
    tokenized_datasets["train"].select(range(1000)),
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator
)

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(3):
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_dataloader):.4f}")
```

---

## 7. 练习题

### 基础练习

1. 使用 pipeline 实现一个情感分析服务
2. 加载 IMDB 数据集并用 BERT 微调
3. 用 GPT-2 生成一段故事

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. 情感分析服务
from transformers import pipeline

def sentiment_service():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    while True:
        text = input("输入文本（q 退出）: ")
        if text.lower() == 'q':
            break
        result = classifier(text)[0]
        print(f"情感: {result['label']}, 置信度: {result['score']:.4f}")

# sentiment_service()


# 2. BERT 微调（简化版）
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 加载
dataset = load_dataset("imdb", split="train[:500]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 处理
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

# 训练
args = TrainingArguments(
    output_dir="./bert-imdb",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# trainer.train()


# 3. GPT-2 故事生成
from transformers import pipeline

def generate_story(prompt, max_length=200):
    generator = pipeline("text-generation", model="gpt2")

    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    return result[0]['generated_text']

story = generate_story("In a magical forest, there lived a")
print(story)
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [08-Embedding与向量检索.md](./08-Embedding与向量检索.md)

