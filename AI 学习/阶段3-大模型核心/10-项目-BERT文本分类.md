# 📝 项目：BERT 文本分类

> 使用 BERT 微调完成文本分类任务

---

## 项目概述

### 任务说明

```
数据集：IMDB 电影评论情感分类
- 25,000 条训练样本
- 25,000 条测试样本
- 二分类：正面/负面

目标：
- 使用预训练 BERT 进行微调
- 达到 90%+ 的准确率
```

---

## 完整代码

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np

# ========== 配置 ==========
class Config:
    model_name = "bert-base-uncased"
    max_length = 256
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    warmup_ratio = 0.1
    weight_decay = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
print(f"使用设备: {config.device}")

# ========== 数据准备 ==========
print("加载数据...")
dataset = load_dataset("imdb")

# 使用部分数据进行演示
train_data = dataset["train"].select(range(5000))
test_data = dataset["test"].select(range(1000))

print(f"训练集: {len(train_data)}")
print(f"测试集: {len(test_data)}")
print(f"样本: {train_data[0]}")

# ========== Tokenizer ==========
tokenizer = BertTokenizer.from_pretrained(config.model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    )

# 处理数据
print("Tokenizing...")
train_encodings = tokenizer(
    train_data["text"],
    padding="max_length",
    truncation=True,
    max_length=config.max_length,
    return_tensors="pt"
)

test_encodings = tokenizer(
    test_data["text"],
    padding="max_length",
    truncation=True,
    max_length=config.max_length,
    return_tensors="pt"
)

# ========== Dataset 类 ==========
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx])
        }

train_dataset = IMDBDataset(train_encodings, train_data["label"])
test_dataset = IMDBDataset(test_encodings, test_data["label"])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2)

# ========== 模型 ==========
print("加载模型...")
model = BertForSequenceClassification.from_pretrained(
    config.model_name,
    num_labels=2
)
model = model.to(config.device)

# ========== 优化器和调度器 ==========
optimizer = AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

total_steps = len(train_loader) * config.num_epochs
warmup_steps = int(total_steps * config.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ========== 训练函数 ==========
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy

# ========== 评估函数 ==========
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

# ========== 训练循环 ==========
print("\n开始训练...")
best_accuracy = 0

for epoch in range(config.num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    print('='*50)

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, config.device
    )
    print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    test_acc, _, _ = evaluate(model, test_loader, config.device)
    print(f"测试 - Accuracy: {test_acc:.4f}")

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "best_bert_model.pth")
        print("✓ 保存最佳模型")

print(f"\n训练完成！最佳测试准确率: {best_accuracy:.4f}")

# ========== 详细评估 ==========
print("\n加载最佳模型进行评估...")
model.load_state_dict(torch.load("best_bert_model.pth"))
accuracy, predictions, true_labels = evaluate(model, test_loader, config.device)

print("\n分类报告:")
print(classification_report(
    true_labels, predictions,
    target_names=["Negative", "Positive"]
))

# ========== 推理示例 ==========
def predict(text, model, tokenizer, device):
    model.eval()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    labels = ["Negative", "Positive"]
    return {
        "prediction": labels[pred],
        "confidence": probs[0][pred].item(),
        "probabilities": {
            "Negative": probs[0][0].item(),
            "Positive": probs[0][1].item()
        }
    }

# 测试推理
test_texts = [
    "This movie is absolutely fantastic! I loved every minute of it.",
    "Terrible film. Complete waste of time and money.",
    "It was okay, nothing special but not bad either."
]

print("\n推理测试:")
print("-" * 50)
for text in test_texts:
    result = predict(text, model, tokenizer, config.device)
    print(f"文本: {text[:50]}...")
    print(f"预测: {result['prediction']} (置信度: {result['confidence']:.4f})")
    print("-" * 50)
```

---

## 使用 Trainer API

```python
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

# 数据处理
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

# 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练
trainer.train()

# 评估
results = trainer.evaluate()
print(f"最终结果: {results}")
```

---

## 优化方向

```python
# 1. 使用更好的预训练模型
model = BertForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 2. 冻结部分层
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False

# 3. 学习率分层
optimizer_grouped_parameters = [
    {"params": model.bert.encoder.layer[-4:].parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4},
]
optimizer = AdamW(optimizer_grouped_parameters)

# 4. 数据增强
# - 回译
# - 同义词替换
# - 随机删除

# 5. 集成学习
# 训练多个模型，投票或平均概率
```

---

## 预期效果

```
训练配置: BERT-base, 3 epochs, lr=2e-5
训练集: 25,000 (完整)
测试集: 25,000 (完整)

预期准确率: 92-94%
```

---

## ➡️ 下一步

继续 [11-项目-语义搜索引擎.md](./11-项目-语义搜索引擎.md)

