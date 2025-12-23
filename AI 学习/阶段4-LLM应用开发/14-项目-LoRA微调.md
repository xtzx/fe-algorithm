# 🔧 14 - 项目：LoRA 微调实战

> 使用 LoRA 微调开源模型，使其擅长特定领域

---

## 项目目标

```
目标：微调 Qwen 模型，使其成为一个专业的代码助手

具体要求：
1. 准备代码问答数据集
2. 使用 QLoRA 进行微调
3. 评估微调效果
4. 部署微调后的模型
```

---

## 完整流程

### 1. 环境准备

```bash
# 创建环境
conda create -n lora-finetune python=3.11 -y
conda activate lora-finetune

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate bitsandbytes
pip install trl wandb
```

### 2. 数据准备

```python
"""prepare_data.py - 准备训练数据"""
import json
from datasets import Dataset, load_dataset

def create_code_qa_dataset():
    """创建代码问答数据集"""

    # 方法 1：从现有数据集加载
    # dataset = load_dataset("sahil2801/CodeAlpaca-20k")

    # 方法 2：自定义数据
    data = [
        {
            "instruction": "写一个 Python 函数计算斐波那契数列",
            "input": "",
            "output": """def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# 使用示例
print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"""
        },
        {
            "instruction": "解释 Python 中的装饰器",
            "input": "",
            "output": """装饰器是一种用于修改函数或类行为的设计模式。

基本概念：
1. 装饰器是一个接受函数作为参数的函数
2. 返回一个新的函数，通常包含原函数的功能加上额外的行为
3. 使用 @ 语法糖来应用装饰器

示例：
```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"执行时间: {time.time() - start:.2f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()  # 输出: 执行时间: 1.00秒
```

常见用途：
- 日志记录
- 性能计时
- 访问控制
- 缓存"""
        },
        {
            "instruction": "实现一个 Python 单例模式",
            "input": "",
            "output": """单例模式确保一个类只有一个实例。

方法 1：使用装饰器
```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self, url):
        self.url = url

db1 = Database("localhost")
db2 = Database("remote")
print(db1 is db2)  # True
```

方法 2：使用元类
```python
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=SingletonMeta):
    def __init__(self):
        self.logs = []

log1 = Logger()
log2 = Logger()
print(log1 is log2)  # True
```"""
        },
        # ... 更多数据
    ]

    # 扩展数据集（实际应用中需要更多数据）
    extended_data = data * 100  # 简单复制，实际需要多样化数据

    return Dataset.from_list(extended_data)


def format_instruction(example):
    """格式化为 Qwen 对话格式"""
    if example["input"]:
        prompt = f"""<|im_start|>user
{example["instruction"]}

{example["input"]}<|im_end|>
<|im_start|>assistant
{example["output"]}<|im_end|>"""
    else:
        prompt = f"""<|im_start|>user
{example["instruction"]}<|im_end|>
<|im_start|>assistant
{example["output"]}<|im_end|>"""

    return {"text": prompt}


if __name__ == "__main__":
    # 创建数据集
    dataset = create_code_qa_dataset()

    # 格式化
    dataset = dataset.map(format_instruction)

    # 分割
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # 保存
    dataset.save_to_disk("./code_qa_dataset")

    print(f"训练集: {len(dataset['train'])} 条")
    print(f"测试集: {len(dataset['test'])} 条")
    print(f"\n示例:\n{dataset['train'][0]['text'][:500]}...")
```

### 3. 训练脚本

```python
"""train.py - LoRA 微调训练"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk
import wandb

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./qwen-code-assistant"
MAX_SEQ_LENGTH = 2048

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 训练配置
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# ========== 加载模型 ==========
print("加载模型...")

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 准备模型用于 k-bit 训练
model = prepare_model_for_kbit_training(model)

# ========== 配置 LoRA ==========
print("配置 LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 加载数据 ==========
print("加载数据...")
dataset = load_from_disk("./code_qa_dataset")

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="wandb",  # 或 "none"
    run_name="qwen-code-assistant",
)

# ========== 创建 Trainer ==========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)

# ========== 训练 ==========
print("开始训练...")
trainer.train()

# ========== 保存模型 ==========
print("保存模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"训练完成！模型保存到 {OUTPUT_DIR}")
```

### 4. 合并和导出

```python
"""merge_model.py - 合并 LoRA 权重"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./qwen-code-assistant"
OUTPUT_PATH = "./qwen-code-assistant-merged"

print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("加载 LoRA 权重...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("合并权重...")
model = model.merge_and_unload()

print("保存合并后的模型...")
model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"完成！模型保存到 {OUTPUT_PATH}")
```

### 5. 推理测试

```python
"""inference.py - 推理测试"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "./qwen-code-assistant-merged"

# 加载模型
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_response(instruction: str, max_length: int = 512):
    """生成回答"""
    prompt = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取助手回答
    response = response.split("<|im_start|>assistant")[-1].strip()

    return response


# 测试
test_questions = [
    "写一个 Python 快速排序函数",
    "如何在 Python 中实现多线程？",
    "解释 Python 的 GIL 是什么",
    "写一个简单的 REST API 使用 FastAPI"
]

print("\n" + "="*60)
print("推理测试")
print("="*60)

for q in test_questions:
    print(f"\n问题: {q}")
    print("-" * 40)
    response = generate_response(q)
    print(f"回答:\n{response}")
    print("="*60)
```

### 6. 评估脚本

```python
"""evaluate.py - 评估微调效果"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

def evaluate_model(model_path: str, test_data: list):
    """评估模型"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    results = []

    for item in test_data:
        prompt = f"""<|im_start|>user
{item['instruction']}<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # 低温度用于评估
            do_sample=False
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|im_start|>assistant")[-1].strip()

        results.append({
            "instruction": item["instruction"],
            "expected": item["output"][:200],
            "generated": response[:200]
        })

    return results


# 简单的评估指标
def calculate_metrics(results):
    """计算评估指标"""
    # 这里简化处理，实际可以使用更复杂的指标
    # 如 BLEU、ROUGE、代码执行正确率等

    total = len(results)

    # 计算响应长度
    avg_length = sum(len(r["generated"]) for r in results) / total

    # 计算包含代码的比例
    code_ratio = sum(1 for r in results if "```" in r["generated"] or "def " in r["generated"]) / total

    return {
        "total_samples": total,
        "avg_response_length": avg_length,
        "code_ratio": code_ratio
    }


if __name__ == "__main__":
    # 测试数据
    test_data = [
        {"instruction": "写一个计算阶乘的函数", "output": "def factorial(n)..."},
        {"instruction": "如何读取 JSON 文件", "output": "import json..."},
        {"instruction": "实现二分查找", "output": "def binary_search..."},
    ]

    # 评估原始模型
    print("评估原始模型...")
    base_results = evaluate_model("Qwen/Qwen2.5-1.5B-Instruct", test_data)
    base_metrics = calculate_metrics(base_results)

    # 评估微调模型
    print("评估微调模型...")
    finetuned_results = evaluate_model("./qwen-code-assistant-merged", test_data)
    finetuned_metrics = calculate_metrics(finetuned_results)

    # 对比
    print("\n评估结果对比:")
    print(f"{'指标':<20} {'原始模型':<15} {'微调模型':<15}")
    print("-" * 50)
    for key in base_metrics:
        print(f"{key:<20} {base_metrics[key]:<15.2f} {finetuned_metrics[key]:<15.2f}")
```

---

## 使用 LLaMA-Factory

```yaml
# llama_factory_config.yaml
### Model
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32

### Dataset
dataset: code_alpaca  # 或自定义数据集
template: qwen
cutoff_len: 2048

### Output
output_dir: saves/qwen-code-lora
logging_steps: 10
save_steps: 100

### Train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

### Eval
val_size: 0.1
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100
```

```bash
# 训练
llamafactory-cli train llama_factory_config.yaml

# 推理测试
llamafactory-cli chat \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path saves/qwen-code-lora \
    --template qwen

# 导出
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path saves/qwen-code-lora \
    --template qwen \
    --export_dir qwen-code-merged
```

---

## 注意事项

```
1. 数据质量
   - 数据要多样化，避免过拟合
   - 至少 1000-10000 条高质量数据
   - 格式要统一

2. 超参数选择
   - LoRA rank: 8-64，越大容量越大
   - learning rate: 1e-5 ~ 5e-4
   - batch size * gradient_accumulation >= 16

3. 监控训练
   - 使用 wandb 或 tensorboard
   - 关注 loss 曲线
   - 定期评估验证集

4. 常见问题
   - OOM: 减少 batch_size，使用 4-bit 量化
   - 过拟合: 减少 epochs，增加 dropout
   - 效果差: 增加数据，提高 rank
```

---

## ➡️ 下一步

继续 [15-自测清单.md](./15-自测清单.md)

