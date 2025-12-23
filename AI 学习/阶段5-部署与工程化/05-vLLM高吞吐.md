# 🚀 05 - vLLM 高吞吐推理

> 生产级 LLM 推理引擎

---

## vLLM 简介

```
vLLM 是什么：
- 高性能 LLM 推理和服务引擎
- 由 UC Berkeley 开发
- 生产环境首选

核心特性：
✅ PagedAttention 高效内存管理
✅ Continuous Batching 高吞吐
✅ 支持多种量化格式
✅ OpenAI 兼容 API
✅ 支持多 GPU 并行
✅ 支持 LoRA 热加载
```

---

## 安装

```bash
# 基础安装（需要 CUDA）
pip install vllm

# 特定 CUDA 版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

---

## 快速开始

### Python API

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# 生成
prompts = [
    "什么是人工智能？",
    "Python 有什么特点？",
    "解释机器学习的基本原理"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Output: {generated_text}")
    print("-" * 50)
```

### 对话格式

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

def chat(messages: list, **kwargs) -> str:
    """对话函数"""
    # 转换为模型格式
    prompt = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 512)
    )

    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# 使用
response = chat([
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"}
])
print(response)
```

---

## 启动 API 服务

### 基本启动

```bash
# 启动 OpenAI 兼容服务器
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code

# 简化命令
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

### 高级配置

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --api-key "your-secret-key"
```

### 参数说明

```
常用参数：
--model                    模型路径或名称
--host                     监听地址
--port                     监听端口
--tensor-parallel-size     GPU 并行数
--gpu-memory-utilization   显存使用比例（0-1）
--max-model-len            最大序列长度
--max-num-seqs             最大并发请求数
--dtype                    数据类型（auto/float16/bfloat16）
--quantization             量化方式（awq/gptq/squeezellm）
--enable-prefix-caching    启用前缀缓存
--api-key                  API 密钥
```

---

## API 调用

### 使用 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"
)

# Chat Completions
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "介绍一下Python"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 直接 HTTP 调用

```bash
# Chat
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-secret-key" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.7
    }'

# 获取模型列表
curl http://localhost:8000/v1/models
```

---

## 高级功能

### 量化模型

```bash
# AWQ 量化
vllm serve TheBloke/Qwen-7B-AWQ \
    --quantization awq \
    --dtype float16

# GPTQ 量化
vllm serve TheBloke/Qwen-7B-GPTQ \
    --quantization gptq \
    --dtype float16
```

### 多 GPU 部署

```bash
# 张量并行（需要多 GPU）
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9

# 流水线并行（大模型）
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 2
```

### LoRA 热加载

```bash
# 启动时指定 LoRA
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules my-lora=./lora-weights

# 动态加载（通过 API）
```

```python
# 调用特定 LoRA
response = client.chat.completions.create(
    model="my-lora",  # 使用 LoRA 名称
    messages=[{"role": "user", "content": "你好"}]
)
```

### 前缀缓存

```bash
# 启用前缀缓存（适合固定前缀场景）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-prefix-caching
```

---

## 性能优化

### 显存优化

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.95,  # 提高利用率
    max_num_seqs=128,             # 减少并发
    max_model_len=4096,           # 限制长度
    enforce_eager=True,           # 禁用 CUDA Graph（省显存）
)
```

### 吞吐优化

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.9,
    max_num_seqs=256,             # 增加并发
    enable_chunked_prefill=True,  # 分块预填充
    max_num_batched_tokens=4096,  # 批处理 token 数
)
```

### 延迟优化

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=32,              # 减少并发
    speculative_model="Qwen/Qwen2.5-0.5B-Instruct",  # 推测解码
    num_speculative_tokens=5,
)
```

---

## 监控

### 内置指标

```bash
# vLLM 暴露 Prometheus 指标
curl http://localhost:8000/metrics

# 常见指标：
# vllm:num_requests_running - 运行中的请求
# vllm:num_requests_waiting - 等待中的请求
# vllm:gpu_cache_usage_perc - GPU 缓存使用率
# vllm:cpu_cache_usage_perc - CPU 缓存使用率
```

### Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
```

---

## 与 Ollama 对比

| 特性 | Ollama | vLLM |
|------|--------|------|
| 安装难度 | 简单 | 需要 CUDA |
| CPU 支持 | ✅ | ❌ |
| 吞吐量 | 一般 | 很高 |
| 显存效率 | 一般 | 很高 |
| 并发支持 | 有限 | 很强 |
| 量化支持 | GGUF | AWQ/GPTQ |
| 适用场景 | 本地开发 | 生产部署 |

---

## ➡️ 下一步

继续 [06-FastAPI服务.md](./06-FastAPI服务.md)

