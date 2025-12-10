# 🦙 Ollama 入门

> 最简单的本地 LLM 部署方案

---

## Ollama 简介

```
Ollama 是什么：
- 本地运行大模型的工具
- 类似 Docker 的模型管理
- 开箱即用，无需复杂配置
- 支持 macOS / Linux / Windows

特点：
✅ 安装简单（一行命令）
✅ 模型库丰富
✅ 自动管理模型文件
✅ 提供 REST API
✅ 支持 GPU 加速
```

---

## 安装与配置

### 安装

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS (Homebrew)
brew install ollama

# Windows
# 从 https://ollama.com/download 下载安装包

# 验证安装
ollama --version
```

### 启动服务

```bash
# 前台运行
ollama serve

# 后台运行（Linux）
sudo systemctl start ollama
sudo systemctl enable ollama  # 开机自启

# 检查状态
ollama list
```

### 配置（可选）

```bash
# 环境变量配置
export OLLAMA_HOST=0.0.0.0:11434  # 监听地址
export OLLAMA_MODELS=/path/to/models  # 模型存储路径
export OLLAMA_NUM_PARALLEL=4  # 并行请求数
export OLLAMA_MAX_LOADED_MODELS=2  # 最大加载模型数
```

---

## 模型管理

### 下载模型

```bash
# 下载模型（自动选择合适的量化版本）
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

# 指定版本
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:70b-instruct-q4_K_M

# 常用模型
ollama pull llama3.1       # Meta Llama 3.1
ollama pull qwen2.5        # 通义千问
ollama pull mistral        # Mistral AI
ollama pull codellama      # 代码生成
ollama pull nomic-embed-text  # Embedding
```

### 管理命令

```bash
# 查看已下载的模型
ollama list

# 查看模型详情
ollama show qwen2.5:7b

# 删除模型
ollama rm qwen2.5:7b

# 复制模型（创建别名）
ollama cp qwen2.5:7b my-model
```

### 模型库

```
官方模型库：https://ollama.com/library

热门模型：
- llama3.1: 8b/70b/405b
- qwen2.5: 0.5b/1.5b/3b/7b/14b/32b/72b
- mistral: 7b
- phi3: 3.8b/14b
- gemma2: 2b/9b/27b
- codellama: 7b/13b/34b
- deepseek-coder: 1.3b/6.7b/33b
```

---

## 命令行使用

### 交互对话

```bash
# 启动对话
ollama run qwen2.5:7b

>>> 你好，请介绍一下自己
我是通义千问，一个由阿里云开发的AI助手...

>>> /bye  # 退出
```

### 单次调用

```bash
# 直接获取回答
ollama run qwen2.5:7b "什么是机器学习？"

# 从文件读取
ollama run qwen2.5:7b < prompt.txt

# 输出到文件
ollama run qwen2.5:7b "写一首诗" > output.txt
```

### 参数设置

```bash
# 设置温度和 token 数
ollama run qwen2.5:7b --temperature 0.7 --num-predict 200 "讲个故事"

# 设置系统提示
ollama run qwen2.5:7b --system "你是一个专业的Python教师"
```

---

## REST API

### 基本调用

```bash
# 生成（非流式）
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "什么是人工智能？",
  "stream": false
}'

# 生成（流式）
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "什么是人工智能？",
  "stream": true
}'

# 对话
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}'
```

### Python 调用

```python
import requests
import json

OLLAMA_URL = "http://localhost:11434"

def generate(prompt: str, model: str = "qwen2.5:7b") -> str:
    """非流式生成"""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def generate_stream(prompt: str, model: str = "qwen2.5:7b"):
    """流式生成"""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done"):
                yield data["response"]

def chat(messages: list, model: str = "qwen2.5:7b") -> str:
    """对话"""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# 使用
print(generate("1+1等于几？"))

for chunk in generate_stream("写一首短诗"):
    print(chunk, end="", flush=True)

result = chat([
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    {"role": "user", "content": "今天天气怎么样？"}
])
print(result)
```

### Embedding

```python
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """获取文本 embedding"""
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": model,
            "prompt": text
        }
    )
    return response.json()["embedding"]

embedding = get_embedding("Hello, world!")
print(f"维度: {len(embedding)}")  # 768
```

---

## OpenAI 兼容接口

```
Ollama 提供 OpenAI 兼容的 API：
- /v1/chat/completions
- /v1/completions
- /v1/embeddings

可以直接使用 openai SDK！
```

```python
from openai import OpenAI

# 指向 Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意值即可
)

# 对话
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "讲个故事"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 自定义 Modelfile

### 创建自定义模型

```dockerfile
# Modelfile
FROM qwen2.5:7b

# 设置参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

# 设置系统提示
SYSTEM """
你是一个专业的Python编程助手。
- 回答要简洁准确
- 代码要有注释
- 遇到不确定的问题要说明
"""

# 添加模板（可选）
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

```bash
# 构建模型
ollama create python-assistant -f Modelfile

# 运行
ollama run python-assistant "如何读取CSV文件？"

# 查看
ollama show python-assistant
```

### 导入 GGUF 模型

```dockerfile
# 从本地 GGUF 文件创建
FROM ./my-model.gguf

PARAMETER temperature 0.7
SYSTEM "你是一个AI助手"
```

```bash
ollama create my-model -f Modelfile
ollama run my-model
```

---

## 最佳实践

```
1. 模型选择
   - 开发测试：小模型（1.5B-7B）
   - 生产：根据需求选择（7B-72B）
   - Embedding：nomic-embed-text

2. 性能优化
   - 使用 SSD 存储模型
   - 预热模型（首次加载较慢）
   - 合理设置 num_ctx

3. 资源管理
   - 限制加载的模型数量
   - 监控显存使用
   - 定期清理不用的模型

4. 生产部署
   - 配置为系统服务
   - 设置合适的监听地址
   - 添加反向代理（Nginx）
```

---

## ➡️ 下一步

继续 [05-vLLM高吞吐.md](./05-vLLM高吞吐.md)

