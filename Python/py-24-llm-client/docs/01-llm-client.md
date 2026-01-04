# LLM 客户端

## 概述

LLM 客户端提供统一的接口访问各种大语言模型。

## 1. 设计原则

### 1.1 不绑定厂商

```python
# 使用 httpx 而非官方 SDK
import httpx

class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
```

**优势**：
- 可切换到兼容 API（Azure、本地模型）
- 减少依赖
- 完全控制请求/响应

### 1.2 幂等请求 ID

```python
import uuid

def chat(self, messages, request_id=None):
    request_id = request_id or str(uuid.uuid4())
    
    # 用于日志追踪和重试去重
    logger.info("request_started", request_id=request_id)
```

## 2. 基础使用

### 2.1 创建客户端

```python
from llm_kit.client import OpenAIClient

# OpenAI
client = OpenAIClient(api_key="sk-...")

# Azure
client = OpenAIClient(
    api_key="...",
    base_url="https://xxx.openai.azure.com/openai/deployments/gpt-4",
)

# 本地模型（Ollama）
client = OpenAIClient(
    api_key="not-needed",
    base_url="http://localhost:11434/v1",
)
```

### 2.2 聊天补全

```python
response = client.chat(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ],
    model="gpt-4o-mini",
    temperature=0.7,
)

print(response.content)
print(f"Tokens: {response.usage.total_tokens}")
print(f"Cost: ${response.usage.estimated_cost:.6f}")
```

## 3. 超时和重试

### 3.1 配置

```python
client = OpenAIClient(
    api_key="...",
    timeout=60.0,      # 请求超时
    max_retries=3,     # 最大重试次数
)
```

### 3.2 自动重试

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(3),
)
def chat_with_retry(client, messages):
    return client.chat(messages)
```

## 4. 成本统计

```python
# 单次请求
response = client.chat(messages)
print(f"This request: {response.usage.estimated_cost:.6f}")

# 累计统计
stats = client.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total cost: ${stats['total_cost']:.4f}")
```

## 5. 错误处理

```python
from llm_kit.client.base import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)

try:
    response = client.chat(messages)
except RateLimitError:
    # 等待重试
    time.sleep(60)
except AuthenticationError:
    # 检查 API key
    pass
except InvalidRequestError as e:
    # 检查请求参数
    print(f"Invalid request: {e}")
```

## 6. 最佳实践

1. **使用连接池**：复用 httpx 客户端
2. **设置合理超时**：防止请求挂起
3. **实现幂等重试**：使用 request_id
4. **监控成本**：定期检查统计
5. **错误处理**：区分可重试和不可重试错误


