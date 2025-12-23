# 🌐 06 - FastAPI 服务化

> 封装 LLM API 与流式输出

---

## FastAPI 基础

### 最小示例

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM API")

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 调用 LLM
    response = "这是一个示例回复"
    return ChatResponse(response=response)

# 运行: uvicorn main:app --reload
```

---

## 集成 LLM

### 集成 Ollama

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx

app = FastAPI(title="LLM Chat API")

OLLAMA_URL = "http://localhost:11434"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "qwen2.5:7b"
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    content: str
    model: str
    usage: dict

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI 兼容的 Chat API"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": request.model,
                "messages": [m.model_dump() for m in request.messages],
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            },
            timeout=60.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM service error")

        data = response.json()

        return {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": data["message"]["content"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
        }
```

### 集成 vLLM

```python
from openai import AsyncOpenAI

VLLM_URL = "http://localhost:8000/v1"
vllm_client = AsyncOpenAI(base_url=VLLM_URL, api_key="dummy")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """转发到 vLLM"""

    response = await vllm_client.chat.completions.create(
        model=request.model,
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    return response.model_dump()
```

---

## 流式输出（SSE）

### 实现流式响应

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json
import httpx
import asyncio

app = FastAPI()

async def stream_chat(messages: list, model: str):
    """流式生成器"""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True
            },
            timeout=60.0
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        # OpenAI 格式
                        chunk = {
                            "id": "chatcmpl-xxx",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": data["message"]["content"]
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # 结束标记
                        chunk = {
                            "id": "chatcmpl-xxx",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """支持流式和非流式"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            stream_chat(messages, request.model),
            media_type="text/event-stream"
        )
    else:
        # 非流式处理...
        pass
```

### 使用 SSE-Starlette

```python
from sse_starlette.sse import EventSourceResponse

async def event_generator(messages: list, model: str):
    """SSE 事件生成器"""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "content": data.get("message", {}).get("content", ""),
                            "done": data.get("done", False)
                        })
                    }

@app.post("/stream")
async def stream_endpoint(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    return EventSourceResponse(event_generator(messages, request.model))
```

### 客户端消费流式响应

```python
import httpx

async def consume_stream():
    """消费 SSE 流"""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "写一首诗"}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    print(content, end="", flush=True)

# 使用 openai SDK
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

async def stream_with_sdk():
    stream = await client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "写一首诗"}],
        stream=True
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

---

## 完整 API 服务

```python
"""完整的 LLM API 服务"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import httpx
import json
import time
import uuid

app = FastAPI(
    title="LLM API Service",
    description="OpenAI 兼容的 LLM API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
LLM_BACKEND = "http://localhost:11434"  # Ollama
# LLM_BACKEND = "http://localhost:8000/v1"  # vLLM

# ========== 数据模型 ==========
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    top_p: float = Field(default=1.0, ge=0, le=1)

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# ========== API 端点 ==========
@app.get("/")
async def root():
    return {"message": "LLM API Service", "version": "1.0.0"}

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {"id": "qwen2.5:7b", "object": "model", "owned_by": "local"},
            {"id": "llama3.1:8b", "object": "model", "owned_by": "local"},
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat Completions API"""

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _stream_response(messages, request),
            media_type="text/event-stream"
        )
    else:
        return await _non_stream_response(messages, request)

async def _non_stream_response(messages: list, request: ChatCompletionRequest):
    """非流式响应"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LLM_BACKEND}/api/chat",
            json={
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": request.top_p
                }
            },
            timeout=120.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Backend error")

        data = response.json()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=data["message"]["content"]
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            )
        )

async def _stream_response(messages: list, request: ChatCompletionRequest):
    """流式响应"""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{LLM_BACKEND}/api/chat",
            json={
                "model": request.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": request.top_p
                }
            },
            timeout=120.0
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)

                    if not data.get("done"):
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": data["message"]["content"]},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"

# ========== 健康检查 ==========
@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LLM_BACKEND}/api/tags", timeout=5.0)
            if response.status_code == 200:
                return {"status": "healthy", "backend": "connected"}
    except:
        pass
    return {"status": "unhealthy", "backend": "disconnected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## 运行与测试

```bash
# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# 测试非流式
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "你好"}]}'

# 测试流式
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "你好"}], "stream": true}'
```

---

## ➡️ 下一步

继续 [07-认证与限流.md](./07-认证与限流.md)

