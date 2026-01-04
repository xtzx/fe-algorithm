# 流式处理

## 概述

流式输出可以提升用户体验，让用户看到实时生成的内容。

## 1. 基础流式输出

```python
for chunk in client.chat_stream(messages):
    print(chunk.delta, end="", flush=True)
    
    if chunk.is_done:
        print(f"\nFinish reason: {chunk.finish_reason}")
```

## 2. SSE 格式

OpenAI API 使用 Server-Sent Events (SSE) 格式：

```
data: {"choices": [{"delta": {"content": "Hello"}}]}

data: {"choices": [{"delta": {"content": " world"}}]}

data: [DONE]
```

解析：

```python
for line in response.iter_lines():
    if line.startswith("data: "):
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        data = json.loads(data_str)
        delta = data["choices"][0]["delta"].get("content", "")
        yield delta
```

## 3. 流式处理器

```python
from llm_kit.client import StreamProcessor

processor = StreamProcessor()

for chunk in client.chat_stream(messages):
    processor.process(chunk)
    print(chunk.delta, end="", flush=True)

# 获取完整结果
result = processor.get_result()
print(f"\n\nTotal: {result.content}")
print(f"Chunks: {result.chunk_count}")
```

## 4. 流式收集器

```python
from llm_kit.client import StreamCollector

# 简单收集
content = StreamCollector.to_string(client.chat_stream(messages))

# 带回调
def on_delta(delta):
    print(delta, end="", flush=True)

for chunk in StreamCollector.collect(stream, on_delta=on_delta):
    pass
```

## 5. 中断处理

```python
from llm_kit.client import StreamInterrupter

interrupter = StreamInterrupter()

for chunk in interrupter.wrap(client.chat_stream(messages)):
    print(chunk.delta, end="")
    
    # 用户取消
    if user_cancelled:
        interrupter.interrupt()
        break

# 获取已生成的内容
partial_content = interrupter.content
print(f"\nPartial: {partial_content}")
```

## 6. 工具调用流式收集

```python
processor = StreamProcessor()

for chunk in client.chat_stream(messages, tools=tools):
    processor.process(chunk)

result = processor.get_result()

# 工具调用也被正确收集
if result.tool_calls:
    for tc in result.tool_calls:
        print(f"Function: {tc['function']['name']}")
        print(f"Arguments: {tc['function']['arguments']}")
```

## 7. FastAPI 集成

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        for chunk in client.chat_stream(request.messages):
            yield f"data: {json.dumps({'content': chunk.delta})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
```

## 8. 最佳实践

1. **使用 flush**：确保即时显示
2. **处理中断**：允许用户取消
3. **收集完整内容**：用于日志和统计
4. **错误处理**：捕获流式错误
5. **超时设置**：防止流式挂起


