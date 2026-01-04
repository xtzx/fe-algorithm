# FastAPI 快速入门

## 简介

FastAPI 是一个现代、快速（高性能）的 Web 框架，用于基于标准 Python 类型提示构建 API。

## 主要特点

### 性能优异

FastAPI 基于 Starlette 和 Pydantic，性能与 NodeJS 和 Go 相当。

### 类型安全

使用 Python 类型提示进行数据验证和序列化。

### 自动文档

自动生成 OpenAPI（Swagger）和 ReDoc 文档。

## 快速开始

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## 请求验证

使用 Pydantic 模型进行请求验证：

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## 依赖注入

FastAPI 内置强大的依赖注入系统：

```python
from fastapi import Depends

async def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def read_items(db = Depends(get_db)):
    return db.get_all()
```

## 流式响应

支持 SSE（Server-Sent Events）流式响应：

```python
from fastapi.responses import StreamingResponse

async def generate():
    for i in range(10):
        yield f"data: {i}\n\n"
        await asyncio.sleep(1)

@app.get("/stream")
async def stream():
    return StreamingResponse(generate(), media_type="text/event-stream")
```


