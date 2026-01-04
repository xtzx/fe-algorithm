# FastAPI WebSocket

> 使用 FastAPI 构建 WebSocket 服务

## 基础 WebSocket 端点

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受连接
    while True:
        data = await websocket.receive_text()  # 接收消息
        await websocket.send_text(f"Echo: {data}")  # 发送消息
```

### 客户端连接

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Connected');
    ws.send('Hello Server');
};

ws.onmessage = (event) => {
    console.log('Received:', event.data);
};

ws.onclose = () => {
    console.log('Disconnected');
};

ws.onerror = (error) => {
    console.error('Error:', error);
};
```

---

## 接收与发送消息

### 文本消息

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 发送文本
    await websocket.send_text("Hello Client")

    # 接收文本
    text = await websocket.receive_text()
    print(f"Received: {text}")
```

### JSON 消息

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 发送 JSON
    await websocket.send_json({"type": "greeting", "message": "Hello"})

    # 接收 JSON
    data = await websocket.receive_json()
    print(f"Received: {data}")
```

### 二进制消息

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 发送二进制
    await websocket.send_bytes(b"\x00\x01\x02")

    # 接收二进制
    data = await websocket.receive_bytes()
```

### 通用接收

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # 接收任意类型消息
        message = await websocket.receive()

        if message["type"] == "websocket.receive":
            if "text" in message:
                print(f"Text: {message['text']}")
            elif "bytes" in message:
                print(f"Bytes: {message['bytes']}")
        elif message["type"] == "websocket.disconnect":
            print("Client disconnected")
            break
```

---

## 路径参数与查询参数

### 路径参数

```python
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await websocket.send_text(f"Hello, {client_id}")

    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"{client_id}: {data}")
```

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/user123');
```

### 查询参数

```python
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str | None = None,
    room: str = "default"
):
    if not token or not verify_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    await websocket.send_text(f"Joined room: {room}")
```

```javascript
const ws = new WebSocket('ws://localhost:8000/ws?token=abc123&room=general');
```

---

## WebSocket 认证

### 方式 1：查询参数

```python
from fastapi import Query

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    try:
        user = verify_token(token)
    except InvalidToken:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()
    # 已认证的连接
```

### 方式 2：首条消息认证

```python
import json
import asyncio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 等待认证消息（5秒超时）
    try:
        auth_message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=5.0
        )

        if auth_message.get("type") != "auth":
            raise ValueError("Expected auth message")

        token = auth_message.get("token")
        user = verify_token(token)

    except asyncio.TimeoutError:
        await websocket.close(code=4000, reason="Auth timeout")
        return
    except Exception:
        await websocket.close(code=4001, reason="Auth failed")
        return

    # 已认证，继续通信
    await websocket.send_json({"type": "auth_success", "user": user.name})

    while True:
        data = await websocket.receive_json()
        await handle_message(websocket, user, data)
```

### 方式 3：Cookie

```python
from fastapi import Cookie, Depends

async def get_user_from_cookie(
    session_id: str | None = Cookie(None)
) -> User | None:
    if not session_id:
        return None
    return await get_user_by_session(session_id)

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user: User | None = Depends(get_user_from_cookie)
):
    if not user:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    # 已认证
```

---

## 异常处理

### WebSocketDisconnect

```python
from fastapi import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
```

### 关闭连接

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "quit":
                await websocket.close(code=1000, reason="Goodbye")
                break

            # 处理其他消息

    except WebSocketDisconnect as e:
        print(f"Disconnected: code={e.code}, reason={e.reason}")
```

### 关闭码

| 码 | 说明 |
|----|------|
| 1000 | 正常关闭 |
| 1001 | 离开（如刷新页面）|
| 1002 | 协议错误 |
| 1003 | 不支持的数据类型 |
| 1008 | 违反策略 |
| 1011 | 服务器错误 |
| 4000-4999 | 应用自定义 |

---

## 与依赖注入配合

```python
from fastapi import Depends, WebSocket

# 依赖
async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def get_manager() -> ConnectionManager:
    return manager

# 使用依赖
@app.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: str,
    db = Depends(get_db),
    manager: ConnectionManager = Depends(get_manager)
):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # 使用 db 和 manager
            await manager.broadcast(f"Room {room_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## 与 HTTP 端点配合

```python
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel

app = FastAPI()
manager = ConnectionManager()

class Message(BaseModel):
    content: str
    room: str

# HTTP 端点触发 WebSocket 推送
@app.post("/broadcast")
async def broadcast_message(message: Message):
    await manager.broadcast_to_room(message.room, message.content)
    return {"status": "sent"}

# WebSocket 端点
@app.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    await manager.connect(websocket, room)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast_to_room(room, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
```

---

## 并发处理

### 同时发送和接收

```python
import asyncio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def receiver():
        """接收消息"""
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")

    async def sender():
        """定时发送消息"""
        while True:
            await asyncio.sleep(5)
            await websocket.send_text("Ping")

    # 并发运行
    try:
        await asyncio.gather(receiver(), sender())
    except WebSocketDisconnect:
        print("Disconnected")
```

### 使用 asyncio.create_task

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 启动后台任务
    heartbeat_task = asyncio.create_task(heartbeat(websocket))

    try:
        while True:
            data = await websocket.receive_text()
            await process_message(websocket, data)
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()

async def heartbeat(websocket: WebSocket):
    while True:
        await asyncio.sleep(30)
        await websocket.send_json({"type": "ping"})
```

---

## 测试 WebSocket

```python
from fastapi.testclient import TestClient

def test_websocket():
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        # 发送消息
        websocket.send_text("Hello")

        # 接收消息
        data = websocket.receive_text()
        assert data == "Echo: Hello"

def test_websocket_json():
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "greeting"})
        data = websocket.receive_json()
        assert data["type"] == "response"

def test_websocket_disconnect():
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("quit")
        # 连接应该关闭
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 忘记 accept() | 连接不会建立 | 先调用 accept() |
| 忘记处理断开 | 异常导致崩溃 | 捕获 WebSocketDisconnect |
| 阻塞主循环 | 无法接收消息 | 使用 asyncio.create_task |
| 认证不安全 | token 暴露在 URL | 使用首条消息或 Cookie |
| 没有超时 | 资源泄漏 | 设置合理超时 |

---

## 小结

1. **基础**：`websocket.accept()`、`send_*`、`receive_*`
2. **认证**：查询参数、首条消息、Cookie
3. **异常**：捕获 `WebSocketDisconnect`
4. **并发**：`asyncio.gather` 或 `create_task`
5. **测试**：`TestClient.websocket_connect()`

