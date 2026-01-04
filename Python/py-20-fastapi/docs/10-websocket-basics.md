# WebSocket 基础

> 理解 WebSocket 协议及其应用场景

## 什么是 WebSocket

WebSocket 是一种在单个 TCP 连接上进行全双工通信的协议，使得客户端和服务器之间可以实时双向传输数据。

```
HTTP 请求-响应模式:
Client ──请求──> Server
Client <──响应── Server
（每次通信需要新建连接）

WebSocket 全双工模式:
Client <══════> Server
（建立连接后，双方可随时发送数据）
```

---

## WebSocket vs HTTP

| 特性 | HTTP | WebSocket |
|------|------|-----------|
| 连接方式 | 短连接（请求-响应）| 长连接（持久）|
| 通信方向 | 单向（客户端发起）| 双向（任一方发起）|
| 头部开销 | 每次请求都带完整头 | 握手后头部很小 |
| 实时性 | 需要轮询 | 原生实时 |
| 适用场景 | 普通 API | 实时应用 |

---

## WebSocket vs 其他实时技术

### 1. HTTP 轮询 (Polling)

```javascript
// 客户端不断请求
setInterval(async () => {
    const response = await fetch('/api/messages');
    // 处理新消息
}, 1000);
```

**缺点**：
- 大量无效请求
- 延迟高（最高等于轮询间隔）
- 服务器压力大

### 2. 长轮询 (Long Polling)

```javascript
// 服务器保持连接直到有新数据
async function poll() {
    const response = await fetch('/api/messages?wait=30');
    handleMessages(response);
    poll(); // 立即开始下一次
}
```

**缺点**：
- 仍需要频繁建立连接
- 服务器需要维护挂起的请求
- 超时处理复杂

### 3. Server-Sent Events (SSE)

```javascript
// 单向：服务器推送到客户端
const source = new EventSource('/api/events');
source.onmessage = (event) => {
    console.log(event.data);
};
```

**特点**：
- 单向通信（服务器到客户端）
- 基于 HTTP，兼容性好
- 自动重连
- 适合：通知、实时更新

### 4. WebSocket

```javascript
// 双向实时通信
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    console.log(event.data);
};
ws.send('Hello Server');
```

**特点**：
- 双向通信
- 低延迟、低开销
- 适合：聊天、游戏、协作编辑

### 选型指南

| 场景 | 推荐技术 |
|------|---------|
| 实时通知（单向）| SSE |
| 聊天应用 | WebSocket |
| 实时协作 | WebSocket |
| 股票行情 | WebSocket 或 SSE |
| 简单轮询数据 | HTTP 轮询 |
| 偶尔的实时更新 | 长轮询 |

---

## WebSocket 协议详解

### 握手过程

```
1. 客户端发起 HTTP 升级请求
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

2. 服务器响应确认升级
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

3. 连接建立，开始 WebSocket 通信
```

### 数据帧格式

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               |Masking-key, if MASK set to 1  |
+-------------------------------+-------------------------------+
| Masking-key (continued)       |          Payload Data         |
+-------------------------------- - - - - - - - - - - - - - - - +
:                     Payload Data continued ...                :
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
|                     Payload Data continued ...                |
+---------------------------------------------------------------+
```

### 操作码 (Opcode)

| Opcode | 类型 | 说明 |
|--------|------|------|
| 0x0 | 延续帧 | 数据分片 |
| 0x1 | 文本帧 | UTF-8 文本 |
| 0x2 | 二进制帧 | 二进制数据 |
| 0x8 | 关闭帧 | 关闭连接 |
| 0x9 | Ping | 心跳检测 |
| 0xA | Pong | 心跳响应 |

---

## WebSocket 状态

```python
# WebSocket 连接状态
class WebSocketState:
    CONNECTING = 0  # 正在连接
    OPEN = 1        # 已连接
    CLOSING = 2     # 正在关闭
    CLOSED = 3      # 已关闭
```

---

## 安全考虑

### 1. 使用 WSS（WebSocket Secure）

```javascript
// 生产环境使用加密连接
const ws = new WebSocket('wss://secure.example.com/ws');
```

### 2. 认证和授权

```python
# 方式 1：URL 参数（不推荐，会暴露在日志）
ws://example.com/ws?token=xxx

# 方式 2：第一条消息认证
await websocket.send(json.dumps({"type": "auth", "token": "xxx"}))

# 方式 3：Cookie（如果同域）
# 自动携带
```

### 3. 输入验证

```python
# 验证所有接收的消息
async def handle_message(websocket, message):
    try:
        data = json.loads(message)
        validate_message(data)  # 验证格式和内容
        await process_message(data)
    except (json.JSONDecodeError, ValidationError):
        await websocket.close(code=1003, reason="Invalid message")
```

### 4. 速率限制

```python
# 限制消息频率
from collections import defaultdict
import time

message_counts = defaultdict(list)

async def rate_limit(client_id: str, max_per_second: int = 10):
    now = time.time()
    # 清理旧记录
    message_counts[client_id] = [
        t for t in message_counts[client_id]
        if now - t < 1
    ]
    if len(message_counts[client_id]) >= max_per_second:
        raise RateLimitExceeded()
    message_counts[client_id].append(now)
```

---

## 适用场景

| 场景 | 为什么适合 |
|------|-----------|
| 即时通讯 | 双向实时消息 |
| 在线游戏 | 低延迟状态同步 |
| 协作编辑 | 实时同步文档 |
| 实时仪表盘 | 推送实时数据 |
| 在线客服 | 双向对话 |
| 直播弹幕 | 高并发消息 |
| 金融交易 | 实时行情推送 |

---

## JS 对照

| Python (FastAPI) | JavaScript | 说明 |
|------------------|------------|------|
| `WebSocket` 类 | `WebSocket` API | 客户端连接 |
| `websocket.accept()` | 自动 | 接受连接 |
| `websocket.send_text()` | `ws.send()` | 发送消息 |
| `websocket.receive_text()` | `ws.onmessage` | 接收消息 |
| `websocket.close()` | `ws.close()` | 关闭连接 |

---

## 小结

1. **WebSocket** 提供全双工、低延迟的实时通信
2. **适合**双向实时场景（聊天、游戏、协作）
3. **SSE** 适合单向推送场景
4. **安全性**：使用 WSS、认证、验证、限流
5. **状态管理**：注意连接生命周期

