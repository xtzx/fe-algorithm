# 心跳与重连

> 保持连接活跃，处理断线重连

## 为什么需要心跳

1. **检测死连接**：网络断开时 TCP 可能不会立即感知
2. **保持连接活跃**：防止中间设备（防火墙、NAT）超时断开
3. **测量延迟**：RTT（Round-Trip Time）监控

---

## 服务端心跳实现

### 基础心跳

```python
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

HEARTBEAT_INTERVAL = 30  # 秒
HEARTBEAT_TIMEOUT = 10   # 秒

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def heartbeat():
        """定时发送心跳"""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "pong":
                # 收到心跳响应
                continue

            # 处理其他消息
            await handle_message(websocket, data)

    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
```

### 带超时检测的心跳

```python
import asyncio
from datetime import datetime, timedelta

class HeartbeatManager:
    def __init__(
        self,
        websocket: WebSocket,
        interval: float = 30,
        timeout: float = 10
    ):
        self.websocket = websocket
        self.interval = interval
        self.timeout = timeout
        self.last_pong: datetime = datetime.now()
        self._running = False

    async def start(self):
        """启动心跳任务"""
        self._running = True
        asyncio.create_task(self._heartbeat_loop())

    def stop(self):
        """停止心跳"""
        self._running = False

    def pong_received(self):
        """收到 pong 响应"""
        self.last_pong = datetime.now()

    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(self.interval)

            if not self._running:
                break

            # 检查上次 pong 时间
            if datetime.now() - self.last_pong > timedelta(seconds=self.interval + self.timeout):
                # 心跳超时，关闭连接
                await self.websocket.close(code=4002, reason="Heartbeat timeout")
                break

            # 发送 ping
            try:
                await self.websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
            except Exception:
                break

# 使用
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    heartbeat = HeartbeatManager(websocket)
    await heartbeat.start()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "pong":
                heartbeat.pong_received()
                continue

            await handle_message(websocket, data)

    except WebSocketDisconnect:
        pass
    finally:
        heartbeat.stop()
```

---

## 客户端心跳响应（JavaScript）

```javascript
class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('Connected');
            this.reconnectAttempts = 0;
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'ping') {
                // 响应心跳
                this.ws.send(JSON.stringify({
                    type: 'pong',
                    timestamp: data.timestamp
                }));
                return;
            }

            // 处理其他消息
            this.onMessage(data);
        };

        this.ws.onclose = (event) => {
            console.log('Disconnected:', event.code, event.reason);
            this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnect attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => {
            this.connect();
        }, delay);
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    onMessage(data) {
        // 覆盖此方法处理消息
        console.log('Received:', data);
    }
}

// 使用
const client = new WebSocketClient('ws://localhost:8000/ws');
client.onMessage = (data) => {
    console.log('Message:', data);
};
client.connect();
```

---

## 优雅关闭

### 服务端优雅关闭

```python
import signal
import asyncio
from contextlib import asynccontextmanager

class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self._shutdown = False

    async def connect(self, client_id: str, websocket: WebSocket):
        if self._shutdown:
            await websocket.close(code=1001, reason="Server shutting down")
            return False
        await websocket.accept()
        self.connections[client_id] = websocket
        return True

    async def shutdown(self):
        """优雅关闭所有连接"""
        self._shutdown = True

        # 通知所有客户端
        close_tasks = []
        for client_id, ws in self.connections.items():
            close_tasks.append(
                ws.close(code=1001, reason="Server shutting down")
            )

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.connections.clear()

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    yield
    # 关闭
    await manager.shutdown()

app = FastAPI(lifespan=lifespan)
```

### 客户端处理服务端关闭

```javascript
ws.onclose = (event) => {
    if (event.code === 1001) {
        // 服务端正常关闭，可能是维护
        console.log('Server is shutting down');
        // 显示提示，稍后重试
        showNotification('服务器维护中，请稍后重试');
    } else if (event.code === 1006) {
        // 异常断开
        console.log('Connection lost unexpectedly');
        this.attemptReconnect();
    }
};
```

---

## 重连策略

### 指数退避

```python
import asyncio
import random

class ReconnectionStrategy:
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 10,
        jitter: float = 0.1
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter = jitter
        self.attempt = 0

    def reset(self):
        self.attempt = 0

    def get_delay(self) -> float | None:
        """获取下次重连延迟，None 表示不再重试"""
        if self.attempt >= self.max_attempts:
            return None

        # 指数退避
        delay = min(
            self.base_delay * (2 ** self.attempt),
            self.max_delay
        )

        # 添加抖动
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        self.attempt += 1
        return delay

# JavaScript 版本
"""
class ReconnectionStrategy {
    constructor(options = {}) {
        this.baseDelay = options.baseDelay || 1000;
        this.maxDelay = options.maxDelay || 60000;
        this.maxAttempts = options.maxAttempts || 10;
        this.jitter = options.jitter || 0.1;
        this.attempt = 0;
    }

    reset() {
        this.attempt = 0;
    }

    getDelay() {
        if (this.attempt >= this.maxAttempts) {
            return null;
        }

        let delay = Math.min(
            this.baseDelay * Math.pow(2, this.attempt),
            this.maxDelay
        );

        const jitterRange = delay * this.jitter;
        delay += (Math.random() * 2 - 1) * jitterRange;

        this.attempt++;
        return delay;
    }
}
"""
```

### 带状态恢复的重连

```javascript
class StatefulWebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectStrategy = new ReconnectionStrategy();
        this.messageQueue = [];  // 断线期间的消息队列
        this.lastMessageId = null;  // 最后收到的消息ID
    }

    connect() {
        // 重连时带上最后消息ID
        let connectUrl = this.url;
        if (this.lastMessageId) {
            connectUrl += `?last_message_id=${this.lastMessageId}`;
        }

        this.ws = new WebSocket(connectUrl);

        this.ws.onopen = () => {
            console.log('Connected');
            this.reconnectStrategy.reset();

            // 发送队列中的消息
            while (this.messageQueue.length > 0) {
                const msg = this.messageQueue.shift();
                this.ws.send(JSON.stringify(msg));
            }
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.message_id) {
                this.lastMessageId = data.message_id;
            }
            this.onMessage(data);
        };

        this.ws.onclose = () => {
            this.attemptReconnect();
        };
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            // 离线时加入队列
            this.messageQueue.push(data);
        }
    }

    attemptReconnect() {
        const delay = this.reconnectStrategy.getDelay();
        if (delay === null) {
            console.log('Max reconnect attempts reached');
            this.onMaxReconnectReached();
            return;
        }

        setTimeout(() => this.connect(), delay);
    }
}
```

### 服务端支持断点续传

```python
from collections import deque
from typing import Dict, Deque

class MessageBuffer:
    """消息缓冲区，支持断点续传"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        # room -> deque of (message_id, message)
        self.buffers: Dict[str, Deque[tuple]] = {}
        self.message_counter = 0

    def add_message(self, room: str, message: dict) -> int:
        """添加消息并返回消息ID"""
        if room not in self.buffers:
            self.buffers[room] = deque(maxlen=self.max_size)

        self.message_counter += 1
        message_id = self.message_counter
        message["message_id"] = message_id

        self.buffers[room].append((message_id, message))
        return message_id

    def get_messages_after(self, room: str, last_id: int) -> list[dict]:
        """获取指定ID之后的所有消息"""
        if room not in self.buffers:
            return []

        messages = []
        for msg_id, msg in self.buffers[room]:
            if msg_id > last_id:
                messages.append(msg)

        return messages

message_buffer = MessageBuffer()

@app.websocket("/ws/{room}")
async def websocket_endpoint(
    websocket: WebSocket,
    room: str,
    last_message_id: int | None = None
):
    await websocket.accept()

    # 发送错过的消息
    if last_message_id is not None:
        missed_messages = message_buffer.get_messages_after(room, last_message_id)
        for msg in missed_messages:
            await websocket.send_json(msg)

    # 继续正常处理...
```

---

## 连接状态监控

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass
class ConnectionStats:
    connected_at: datetime
    last_ping: datetime | None = None
    last_pong: datetime | None = None
    messages_sent: int = 0
    messages_received: int = 0
    latency_ms: float | None = None

class MonitoredConnectionManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.stats: Dict[str, ConnectionStats] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[client_id] = websocket
        self.stats[client_id] = ConnectionStats(connected_at=datetime.now())

    def record_ping(self, client_id: str):
        if client_id in self.stats:
            self.stats[client_id].last_ping = datetime.now()

    def record_pong(self, client_id: str, ping_timestamp: datetime):
        if client_id in self.stats:
            now = datetime.now()
            self.stats[client_id].last_pong = now
            self.stats[client_id].latency_ms = (now - ping_timestamp).total_seconds() * 1000

    def get_health_status(self) -> dict:
        """获取所有连接的健康状态"""
        now = datetime.now()
        status = {
            "total_connections": len(self.connections),
            "healthy": 0,
            "unhealthy": 0,
            "connections": {}
        }

        for client_id, stats in self.stats.items():
            is_healthy = True
            if stats.last_pong:
                # 超过60秒没有pong认为不健康
                if (now - stats.last_pong).total_seconds() > 60:
                    is_healthy = False

            status["connections"][client_id] = {
                "healthy": is_healthy,
                "latency_ms": stats.latency_ms,
                "uptime_seconds": (now - stats.connected_at).total_seconds()
            }

            if is_healthy:
                status["healthy"] += 1
            else:
                status["unhealthy"] += 1

        return status

# 健康检查端点
@app.get("/ws/health")
async def websocket_health():
    return manager.get_health_status()
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 心跳间隔太长 | 中间设备超时断开 | 30秒以内 |
| 没有超时检测 | 死连接占用资源 | 检测 pong 超时 |
| 重连太激进 | 服务器压力大 | 指数退避 + 抖动 |
| 没有消息队列 | 离线消息丢失 | 本地队列 + 重发 |
| 没有优雅关闭 | 客户端不知道原因 | 发送关闭原因 |

