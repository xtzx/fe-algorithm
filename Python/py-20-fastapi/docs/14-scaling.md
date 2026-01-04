# WebSocket 扩展与部署

> 多实例部署、Redis Pub/Sub、负载均衡

## 单实例的局限

```
问题：WebSocket 连接只存在于单个服务器实例

实例 A: [User1, User2, User3]
实例 B: [User4, User5, User6]

User1 发送消息给 User4？
→ 消息在实例 A，User4 在实例 B
→ 无法直接送达
```

---

## Redis Pub/Sub 跨实例通信

### 架构图

```
┌─────────────────────────────────────────────────┐
│                    Nginx                         │
│              (负载均衡/WebSocket代理)            │
└───────────────┬─────────────┬───────────────────┘
                │             │
        ┌───────▼───────┐ ┌───▼───────────┐
        │   实例 A      │ │   实例 B      │
        │ [U1,U2,U3]    │ │ [U4,U5,U6]    │
        └───────┬───────┘ └───┬───────────┘
                │             │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │    Redis    │
                │  Pub/Sub    │
                └─────────────┘
```

### 实现

```python
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

app = FastAPI()

# Redis 连接
redis_client: redis.Redis = None

class DistributedConnectionManager:
    def __init__(self):
        self.local_connections: dict[str, WebSocket] = {}
        self.pubsub = None

    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        global redis_client
        redis_client = redis.from_url(redis_url)
        self.pubsub = redis_client.pubsub()
        await self.pubsub.subscribe("chat:broadcast")

        # 启动监听任务
        asyncio.create_task(self._listen_redis())

    async def _listen_redis(self):
        """监听 Redis 消息并转发给本地连接"""
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await self._handle_redis_message(data)

    async def _handle_redis_message(self, data: dict):
        """处理来自 Redis 的消息"""
        msg_type = data.get("type")

        if msg_type == "broadcast":
            # 广播给所有本地连接
            message = data["message"]
            for ws in self.local_connections.values():
                await ws.send_text(message)

        elif msg_type == "direct":
            # 直接发送给特定用户
            user_id = data["user_id"]
            if user_id in self.local_connections:
                await self.local_connections[user_id].send_text(data["message"])

        elif msg_type == "room":
            # 发送给房间成员（本地）
            room = data["room"]
            members = data.get("members", [])
            message = data["message"]
            for user_id in members:
                if user_id in self.local_connections:
                    await self.local_connections[user_id].send_text(message)

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.local_connections[user_id] = websocket

        # 在 Redis 中记录用户在线状态
        await redis_client.hset("online_users", user_id, "1")

    async def disconnect(self, user_id: str):
        self.local_connections.pop(user_id, None)
        await redis_client.hdel("online_users", user_id)

    async def broadcast(self, message: str):
        """广播消息（通过 Redis 发布）"""
        await redis_client.publish("chat:broadcast", json.dumps({
            "type": "broadcast",
            "message": message
        }))

    async def send_to_user(self, user_id: str, message: str):
        """发送给特定用户（通过 Redis 发布）"""
        # 先尝试本地
        if user_id in self.local_connections:
            await self.local_connections[user_id].send_text(message)
            return

        # 通过 Redis 发布
        await redis_client.publish("chat:broadcast", json.dumps({
            "type": "direct",
            "user_id": user_id,
            "message": message
        }))

    async def get_online_users(self) -> list[str]:
        """获取所有在线用户"""
        users = await redis_client.hkeys("online_users")
        return [u.decode() for u in users]

manager = DistributedConnectionManager()

@app.on_event("startup")
async def startup():
    await manager.init_redis()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "broadcast":
                await manager.broadcast(f"{user_id}: {data['message']}")

            elif data["type"] == "direct":
                await manager.send_to_user(
                    data["to"],
                    f"[Private from {user_id}]: {data['message']}"
                )

    except WebSocketDisconnect:
        await manager.disconnect(user_id)
```

---

## 房间管理（Redis 版）

```python
class DistributedRoomManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_connections: dict[str, WebSocket] = {}

    async def join_room(self, user_id: str, room: str):
        """加入房间"""
        # 添加到 Redis Set
        await self.redis.sadd(f"room:{room}:members", user_id)
        # 记录用户所在房间
        await self.redis.sadd(f"user:{user_id}:rooms", room)

    async def leave_room(self, user_id: str, room: str):
        """离开房间"""
        await self.redis.srem(f"room:{room}:members", user_id)
        await self.redis.srem(f"user:{user_id}:rooms", room)

    async def get_room_members(self, room: str) -> list[str]:
        """获取房间成员"""
        members = await self.redis.smembers(f"room:{room}:members")
        return [m.decode() for m in members]

    async def broadcast_to_room(self, room: str, message: str, exclude: str | None = None):
        """向房间广播"""
        members = await self.get_room_members(room)

        await self.redis.publish("chat:room", json.dumps({
            "type": "room",
            "room": room,
            "members": [m for m in members if m != exclude],
            "message": message
        }))

    async def disconnect_user(self, user_id: str):
        """用户断开连接"""
        # 获取用户所在的所有房间
        rooms = await self.redis.smembers(f"user:{user_id}:rooms")

        # 从所有房间移除
        for room in rooms:
            room = room.decode()
            await self.leave_room(user_id, room)

        # 清理用户房间记录
        await self.redis.delete(f"user:{user_id}:rooms")
```

---

## Nginx 配置

### WebSocket 代理

```nginx
# /etc/nginx/nginx.conf

upstream websocket_backend {
    # IP Hash 保证同一用户连到同一实例（粘性会话）
    ip_hash;

    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    server_name example.com;

    location /ws {
        proxy_pass http://websocket_backend;

        # WebSocket 必需的头
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # 其他代理头
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    location / {
        proxy_pass http://websocket_backend;
    }
}
```

### SSL/WSS 配置

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;

    location /ws {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        # ... 其他配置
    }
}
```

---

## Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  app1:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379
      - INSTANCE_ID=app1
    depends_on:
      - redis

  app2:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379
      - INSTANCE_ID=app2
    depends_on:
      - redis

  app3:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379
      - INSTANCE_ID=app3
    depends_on:
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - app1
      - app2
      - app3

volumes:
  redis_data:
```

---

## 消息持久化

```python
from datetime import datetime
from typing import List

class MessageStore:
    """消息持久化存储"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.max_messages = 1000  # 每个房间最多保存消息数

    async def save_message(self, room: str, message: dict):
        """保存消息"""
        message["id"] = await self.redis.incr("message:id_counter")
        message["timestamp"] = datetime.now().isoformat()

        # 添加到房间消息列表
        await self.redis.lpush(
            f"room:{room}:messages",
            json.dumps(message)
        )

        # 保持列表长度
        await self.redis.ltrim(f"room:{room}:messages", 0, self.max_messages - 1)

        return message["id"]

    async def get_recent_messages(self, room: str, count: int = 50) -> List[dict]:
        """获取最近消息"""
        messages = await self.redis.lrange(f"room:{room}:messages", 0, count - 1)
        return [json.loads(m) for m in messages][::-1]  # 反转为时间正序

    async def get_messages_after(self, room: str, message_id: int) -> List[dict]:
        """获取指定ID之后的消息"""
        # 获取所有消息并过滤
        all_messages = await self.get_recent_messages(room, self.max_messages)
        return [m for m in all_messages if m["id"] > message_id]

# 使用
message_store = MessageStore(redis_client)

@app.websocket("/ws/{room}/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room: str,
    user_id: str,
    last_message_id: int | None = None
):
    await websocket.accept()

    # 发送历史消息
    if last_message_id:
        messages = await message_store.get_messages_after(room, last_message_id)
    else:
        messages = await message_store.get_recent_messages(room)

    for msg in messages:
        await websocket.send_json(msg)

    # 继续实时通信...
```

---

## 监控与告警

```python
from prometheus_client import Counter, Gauge, Histogram

# Prometheus 指标
ws_connections = Gauge(
    'websocket_connections_total',
    'Total WebSocket connections',
    ['instance']
)

ws_messages_sent = Counter(
    'websocket_messages_sent_total',
    'Total messages sent',
    ['instance', 'type']
)

ws_messages_received = Counter(
    'websocket_messages_received_total',
    'Total messages received',
    ['instance', 'type']
)

ws_connection_duration = Histogram(
    'websocket_connection_duration_seconds',
    'WebSocket connection duration',
    ['instance']
)

# 使用
class MonitoredConnectionManager:
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.connections: dict[str, WebSocket] = {}
        self.connection_times: dict[str, datetime] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[user_id] = websocket
        self.connection_times[user_id] = datetime.now()

        ws_connections.labels(instance=self.instance_id).inc()

    async def disconnect(self, user_id: str):
        if user_id in self.connections:
            # 记录连接时长
            if user_id in self.connection_times:
                duration = (datetime.now() - self.connection_times[user_id]).total_seconds()
                ws_connection_duration.labels(
                    instance=self.instance_id
                ).observe(duration)
                del self.connection_times[user_id]

            del self.connections[user_id]
            ws_connections.labels(instance=self.instance_id).dec()

    async def send(self, user_id: str, message: str, msg_type: str = "text"):
        if user_id in self.connections:
            await self.connections[user_id].send_text(message)
            ws_messages_sent.labels(
                instance=self.instance_id,
                type=msg_type
            ).inc()
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 没有粘性会话 | 用户重连到不同实例 | Nginx ip_hash 或 Redis 状态 |
| Nginx 超时太短 | 连接意外断开 | 设置长超时 + 心跳 |
| Redis 单点故障 | 整个系统不可用 | Redis Cluster/Sentinel |
| 消息丢失 | 实例间同步延迟 | 消息持久化 + 确认机制 |
| 连接数过多 | 单实例内存耗尽 | 水平扩展 + 连接数限制 |

---

## 小结

1. **单实例局限**：连接在内存中，无法跨实例通信
2. **Redis Pub/Sub**：实现跨实例消息同步
3. **Nginx 配置**：WebSocket 代理 + 粘性会话
4. **消息持久化**：支持断线重连、历史消息
5. **监控**：Prometheus 指标收集

