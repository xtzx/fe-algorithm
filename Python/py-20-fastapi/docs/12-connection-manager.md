# 连接管理

> 管理多个 WebSocket 连接：广播、私聊、房间

## ConnectionManager 设计

### 基础版本

```python
from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# 使用
manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"{client_id} left the chat")
```

---

## 带用户标识的连接管理

```python
from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    def __init__(self):
        # client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        # 如果已有连接，先断开旧的
        if client_id in self.active_connections:
            old_ws = self.active_connections[client_id]
            await old_ws.close(code=4000, reason="Replaced by new connection")
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_to_user(self, client_id: str, message: str):
        """发送给特定用户"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: str, exclude: str | None = None):
        """广播给所有用户"""
        for client_id, connection in self.active_connections.items():
            if client_id != exclude:
                await connection.send_text(message)

    def get_online_users(self) -> list[str]:
        """获取在线用户列表"""
        return list(self.active_connections.keys())
```

---

## 房间/频道管理

```python
from fastapi import WebSocket
from typing import Dict, Set
from dataclasses import dataclass, field

@dataclass
class Room:
    name: str
    members: Set[str] = field(default_factory=set)

class RoomManager:
    def __init__(self):
        # client_id -> WebSocket
        self.connections: Dict[str, WebSocket] = {}
        # room_name -> Room
        self.rooms: Dict[str, Room] = {}
        # client_id -> set of room names
        self.user_rooms: Dict[str, Set[str]] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[client_id] = websocket
        self.user_rooms[client_id] = set()

    def disconnect(self, client_id: str):
        # 从所有房间移除
        if client_id in self.user_rooms:
            for room_name in self.user_rooms[client_id]:
                if room_name in self.rooms:
                    self.rooms[room_name].members.discard(client_id)
            del self.user_rooms[client_id]

        if client_id in self.connections:
            del self.connections[client_id]

    async def join_room(self, client_id: str, room_name: str):
        """加入房间"""
        if room_name not in self.rooms:
            self.rooms[room_name] = Room(name=room_name)

        self.rooms[room_name].members.add(client_id)
        self.user_rooms[client_id].add(room_name)

        # 通知房间内其他成员
        await self.broadcast_to_room(
            room_name,
            f"{client_id} joined {room_name}",
            exclude=client_id
        )

    async def leave_room(self, client_id: str, room_name: str):
        """离开房间"""
        if room_name in self.rooms:
            self.rooms[room_name].members.discard(client_id)

            # 如果房间空了，删除
            if not self.rooms[room_name].members:
                del self.rooms[room_name]

        if client_id in self.user_rooms:
            self.user_rooms[client_id].discard(room_name)

        # 通知房间内其他成员
        await self.broadcast_to_room(
            room_name,
            f"{client_id} left {room_name}"
        )

    async def broadcast_to_room(
        self,
        room_name: str,
        message: str,
        exclude: str | None = None
    ):
        """向房间广播消息"""
        if room_name not in self.rooms:
            return

        for member_id in self.rooms[room_name].members:
            if member_id != exclude and member_id in self.connections:
                await self.connections[member_id].send_text(message)

    async def send_to_user(self, client_id: str, message: str):
        """私聊"""
        if client_id in self.connections:
            await self.connections[client_id].send_text(message)

    def get_room_members(self, room_name: str) -> Set[str]:
        """获取房间成员"""
        if room_name in self.rooms:
            return self.rooms[room_name].members.copy()
        return set()

# 使用示例
room_manager = RoomManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await room_manager.connect(client_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "join":
                await room_manager.join_room(client_id, data["room"])

            elif data["type"] == "leave":
                await room_manager.leave_room(client_id, data["room"])

            elif data["type"] == "message":
                await room_manager.broadcast_to_room(
                    data["room"],
                    f"{client_id}: {data['content']}"
                )

            elif data["type"] == "private":
                await room_manager.send_to_user(
                    data["to"],
                    f"[Private from {client_id}]: {data['content']}"
                )

    except WebSocketDisconnect:
        room_manager.disconnect(client_id)
```

---

## 连接数限制

```python
from fastapi import WebSocket, HTTPException
from typing import Dict
import asyncio

class LimitedConnectionManager:
    def __init__(self, max_connections: int = 1000, max_per_user: int = 5):
        self.max_connections = max_connections
        self.max_per_user = max_per_user
        self.connections: Dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, user_id: str, websocket: WebSocket) -> bool:
        async with self._lock:
            # 检查总连接数
            total = sum(len(conns) for conns in self.connections.values())
            if total >= self.max_connections:
                await websocket.close(code=4003, reason="Server at capacity")
                return False

            # 检查单用户连接数
            if user_id not in self.connections:
                self.connections[user_id] = []

            if len(self.connections[user_id]) >= self.max_per_user:
                # 关闭最旧的连接
                old_ws = self.connections[user_id].pop(0)
                await old_ws.close(code=4000, reason="Too many connections")

            await websocket.accept()
            self.connections[user_id].append(websocket)
            return True

    async def disconnect(self, user_id: str, websocket: WebSocket):
        async with self._lock:
            if user_id in self.connections:
                if websocket in self.connections[user_id]:
                    self.connections[user_id].remove(websocket)
                if not self.connections[user_id]:
                    del self.connections[user_id]

    def get_stats(self) -> dict:
        """获取连接统计"""
        return {
            "total_connections": sum(len(c) for c in self.connections.values()),
            "unique_users": len(self.connections),
            "max_connections": self.max_connections,
        }
```

---

## 消息类型和协议

### 定义消息协议

```python
from pydantic import BaseModel
from typing import Literal, Any
from datetime import datetime

class BaseMessage(BaseModel):
    type: str
    timestamp: datetime = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)

class JoinRoomMessage(BaseMessage):
    type: Literal["join"] = "join"
    room: str

class LeaveRoomMessage(BaseMessage):
    type: Literal["leave"] = "leave"
    room: str

class ChatMessage(BaseMessage):
    type: Literal["chat"] = "chat"
    room: str
    content: str
    sender: str | None = None

class PrivateMessage(BaseMessage):
    type: Literal["private"] = "private"
    to: str
    content: str
    sender: str | None = None

class SystemMessage(BaseMessage):
    type: Literal["system"] = "system"
    content: str

# 消息解析
from typing import Union

MessageType = Union[JoinRoomMessage, LeaveRoomMessage, ChatMessage, PrivateMessage]

def parse_message(data: dict) -> MessageType:
    msg_type = data.get("type")

    if msg_type == "join":
        return JoinRoomMessage(**data)
    elif msg_type == "leave":
        return LeaveRoomMessage(**data)
    elif msg_type == "chat":
        return ChatMessage(**data)
    elif msg_type == "private":
        return PrivateMessage(**data)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
```

### 使用消息协议

```python
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()

            try:
                message = parse_message(data)
                message.sender = client_id

                if isinstance(message, JoinRoomMessage):
                    await manager.join_room(client_id, message.room)

                elif isinstance(message, ChatMessage):
                    await manager.broadcast_to_room(
                        message.room,
                        message.model_dump_json()
                    )

                elif isinstance(message, PrivateMessage):
                    await manager.send_to_user(
                        message.to,
                        message.model_dump_json()
                    )

            except ValueError as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
```

---

## 线程安全考虑

```python
import asyncio
from typing import Dict, Set

class ThreadSafeConnectionManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket):
        async with self._lock:
            await websocket.accept()
            self.connections[client_id] = websocket

    async def disconnect(self, client_id: str):
        async with self._lock:
            if client_id in self.connections:
                del self.connections[client_id]

    async def broadcast(self, message: str):
        # 复制连接列表以避免迭代时修改
        async with self._lock:
            connections = list(self.connections.items())

        # 并发发送
        tasks = []
        for client_id, websocket in connections:
            tasks.append(self._safe_send(client_id, websocket, message))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_send(self, client_id: str, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception:
            # 发送失败，移除连接
            await self.disconnect(client_id)
```

---

## 完整聊天室示例

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Set
from datetime import datetime
import json

app = FastAPI()

class ChatRoom:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[user_id] = websocket
        await self.send_to_user(user_id, {
            "type": "connected",
            "user_id": user_id,
            "rooms": list(self.rooms.keys())
        })

    def disconnect(self, user_id: str):
        # 从所有房间移除
        for room_members in self.rooms.values():
            room_members.discard(user_id)
        self.connections.pop(user_id, None)

    async def join(self, user_id: str, room: str):
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(user_id)

        await self.broadcast_to_room(room, {
            "type": "user_joined",
            "user_id": user_id,
            "room": room,
            "members": list(self.rooms[room])
        })

    async def leave(self, user_id: str, room: str):
        if room in self.rooms:
            self.rooms[room].discard(user_id)
            await self.broadcast_to_room(room, {
                "type": "user_left",
                "user_id": user_id,
                "room": room
            })

    async def send_message(self, user_id: str, room: str, content: str):
        await self.broadcast_to_room(room, {
            "type": "message",
            "user_id": user_id,
            "room": room,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    async def send_to_user(self, user_id: str, data: dict):
        if user_id in self.connections:
            await self.connections[user_id].send_json(data)

    async def broadcast_to_room(self, room: str, data: dict):
        if room not in self.rooms:
            return
        for user_id in self.rooms[room]:
            await self.send_to_user(user_id, data)

chat = ChatRoom()

@app.websocket("/chat/{user_id}")
async def chat_endpoint(websocket: WebSocket, user_id: str):
    await chat.connect(user_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "join":
                await chat.join(user_id, data["room"])
            elif action == "leave":
                await chat.leave(user_id, data["room"])
            elif action == "message":
                await chat.send_message(user_id, data["room"], data["content"])

    except WebSocketDisconnect:
        chat.disconnect(user_id)
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 迭代时修改字典 | RuntimeError | 先复制再迭代 |
| 没有锁保护 | 竞争条件 | 使用 asyncio.Lock |
| 发送失败不处理 | 连接堆积 | try-except 并移除 |
| 房间空了不清理 | 内存泄漏 | 检查并删除空房间 |

