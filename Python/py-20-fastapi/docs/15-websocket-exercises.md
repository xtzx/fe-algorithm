# WebSocket 练习题

## 基础练习

### 练习 1：Echo 服务器

实现一个简单的 WebSocket echo 服务器。

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/echo")
async def echo_endpoint(websocket: WebSocket):
    """
    接收消息并原样返回
    - 文本消息：前面加 "Echo: "
    - JSON 消息：添加 echoed: true 字段
    """
    # 你的代码
    pass
```

<details>
<summary>参考答案</summary>

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/echo")
async def echo_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 尝试接收 JSON
            try:
                data = await websocket.receive_json()
                data["echoed"] = True
                await websocket.send_json(data)
            except:
                # 如果不是 JSON，当作文本处理
                text = await websocket.receive_text()
                await websocket.send_text(f"Echo: {text}")
    except WebSocketDisconnect:
        pass
```
</details>

---

### 练习 2：广播服务器

实现一个广播服务器，所有连接的客户端都能收到消息。

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class Broadcaster:
    def __init__(self):
        self.connections: List[WebSocket] = []

    # 实现 connect, disconnect, broadcast 方法

broadcaster = Broadcaster()

@app.websocket("/broadcast")
async def broadcast_endpoint(websocket: WebSocket):
    # 实现
    pass
```

<details>
<summary>参考答案</summary>

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class Broadcaster:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

broadcaster = Broadcaster()

@app.websocket("/broadcast")
async def broadcast_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await broadcaster.broadcast(data)
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
```
</details>

---

### 练习 3：带认证的 WebSocket

实现一个需要 token 认证的 WebSocket 端点。

```python
@app.websocket("/secure")
async def secure_endpoint(websocket: WebSocket, token: str | None = None):
    """
    - token 为空或无效：关闭连接，code=4001
    - token 有效：接受连接并正常通信
    """
    # 假设这是验证函数
    def verify_token(token: str) -> bool:
        return token == "valid_token"

    # 你的代码
    pass
```

<details>
<summary>参考答案</summary>

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

app = FastAPI()

def verify_token(token: str) -> bool:
    return token == "valid_token"

@app.websocket("/secure")
async def secure_endpoint(websocket: WebSocket, token: str | None = Query(None)):
    if not token or not verify_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Secure echo: {data}")
    except WebSocketDisconnect:
        pass
```
</details>

---

### 练习 4：心跳机制

实现带心跳检测的 WebSocket。

```python
import asyncio

@app.websocket("/heartbeat")
async def heartbeat_endpoint(websocket: WebSocket):
    """
    - 每 10 秒发送 ping
    - 客户端需要在 5 秒内响应 pong
    - 超时则关闭连接
    """
    # 你的代码
    pass
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from datetime import datetime, timedelta

@app.websocket("/heartbeat")
async def heartbeat_endpoint(websocket: WebSocket):
    await websocket.accept()

    last_pong = datetime.now()

    async def heartbeat():
        nonlocal last_pong
        while True:
            await asyncio.sleep(10)

            # 检查上次 pong
            if datetime.now() - last_pong > timedelta(seconds=15):
                await websocket.close(code=4002, reason="Heartbeat timeout")
                return

            await websocket.send_json({"type": "ping"})

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "pong":
                last_pong = datetime.now()
            else:
                # 处理其他消息
                await websocket.send_json({"echo": data})

    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
```
</details>

---

## 进阶练习

### 练习 5：聊天室

实现一个多房间聊天室。

```python
class ChatRoom:
    """
    功能：
    - join: 加入房间
    - leave: 离开房间
    - message: 发送消息到房间
    - list_members: 获取房间成员
    """
    pass

@app.websocket("/chat/{user_id}")
async def chat_endpoint(websocket: WebSocket, user_id: str):
    """
    消息格式：
    {"action": "join", "room": "general"}
    {"action": "leave", "room": "general"}
    {"action": "message", "room": "general", "content": "Hello"}
    """
    pass
```

<details>
<summary>参考答案</summary>

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set

app = FastAPI()

class ChatRoom:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[user_id] = websocket

    def disconnect(self, user_id: str):
        # 从所有房间移除
        for members in self.rooms.values():
            members.discard(user_id)
        self.connections.pop(user_id, None)

    async def join(self, user_id: str, room: str):
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(user_id)

        await self.broadcast_to_room(room, {
            "type": "system",
            "message": f"{user_id} joined the room"
        }, exclude=user_id)

    async def leave(self, user_id: str, room: str):
        if room in self.rooms:
            self.rooms[room].discard(user_id)
            await self.broadcast_to_room(room, {
                "type": "system",
                "message": f"{user_id} left the room"
            })

    async def send_message(self, user_id: str, room: str, content: str):
        await self.broadcast_to_room(room, {
            "type": "message",
            "user": user_id,
            "room": room,
            "content": content
        })

    async def broadcast_to_room(self, room: str, data: dict, exclude: str = None):
        if room not in self.rooms:
            return
        for user_id in self.rooms[room]:
            if user_id != exclude and user_id in self.connections:
                await self.connections[user_id].send_json(data)

    def get_members(self, room: str) -> list:
        return list(self.rooms.get(room, set()))

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
                await websocket.send_json({
                    "type": "joined",
                    "room": data["room"],
                    "members": chat.get_members(data["room"])
                })

            elif action == "leave":
                await chat.leave(user_id, data["room"])

            elif action == "message":
                await chat.send_message(user_id, data["room"], data["content"])

    except WebSocketDisconnect:
        chat.disconnect(user_id)
```
</details>

---

### 练习 6：私聊功能

扩展聊天室，添加私聊功能。

```python
"""
消息格式：
{"action": "private", "to": "user2", "content": "Hello privately"}
"""
```

<details>
<summary>参考答案</summary>

```python
class ChatRoomWithPrivate(ChatRoom):
    async def send_private(self, from_user: str, to_user: str, content: str):
        """发送私聊消息"""
        message = {
            "type": "private",
            "from": from_user,
            "content": content
        }

        # 发送给接收者
        if to_user in self.connections:
            await self.connections[to_user].send_json(message)

        # 发送给发送者（确认）
        if from_user in self.connections:
            await self.connections[from_user].send_json({
                "type": "private_sent",
                "to": to_user,
                "content": content
            })

# 在 endpoint 中添加处理
if action == "private":
    await chat.send_private(user_id, data["to"], data["content"])
```
</details>

---

### 练习 7：消息历史

实现消息历史记录功能。

```python
from collections import deque

class MessageHistory:
    """
    - 每个房间保存最近 100 条消息
    - 用户加入房间时发送历史消息
    """
    pass
```

<details>
<summary>参考答案</summary>

```python
from collections import deque
from datetime import datetime
from typing import Dict, List

class MessageHistory:
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self.history: Dict[str, deque] = {}
        self.message_id = 0

    def add_message(self, room: str, user: str, content: str) -> dict:
        if room not in self.history:
            self.history[room] = deque(maxlen=self.max_messages)

        self.message_id += 1
        message = {
            "id": self.message_id,
            "user": user,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.history[room].append(message)
        return message

    def get_history(self, room: str, count: int = 50) -> List[dict]:
        if room not in self.history:
            return []
        return list(self.history[room])[-count:]

# 使用
history = MessageHistory()

class ChatRoomWithHistory(ChatRoom):
    async def join(self, user_id: str, room: str):
        await super().join(user_id, room)

        # 发送历史消息
        messages = history.get_history(room)
        await self.connections[user_id].send_json({
            "type": "history",
            "room": room,
            "messages": messages
        })

    async def send_message(self, user_id: str, room: str, content: str):
        # 保存消息
        message = history.add_message(room, user_id, content)

        # 广播
        await self.broadcast_to_room(room, {
            "type": "message",
            **message
        })
```
</details>

---

## 挑战练习

### 挑战 1：实时协作编辑

实现一个简单的协作文本编辑器。

```python
"""
功能：
- 多用户同时编辑同一文档
- 实时同步光标位置
- 同步文本变化

消息格式：
{"type": "cursor", "position": 42}
{"type": "insert", "position": 10, "text": "hello"}
{"type": "delete", "position": 10, "length": 5}
"""
```

<details>
<summary>参考答案</summary>

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
from dataclasses import dataclass

app = FastAPI()

@dataclass
class Cursor:
    user_id: str
    position: int

class CollaborativeDocument:
    def __init__(self):
        self.content = ""
        self.connections: Dict[str, WebSocket] = {}
        self.cursors: Dict[str, Cursor] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[user_id] = websocket
        self.cursors[user_id] = Cursor(user_id, 0)

        # 发送当前文档状态
        await websocket.send_json({
            "type": "init",
            "content": self.content,
            "cursors": [
                {"user_id": c.user_id, "position": c.position}
                for c in self.cursors.values()
            ]
        })

        # 通知其他人
        await self.broadcast({
            "type": "user_joined",
            "user_id": user_id
        }, exclude=user_id)

    def disconnect(self, user_id: str):
        self.connections.pop(user_id, None)
        self.cursors.pop(user_id, None)

    async def update_cursor(self, user_id: str, position: int):
        self.cursors[user_id].position = position
        await self.broadcast({
            "type": "cursor",
            "user_id": user_id,
            "position": position
        }, exclude=user_id)

    async def insert(self, user_id: str, position: int, text: str):
        # 插入文本
        self.content = self.content[:position] + text + self.content[position:]

        # 更新光标
        self.cursors[user_id].position = position + len(text)

        # 更新其他用户光标
        for cursor in self.cursors.values():
            if cursor.user_id != user_id and cursor.position >= position:
                cursor.position += len(text)

        await self.broadcast({
            "type": "insert",
            "user_id": user_id,
            "position": position,
            "text": text
        }, exclude=user_id)

    async def delete(self, user_id: str, position: int, length: int):
        # 删除文本
        self.content = self.content[:position] + self.content[position + length:]

        # 更新光标
        self.cursors[user_id].position = position

        # 更新其他用户光标
        for cursor in self.cursors.values():
            if cursor.user_id != user_id:
                if cursor.position > position + length:
                    cursor.position -= length
                elif cursor.position > position:
                    cursor.position = position

        await self.broadcast({
            "type": "delete",
            "user_id": user_id,
            "position": position,
            "length": length
        }, exclude=user_id)

    async def broadcast(self, data: dict, exclude: str = None):
        for user_id, ws in self.connections.items():
            if user_id != exclude:
                await ws.send_json(data)

document = CollaborativeDocument()

@app.websocket("/collab/{user_id}")
async def collab_endpoint(websocket: WebSocket, user_id: str):
    await document.connect(user_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "cursor":
                await document.update_cursor(user_id, data["position"])
            elif msg_type == "insert":
                await document.insert(user_id, data["position"], data["text"])
            elif msg_type == "delete":
                await document.delete(user_id, data["position"], data["length"])

    except WebSocketDisconnect:
        document.disconnect(user_id)
```
</details>

---

### 挑战 2：实时游戏状态同步

实现一个简单的多人实时游戏状态同步。

```python
"""
功能：
- 多玩家加入同一游戏
- 实时同步玩家位置
- 30fps 更新频率
"""
```

<details>
<summary>参考答案</summary>

```python
import asyncio
from dataclasses import dataclass
from typing import Dict

@dataclass
class Player:
    id: str
    x: float = 0
    y: float = 0
    velocity_x: float = 0
    velocity_y: float = 0

class GameRoom:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.players: Dict[str, Player] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.running = False

    async def connect(self, player_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[player_id] = websocket
        self.players[player_id] = Player(id=player_id)

        # 发送当前状态
        await websocket.send_json({
            "type": "init",
            "player_id": player_id,
            "players": [
                {"id": p.id, "x": p.x, "y": p.y}
                for p in self.players.values()
            ]
        })

        if not self.running:
            self.running = True
            asyncio.create_task(self.game_loop())

    def disconnect(self, player_id: str):
        self.connections.pop(player_id, None)
        self.players.pop(player_id, None)
        if not self.connections:
            self.running = False

    def update_input(self, player_id: str, vx: float, vy: float):
        if player_id in self.players:
            self.players[player_id].velocity_x = vx
            self.players[player_id].velocity_y = vy

    async def game_loop(self):
        """30 FPS 游戏循环"""
        while self.running and self.connections:
            # 更新位置
            for player in self.players.values():
                player.x += player.velocity_x
                player.y += player.velocity_y

            # 广播状态
            state = {
                "type": "state",
                "players": [
                    {"id": p.id, "x": p.x, "y": p.y}
                    for p in self.players.values()
                ]
            }

            for ws in list(self.connections.values()):
                try:
                    await ws.send_json(state)
                except:
                    pass

            await asyncio.sleep(1/30)  # 30 FPS

games: Dict[str, GameRoom] = {}

@app.websocket("/game/{room_id}/{player_id}")
async def game_endpoint(websocket: WebSocket, room_id: str, player_id: str):
    if room_id not in games:
        games[room_id] = GameRoom(room_id)

    game = games[room_id]
    await game.connect(player_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "input":
                game.update_input(player_id, data["vx"], data["vy"])

    except WebSocketDisconnect:
        game.disconnect(player_id)
```
</details>

---

## 面试题

### 1. WebSocket 和 HTTP 的主要区别是什么？

<details>
<summary>答案</summary>

- **连接方式**：HTTP 短连接，WebSocket 长连接
- **通信方向**：HTTP 单向（请求-响应），WebSocket 双向
- **头部开销**：HTTP 每次请求都带头部，WebSocket 握手后头部很小
- **实时性**：HTTP 需要轮询，WebSocket 原生实时
</details>

### 2. 如何实现 WebSocket 认证？

<details>
<summary>答案</summary>

三种常见方式：
1. **查询参数**：`ws://example.com/ws?token=xxx`
2. **首条消息认证**：连接后第一条消息发送 token
3. **Cookie**：如果同域，自动携带 cookie
</details>

### 3. 如何处理多实例部署的 WebSocket？

<details>
<summary>答案</summary>

使用 Redis Pub/Sub：
1. 每个实例订阅 Redis 频道
2. 发送消息时发布到 Redis
3. 其他实例收到后转发给本地连接
</details>

### 4. WebSocket 断线重连如何实现？

<details>
<summary>答案</summary>

客户端实现：
1. 检测 onclose 事件
2. 指数退避重连（避免服务器压力）
3. 保存最后消息 ID，重连后请求丢失消息
4. 离线时消息入队列，重连后发送
</details>

### 5. 如何防止 WebSocket 连接泄漏？

<details>
<summary>答案</summary>

1. **心跳检测**：定期 ping/pong，超时断开
2. **异常处理**：捕获 WebSocketDisconnect
3. **连接数限制**：限制单用户/总连接数
4. **优雅关闭**：服务器关闭时通知客户端
</details>

