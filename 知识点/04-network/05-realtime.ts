/**
 * ============================================================
 * 📚 实时通信
 * ============================================================
 *
 * 面试考察重点：
 * 1. 实时通信方案对比
 * 2. WebSocket 原理与应用
 * 3. 心跳机制与重连策略
 * 4. 实际业务场景
 */

// ============================================================
// 1. 实时通信方案对比
// ============================================================

/**
 * 📊 方案对比
 *
 * ┌───────────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
 * │ 方案               │ 短轮询                 │ 长轮询                  │ SSE                    │
 * ├───────────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
 * │ 原理               │ 定时发起请求           │ 服务端挂起，有数据返回  │ 服务端单向推送         │
 * │ 实时性             │ 差（取决于轮询间隔）   │ 较好                    │ 好                     │
 * │ 服务器压力         │ 高                     │ 中                      │ 低                     │
 * │ 实现复杂度         │ 低                     │ 中                      │ 低                     │
 * │ 兼容性             │ 最好                   │ 好                      │ 好（IE 不支持）        │
 * │ 双向通信           │ 否（需要两个连接）     │ 否                      │ 否                     │
 * └───────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
 *
 * ┌───────────────────┬────────────────────────┐
 * │ 方案               │ WebSocket              │
 * ├───────────────────┼────────────────────────┤
 * │ 原理               │ 全双工通信             │
 * │ 实时性             │ 最好                   │
 * │ 服务器压力         │ 低                     │
 * │ 实现复杂度         │ 中                     │
 * │ 兼容性             │ 好（IE10+）            │
 * │ 双向通信           │ 是                     │
 * └───────────────────┴────────────────────────┘
 */

/**
 * 📊 选型建议
 *
 * 短轮询：兼容性要求高，实时性要求低
 * 长轮询：需要兼容老浏览器，单向通知
 * SSE：单向服务端推送（通知、实时数据）
 * WebSocket：双向实时通信（聊天、协作、游戏）
 */

// ============================================================
// 2. 短轮询
// ============================================================

// 短轮询实现
function shortPolling(url: string, interval: number = 3000) {
  let timerId: ReturnType<typeof setInterval> | null = null;
  let stopped = false;

  async function poll() {
    if (stopped) return;
    
    try {
      const response = await fetch(url);
      const data = await response.json();
      console.log('Received:', data);
    } catch (error) {
      console.error('Polling error:', error);
    }
  }

  // 开始轮询
  function start() {
    stopped = false;
    poll(); // 立即执行一次
    timerId = setInterval(poll, interval);
  }

  // 停止轮询
  function stop() {
    stopped = true;
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
  }

  return { start, stop };
}

// ============================================================
// 3. 长轮询
// ============================================================

// 长轮询实现
async function longPolling(url: string, onMessage: (data: any) => void) {
  let stopped = false;

  async function poll() {
    while (!stopped) {
      try {
        const response = await fetch(url, {
          // 超时时间应该比服务端挂起时间长
          signal: AbortSignal.timeout(60000),
        });
        
        if (response.ok) {
          const data = await response.json();
          onMessage(data);
        }
      } catch (error) {
        if ((error as Error).name === 'TimeoutError') {
          // 超时是正常的，继续轮询
          continue;
        }
        console.error('Long polling error:', error);
        // 出错后等待一段时间再重试
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }
  }

  function start() {
    stopped = false;
    poll();
  }

  function stop() {
    stopped = true;
  }

  return { start, stop };
}

// ============================================================
// 4. SSE（Server-Sent Events）
// ============================================================

/**
 * 📊 SSE 特点
 *
 * - 基于 HTTP，单向通信（服务端 → 客户端）
 * - 自动重连
 * - 支持自定义事件类型
 * - 轻量级，适合服务端推送
 *
 * 使用场景：
 * - 实时通知
 * - 股票行情
 * - 日志流
 * - ChatGPT 流式响应
 */

// SSE 客户端实现
function createSSE(url: string) {
  const eventSource = new EventSource(url);

  // 默认消息
  eventSource.onmessage = (event) => {
    console.log('Message:', event.data);
  };

  // 连接打开
  eventSource.onopen = () => {
    console.log('SSE connected');
  };

  // 错误处理（会自动重连）
  eventSource.onerror = (error) => {
    console.error('SSE error:', error);
  };

  // 自定义事件
  eventSource.addEventListener('notification', (event) => {
    console.log('Notification:', event.data);
  });

  // 关闭连接
  function close() {
    eventSource.close();
  }

  return { eventSource, close };
}

// SSE 服务端响应格式
const sseServerExample = `
  // 响应头
  Content-Type: text/event-stream
  Cache-Control: no-cache
  Connection: keep-alive
  
  // 响应体格式
  data: Hello World                    // 默认消息
  
  event: notification                  // 自定义事件
  data: {"type": "new_message"}
  
  id: 123                              // 消息 ID（用于重连）
  data: Message with ID
  
  retry: 5000                          // 重连间隔（毫秒）
  
  // 多行数据
  data: Line 1
  data: Line 2
  
  // 消息之间用空行分隔
`;

// ============================================================
// 5. WebSocket 详解
// ============================================================

/**
 * 📊 WebSocket 特点
 *
 * - 全双工通信
 * - 基于 TCP，通过 HTTP 升级
 * - 没有同源限制
 * - 二进制和文本数据支持
 * - 低延迟
 *
 * 📊 WebSocket vs HTTP
 *
 * HTTP：
 * - 请求-响应模式
 * - 每次请求需要完整头部
 * - 单向通信
 *
 * WebSocket：
 * - 持久连接
 * - 头部开销小（2-14 字节）
 * - 双向通信
 */

/**
 * 📊 WebSocket 握手过程
 *
 * 1. 客户端发起 HTTP 请求（Upgrade）
 *
 * GET /chat HTTP/1.1
 * Host: example.com
 * Upgrade: websocket
 * Connection: Upgrade
 * Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
 * Sec-WebSocket-Version: 13
 *
 * 2. 服务端返回 101 状态码
 *
 * HTTP/1.1 101 Switching Protocols
 * Upgrade: websocket
 * Connection: Upgrade
 * Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
 *
 * 3. 协议升级完成，开始 WebSocket 通信
 *
 * ⚠️ 面试追问：
 * Q: Sec-WebSocket-Accept 是怎么计算的？
 * A: Base64(SHA1(Sec-WebSocket-Key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
 */

// WebSocket 封装（生产级）
class WebSocketClient {
  private url: string;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private heartbeatTimeout = 30000;
  private messageHandlers: Map<string, Function[]> = new Map();

  constructor(url: string) {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        resolve();
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.stopHeartbeat();
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  // 心跳机制
  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, this.heartbeatTimeout);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // 重连机制
  private handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    // 指数退避
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    
    setTimeout(() => {
      this.connect().catch(console.error);
    }, Math.min(delay, 30000)); // 最大 30 秒
  }

  // 消息处理
  private handleMessage(data: string) {
    try {
      const message = JSON.parse(data);
      
      // 心跳响应
      if (message.type === 'pong') {
        return;
      }

      // 分发消息
      const handlers = this.messageHandlers.get(message.type) || [];
      handlers.forEach(handler => handler(message.data));
    } catch (error) {
      console.error('Message parse error:', error);
    }
  }

  // 发送消息
  send(data: object) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  // 订阅消息
  on(type: string, handler: Function) {
    const handlers = this.messageHandlers.get(type) || [];
    handlers.push(handler);
    this.messageHandlers.set(type, handlers);
  }

  // 取消订阅
  off(type: string, handler: Function) {
    const handlers = this.messageHandlers.get(type) || [];
    const index = handlers.indexOf(handler);
    if (index > -1) {
      handlers.splice(index, 1);
    }
  }

  // 关闭连接
  close() {
    this.maxReconnectAttempts = 0; // 阻止重连
    this.stopHeartbeat();
    this.ws?.close();
  }
}

// ============================================================
// 6. Socket.IO
// ============================================================

/**
 * 📊 Socket.IO 特点
 *
 * - 封装了 WebSocket + 降级方案
 * - 自动重连
 * - 房间（Room）和命名空间（Namespace）
 * - 事件机制
 * - 广播支持
 *
 * 💡 面试追问：
 * Q: Socket.IO 和 WebSocket 的区别？
 * A:
 * - Socket.IO 不是 WebSocket 的封装，是独立协议
 * - Socket.IO 支持降级到轮询
 * - Socket.IO 有更多功能（房间、广播、ack）
 * - Socket.IO 客户端必须使用 Socket.IO 服务端
 */

// Socket.IO 使用示例
const socketIOExample = `
  // 客户端
  import { io } from "socket.io-client";
  
  const socket = io("http://localhost:3000", {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
  });
  
  // 连接
  socket.on("connect", () => {
    console.log("Connected:", socket.id);
  });
  
  // 监听事件
  socket.on("message", (data) => {
    console.log("Received:", data);
  });
  
  // 发送事件
  socket.emit("chat", { text: "Hello" });
  
  // 带回调的发送
  socket.emit("chat", { text: "Hello" }, (response) => {
    console.log("Ack:", response);
  });
  
  // 加入房间
  socket.emit("join-room", "room-123");
  
  // 离开房间
  socket.emit("leave-room", "room-123");
`;

// ============================================================
// 7. 实际业务场景
// ============================================================

/**
 * 📊 场景 1：即时聊天
 *
 * 技术方案：WebSocket / Socket.IO
 *
 * 关键点：
 * - 消息确认（已发送、已送达、已读）
 * - 离线消息存储
 * - 消息去重（clientId）
 * - 消息排序（时间戳 + 序号）
 * - 图片/文件走 HTTP 上传
 */

/**
 * 📊 场景 2：实时协作（如在线文档）
 *
 * 技术方案：WebSocket + OT/CRDT
 *
 * 关键点：
 * - 操作转换（Operational Transformation）
 * - 冲突解决
 * - 光标位置同步
 * - 撤销/重做
 */

/**
 * 📊 场景 3：实时通知
 *
 * 技术方案：SSE / 长轮询
 *
 * 关键点：
 * - 通知优先级
 * - 批量推送
 * - 已读状态同步
 */

/**
 * 📊 场景 4：实时数据看板
 *
 * 技术方案：WebSocket / SSE
 *
 * 关键点：
 * - 数据节流（不需要每条数据都推）
 * - 增量更新
 * - 断线重连后的数据同步
 */

// ============================================================
// 8. 高频面试题（增强版）
// ============================================================

/**
 * 题目 1：WebSocket 和 HTTP 的区别？
 *
 * 连接方式：
 * - HTTP：短连接，请求-响应模式
 * - WebSocket：长连接，持久化
 *
 * 通信模式：
 * - HTTP：单向，客户端发起
 * - WebSocket：双向，双方都可以发送
 *
 * 头部开销：
 * - HTTP：每次请求完整头部
 * - WebSocket：帧头部只有 2-14 字节
 *
 * 同源策略：
 * - HTTP：受限制
 * - WebSocket：不受限制
 */

/**
 * 题目 2：WebSocket 如何保持连接？
 *
 * 1. 心跳机制：
 *    - 定期发送心跳包（ping/pong）
 *    - 检测连接是否存活
 *    - 避免被防火墙/代理关闭
 *
 * 2. 重连机制：
 *    - 监听 close 事件
 *    - 指数退避重连
 *    - 最大重试次数限制
 *
 * 3. 状态同步：
 *    - 重连后同步丢失的数据
 *    - 消息序号/时间戳机制
 */

/**
 * 题目 3：SSE 和 WebSocket 怎么选？
 *
 * 选 SSE：
 * - 只需要服务端推送
 * - 需要自动重连
 * - 不需要二进制数据
 * - 更简单的实现
 *
 * 选 WebSocket：
 * - 需要双向通信
 * - 需要低延迟
 * - 需要传输二进制数据
 * - 高频数据交互
 *
 * 💡 实际案例：
 * - ChatGPT 流式响应用 SSE
 * - 微信网页版用 长轮询
 * - 实时聊天用 WebSocket
 */

/**
 * 题目 4：如何处理 WebSocket 断线重连？
 *
 * 1. 检测断线：
 *    - close 事件
 *    - 心跳超时
 *
 * 2. 重连策略：
 *    - 立即重连 + 指数退避
 *    - 最大重试次数
 *    - 网络恢复后重连（navigator.onLine）
 *
 * 3. 状态恢复：
 *    - 重连后发送最后收到的消息 ID
 *    - 服务端补发丢失的消息
 *    - 或客户端主动拉取
 *
 * 4. 用户提示：
 *    - 显示连接状态
 *    - 断线时禁止发送
 */

export {
  shortPolling,
  longPolling,
  createSSE,
  WebSocketClient,
  socketIOExample,
};

