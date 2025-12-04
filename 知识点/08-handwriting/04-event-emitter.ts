/**
 * ============================================================
 * 📚 事件发布订阅
 * ============================================================
 *
 * 面试考察重点：
 * 1. EventEmitter 实现
 * 2. 观察者模式 vs 发布订阅模式
 * 3. 异步事件处理
 * 4. 内存泄漏防护
 */

// ============================================================
// 1. 基础 EventEmitter
// ============================================================

type EventHandler = (...args: any[]) => void;

class EventEmitter {
  private events: Map<string, EventHandler[]> = new Map();

  // 订阅事件
  on(event: string, handler: EventHandler): this {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(handler);
    return this;
  }

  // 取消订阅
  off(event: string, handler: EventHandler): this {
    const handlers = this.events.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
    return this;
  }

  // 触发事件
  emit(event: string, ...args: any[]): boolean {
    const handlers = this.events.get(event);
    if (!handlers || handlers.length === 0) {
      return false;
    }
    handlers.forEach(handler => handler(...args));
    return true;
  }

  // 只订阅一次
  once(event: string, handler: EventHandler): this {
    const wrapper = (...args: any[]) => {
      handler(...args);
      this.off(event, wrapper);
    };
    return this.on(event, wrapper);
  }

  // 移除某个事件的所有监听器
  removeAllListeners(event?: string): this {
    if (event) {
      this.events.delete(event);
    } else {
      this.events.clear();
    }
    return this;
  }

  // 获取监听器数量
  listenerCount(event: string): number {
    return this.events.get(event)?.length || 0;
  }

  // 获取所有事件名
  eventNames(): string[] {
    return Array.from(this.events.keys());
  }
}

// ============================================================
// 2. 增强版 EventEmitter
// ============================================================

interface EventEmitterOptions {
  maxListeners?: number;       // 最大监听器数量
  captureRejections?: boolean; // 捕获异步错误
}

class EnhancedEventEmitter {
  private events: Map<string | symbol, EventHandler[]> = new Map();
  private maxListeners: number;
  private captureRejections: boolean;

  constructor(options: EventEmitterOptions = {}) {
    this.maxListeners = options.maxListeners ?? 10;
    this.captureRejections = options.captureRejections ?? false;
  }

  on(event: string | symbol, handler: EventHandler): this {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }

    const handlers = this.events.get(event)!;

    // 检查最大监听器数量
    if (handlers.length >= this.maxListeners) {
      console.warn(
        `MaxListenersExceededWarning: Possible EventEmitter memory leak detected. ` +
        `${handlers.length + 1} ${String(event)} listeners added.`
      );
    }

    handlers.push(handler);
    return this;
  }

  // 添加到监听器数组开头
  prependListener(event: string | symbol, handler: EventHandler): this {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.unshift(handler);
    return this;
  }

  off(event: string | symbol, handler: EventHandler): this {
    const handlers = this.events.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
    return this;
  }

  emit(event: string | symbol, ...args: any[]): boolean {
    const handlers = this.events.get(event);
    if (!handlers || handlers.length === 0) {
      return false;
    }

    // 复制数组，防止在回调中修改
    const handlersToCall = [...handlers];

    for (const handler of handlersToCall) {
      try {
        const result = handler(...args);

        // 处理异步错误
        if (this.captureRejections && result instanceof Promise) {
          result.catch(error => {
            this.emit('error', error);
          });
        }
      } catch (error) {
        // 同步错误
        if (event !== 'error') {
          this.emit('error', error);
        } else {
          throw error;
        }
      }
    }

    return true;
  }

  once(event: string | symbol, handler: EventHandler): this {
    const wrapper = (...args: any[]) => {
      this.off(event, wrapper);
      handler(...args);
    };
    // 保存原始 handler 引用，方便 off
    (wrapper as any).listener = handler;
    return this.on(event, wrapper);
  }

  // 异步等待事件
  waitFor(event: string | symbol, timeout?: number): Promise<any[]> {
    return new Promise((resolve, reject) => {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;

      const handler = (...args: any[]) => {
        if (timeoutId) clearTimeout(timeoutId);
        resolve(args);
      };

      this.once(event, handler);

      if (timeout) {
        timeoutId = setTimeout(() => {
          this.off(event, handler);
          reject(new Error(`Timeout waiting for ${String(event)}`));
        }, timeout);
      }
    });
  }

  setMaxListeners(n: number): this {
    this.maxListeners = n;
    return this;
  }

  getMaxListeners(): number {
    return this.maxListeners;
  }
}

// ============================================================
// 3. 类型安全的 EventEmitter
// ============================================================

type EventMap = Record<string, any[]>;

class TypedEventEmitter<T extends EventMap> {
  private events: Map<keyof T, Function[]> = new Map();

  on<K extends keyof T>(event: K, handler: (...args: T[K]) => void): this {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(handler);
    return this;
  }

  off<K extends keyof T>(event: K, handler: (...args: T[K]) => void): this {
    const handlers = this.events.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
    return this;
  }

  emit<K extends keyof T>(event: K, ...args: T[K]): boolean {
    const handlers = this.events.get(event);
    if (!handlers || handlers.length === 0) {
      return false;
    }
    handlers.forEach(handler => handler(...args));
    return true;
  }

  once<K extends keyof T>(event: K, handler: (...args: T[K]) => void): this {
    const wrapper = (...args: T[K]) => {
      handler(...args);
      this.off(event, wrapper);
    };
    return this.on(event, wrapper);
  }
}

// 使用示例
interface MyEvents {
  'user:login': [userId: string, timestamp: number];
  'user:logout': [userId: string];
  'message': [content: string, from: string, to: string];
}

const typedEmitter = new TypedEventEmitter<MyEvents>();

// 类型安全的使用
typedEmitter.on('user:login', (userId, timestamp) => {
  console.log(`User ${userId} logged in at ${timestamp}`);
});

typedEmitter.emit('user:login', 'user123', Date.now());

// ============================================================
// 4. 观察者模式
// ============================================================

/**
 * 📊 观察者模式 vs 发布订阅模式
 *
 * 观察者模式：
 * - Subject 直接持有 Observer 引用
 * - 耦合度较高
 *
 * 发布订阅模式：
 * - 通过 EventEmitter 中介
 * - 发布者和订阅者完全解耦
 */

interface Observer {
  update(data: any): void;
}

class Subject {
  private observers: Set<Observer> = new Set();

  attach(observer: Observer): void {
    this.observers.add(observer);
  }

  detach(observer: Observer): void {
    this.observers.delete(observer);
  }

  notify(data: any): void {
    this.observers.forEach(observer => observer.update(data));
  }
}

// 使用示例
class ConcreteObserver implements Observer {
  private name: string;

  constructor(name: string) {
    this.name = name;
  }

  update(data: any): void {
    console.log(`${this.name} received:`, data);
  }
}

// ============================================================
// 5. DOM 事件委托
// ============================================================

/**
 * 📊 事件委托实现
 */

class DOMEventDelegator {
  private root: HTMLElement;
  private handlers: Map<string, Map<string, EventHandler>> = new Map();

  constructor(root: HTMLElement) {
    this.root = root;
  }

  on(eventType: string, selector: string, handler: EventHandler): this {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Map());

      // 在 root 上添加事件监听
      this.root.addEventListener(eventType, (e) => {
        this.handleEvent(eventType, e);
      });
    }

    this.handlers.get(eventType)!.set(selector, handler);
    return this;
  }

  off(eventType: string, selector: string): this {
    this.handlers.get(eventType)?.delete(selector);
    return this;
  }

  private handleEvent(eventType: string, event: Event): void {
    const target = event.target as HTMLElement;
    const handlers = this.handlers.get(eventType);

    if (!handlers) return;

    handlers.forEach((handler, selector) => {
      // 向上查找匹配的元素
      let element: HTMLElement | null = target;

      while (element && element !== this.root) {
        if (element.matches(selector)) {
          handler.call(element, event);
          break;
        }
        element = element.parentElement;
      }
    });
  }
}

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 忘记移除监听器导致内存泄漏
 *    - 组件卸载时要 off
 *    - 使用 once 或自动清理
 *
 * 2. 在回调中修改监听器数组
 *    - 在 emit 时复制数组
 *
 * 3. this 指向问题
 *    - 箭头函数或 bind
 *
 * 4. 同步 emit 导致栈溢出
 *    - 事件循环中 emit 同一事件
 *    - 使用异步 emit
 *
 * 5. 错误处理
 *    - 一个回调报错不应影响其他
 *    - try-catch 包装
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 观察者模式和发布订阅模式的区别？
 * A:
 *    观察者模式：
 *    - Subject 直接通知 Observer
 *    - 耦合度高
 *
 *    发布订阅模式：
 *    - 通过事件中心
 *    - 完全解耦
 *
 * Q2: 如何防止内存泄漏？
 * A:
 *    - 组件卸载时 off
 *    - 使用 WeakMap 存储
 *    - 设置最大监听器数量警告
 *
 * Q3: 如何实现事件优先级？
 * A:
 *    - 使用优先级队列存储 handler
 *    - emit 时按优先级排序执行
 *
 * Q4: Vue 的响应式和 EventEmitter 的关系？
 * A:
 *    - Vue 的响应式基于观察者模式
 *    - Dep 是 Subject
 *    - Watcher 是 Observer
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：全局事件总线
 */

const eventBusExample = `
// 创建全局事件总线
const eventBus = new EventEmitter();

// 组件 A 订阅
eventBus.on('user:update', (user) => {
  console.log('User updated:', user);
});

// 组件 B 发布
eventBus.emit('user:update', { id: 1, name: 'Tom' });

// React 中使用
useEffect(() => {
  const handler = (data) => setData(data);
  eventBus.on('data:change', handler);

  return () => {
    eventBus.off('data:change', handler);
  };
}, []);
`;

/**
 * 🏢 场景 2：WebSocket 消息处理
 */

const websocketExample = `
class WebSocketClient extends EventEmitter {
  private ws: WebSocket;

  constructor(url: string) {
    super();
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.emit(data.type, data.payload);
    };

    this.ws.onopen = () => this.emit('connected');
    this.ws.onclose = () => this.emit('disconnected');
    this.ws.onerror = (e) => this.emit('error', e);
  }

  send(type: string, payload: any) {
    this.ws.send(JSON.stringify({ type, payload }));
  }
}

// 使用
const client = new WebSocketClient('wss://api.example.com');

client.on('connected', () => console.log('Connected'));
client.on('message', (data) => console.log('Message:', data));
client.on('notification', (data) => showNotification(data));
`;

/**
 * 🏢 场景 3：插件系统
 */

const pluginSystemExample = `
class PluginSystem extends EventEmitter {
  private plugins: Map<string, any> = new Map();

  register(name: string, plugin: any) {
    this.plugins.set(name, plugin);

    // 触发插件初始化钩子
    if (plugin.init) {
      plugin.init(this);
    }

    this.emit('plugin:registered', name, plugin);
  }

  unregister(name: string) {
    const plugin = this.plugins.get(name);

    if (plugin?.destroy) {
      plugin.destroy();
    }

    this.plugins.delete(name);
    this.emit('plugin:unregistered', name);
  }
}

// 插件定义
const myPlugin = {
  init(system: PluginSystem) {
    system.on('data:process', this.process);
  },

  process(data: any) {
    // 处理数据
    return transformedData;
  },

  destroy() {
    // 清理资源
  },
};
`;

export {
  EventEmitter,
  EnhancedEventEmitter,
  TypedEventEmitter,
  Subject,
  ConcreteObserver,
  DOMEventDelegator,
  eventBusExample,
  websocketExample,
  pluginSystemExample,
};

