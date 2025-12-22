/**
 * ============================================================
 * 📚 链表 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示链表在前端实际业务中的应用
 */

// ============================================================
// 链表节点定义
// ============================================================

class ListNode<T> {
  val: T;
  next: ListNode<T> | null = null;

  constructor(val: T) {
    this.val = val;
  }
}

class DoublyListNode<T> {
  val: T;
  prev: DoublyListNode<T> | null = null;
  next: DoublyListNode<T> | null = null;

  constructor(val: T) {
    this.val = val;
  }
}

// ============================================================
// 1. 撤销/重做系统（Undo/Redo）
// ============================================================

/**
 * 📝 业务场景：编辑器撤销重做
 *
 * 场景描述：
 * - 用户在编辑器中执行操作
 * - 支持 Ctrl+Z 撤销，Ctrl+Y 重做
 * - 用双向链表存储操作历史
 */
interface EditorState {
  content: string;
  cursorPosition: number;
}

class UndoRedoManager<T> {
  private current: DoublyListNode<T> | null = null;
  private maxHistory: number;

  constructor(initialState: T, maxHistory = 50) {
    this.current = new DoublyListNode(initialState);
    this.maxHistory = maxHistory;
  }

  /**
   * 执行新操作，添加到历史
   */
  push(state: T): void {
    const newNode = new DoublyListNode(state);

    if (this.current) {
      // 如果当前不是最新状态，清除后面的历史
      this.current.next = newNode;
      newNode.prev = this.current;
    }

    this.current = newNode;

    // 限制历史长度
    this.trimHistory();
  }

  /**
   * 撤销
   */
  undo(): T | null {
    if (this.current && this.current.prev) {
      this.current = this.current.prev;
      return this.current.val;
    }
    return null;
  }

  /**
   * 重做
   */
  redo(): T | null {
    if (this.current && this.current.next) {
      this.current = this.current.next;
      return this.current.val;
    }
    return null;
  }

  /**
   * 获取当前状态
   */
  getCurrentState(): T | null {
    return this.current?.val ?? null;
  }

  /**
   * 能否撤销
   */
  canUndo(): boolean {
    return this.current?.prev !== null;
  }

  /**
   * 能否重做
   */
  canRedo(): boolean {
    return this.current?.next !== null;
  }

  /**
   * 限制历史长度
   */
  private trimHistory(): void {
    let count = 0;
    let node = this.current;

    // 往前数
    while (node && count < this.maxHistory) {
      node = node.prev;
      count++;
    }

    // 截断
    if (node) {
      node.next!.prev = null;
    }
  }
}

// 使用示例
const editorHistory = new UndoRedoManager<EditorState>({
  content: '',
  cursorPosition: 0,
});

// 用户输入
editorHistory.push({ content: 'Hello', cursorPosition: 5 });
editorHistory.push({ content: 'Hello World', cursorPosition: 11 });

// 撤销
const prevState = editorHistory.undo();
// console.log(prevState); // { content: 'Hello', cursorPosition: 5 }

// 重做
const nextState = editorHistory.redo();
// console.log(nextState); // { content: 'Hello World', cursorPosition: 11 }

// ============================================================
// 2. 浏览器历史管理
// ============================================================

/**
 * 📝 业务场景：SPA 路由历史
 *
 * 场景描述：
 * - 模拟浏览器的前进/后退功能
 * - 支持跳转到指定页面
 */
interface HistoryEntry {
  url: string;
  title: string;
  timestamp: number;
}

class BrowserHistory {
  private current: DoublyListNode<HistoryEntry>;
  private length = 1;

  constructor(initialUrl: string, initialTitle: string) {
    this.current = new DoublyListNode({
      url: initialUrl,
      title: initialTitle,
      timestamp: Date.now(),
    });
  }

  /**
   * 访问新页面
   */
  visit(url: string, title: string): void {
    const newEntry = new DoublyListNode<HistoryEntry>({
      url,
      title,
      timestamp: Date.now(),
    });

    // 清除当前位置之后的历史
    this.current.next = newEntry;
    newEntry.prev = this.current;
    this.current = newEntry;
    this.length++;
  }

  /**
   * 后退 n 步
   */
  back(steps = 1): string {
    let moved = 0;
    while (moved < steps && this.current.prev) {
      this.current = this.current.prev;
      moved++;
    }
    return this.current.val.url;
  }

  /**
   * 前进 n 步
   */
  forward(steps = 1): string {
    let moved = 0;
    while (moved < steps && this.current.next) {
      this.current = this.current.next;
      moved++;
    }
    return this.current.val.url;
  }

  /**
   * 获取当前 URL
   */
  getCurrentUrl(): string {
    return this.current.val.url;
  }

  /**
   * 获取当前标题
   */
  getCurrentTitle(): string {
    return this.current.val.title;
  }

  /**
   * 能否后退
   */
  canGoBack(): boolean {
    return this.current.prev !== null;
  }

  /**
   * 能否前进
   */
  canGoForward(): boolean {
    return this.current.next !== null;
  }
}

// 使用示例
const browserHistory = new BrowserHistory('/', 'Home');
browserHistory.visit('/products', 'Products');
browserHistory.visit('/products/1', 'Product Detail');
browserHistory.back(); // '/products'
browserHistory.forward(); // '/products/1'

// ============================================================
// 3. 任务队列（优先级插入）
// ============================================================

/**
 * 📝 业务场景：请求队列
 *
 * 场景描述：
 * - 请求按优先级排序执行
 * - 支持取消特定请求
 */
interface Task<T> {
  id: string;
  priority: number;
  data: T;
}

class PriorityTaskQueue<T> {
  private head: ListNode<Task<T>> | null = null;
  private tail: ListNode<Task<T>> | null = null;
  private size = 0;

  /**
   * 添加任务（按优先级插入）
   */
  enqueue(task: Task<T>): void {
    const newNode = new ListNode(task);

    if (!this.head) {
      this.head = this.tail = newNode;
    } else {
      // 找到插入位置（优先级高的在前）
      let current = this.head;
      let prev: ListNode<Task<T>> | null = null;

      while (current && current.val.priority >= task.priority) {
        prev = current;
        current = current.next!;
      }

      if (!prev) {
        // 插入到头部
        newNode.next = this.head;
        this.head = newNode;
      } else if (!current) {
        // 插入到尾部
        prev.next = newNode;
        this.tail = newNode;
      } else {
        // 插入到中间
        prev.next = newNode;
        newNode.next = current;
      }
    }

    this.size++;
  }

  /**
   * 取出最高优先级任务
   */
  dequeue(): Task<T> | null {
    if (!this.head) return null;

    const task = this.head.val;
    this.head = this.head.next;

    if (!this.head) {
      this.tail = null;
    }

    this.size--;
    return task;
  }

  /**
   * 取消指定任务
   */
  cancel(taskId: string): boolean {
    if (!this.head) return false;

    // 特殊处理头节点
    if (this.head.val.id === taskId) {
      this.head = this.head.next;
      if (!this.head) this.tail = null;
      this.size--;
      return true;
    }

    // 遍历查找
    let current = this.head;
    while (current.next) {
      if (current.next.val.id === taskId) {
        current.next = current.next.next;
        if (!current.next) this.tail = current;
        this.size--;
        return true;
      }
      current = current.next;
    }

    return false;
  }

  /**
   * 获取队列长度
   */
  getSize(): number {
    return this.size;
  }

  /**
   * 队列是否为空
   */
  isEmpty(): boolean {
    return this.size === 0;
  }
}

// 使用示例
const taskQueue = new PriorityTaskQueue<{ url: string }>();
taskQueue.enqueue({ id: '1', priority: 1, data: { url: '/api/low' } });
taskQueue.enqueue({ id: '2', priority: 3, data: { url: '/api/high' } });
taskQueue.enqueue({ id: '3', priority: 2, data: { url: '/api/medium' } });

// console.log(taskQueue.dequeue()); // priority: 3
// console.log(taskQueue.dequeue()); // priority: 2

// ============================================================
// 4. 简化版 React Fiber 结构
// ============================================================

/**
 * 📝 业务场景：React Fiber 架构模拟
 *
 * 场景描述：
 * - Fiber 节点形成链表结构
 * - 支持可中断的渲染
 */
interface FiberNode {
  type: string;
  props: Record<string, unknown>;
  child: FiberNode | null;
  sibling: FiberNode | null;
  return: FiberNode | null; // 父节点
  stateNode: unknown; // 真实 DOM
}

function createFiber(
  type: string,
  props: Record<string, unknown>
): FiberNode {
  return {
    type,
    props,
    child: null,
    sibling: null,
    return: null,
    stateNode: null,
  };
}

/**
 * 遍历 Fiber 树（深度优先）
 * 模拟 React 的 workLoop
 */
function* walkFiber(root: FiberNode): Generator<FiberNode> {
  let current: FiberNode | null = root;

  while (current) {
    yield current;

    // 先处理子节点
    if (current.child) {
      current = current.child;
      continue;
    }

    // 没有子节点，找兄弟节点
    while (current) {
      // 没有兄弟节点，回到父节点
      if (!current.sibling) {
        current = current.return;
        // 如果回到了根节点的父节点（null），结束
        if (!current || current === root.return) {
          return;
        }
        continue;
      }

      // 有兄弟节点，处理兄弟节点
      current = current.sibling;
      break;
    }
  }
}

// 使用示例
const appFiber = createFiber('div', { className: 'app' });
const headerFiber = createFiber('header', {});
const mainFiber = createFiber('main', {});

appFiber.child = headerFiber;
headerFiber.return = appFiber;
headerFiber.sibling = mainFiber;
mainFiber.return = appFiber;

// for (const fiber of walkFiber(appFiber)) {
//   console.log(fiber.type);
// }

// ============================================================
// 5. 播放列表（循环链表）
// ============================================================

/**
 * 📝 业务场景：音乐播放器
 *
 * 场景描述：
 * - 支持顺序播放、单曲循环、列表循环
 * - 支持上一首、下一首
 */
interface Track {
  id: string;
  title: string;
  artist: string;
  duration: number;
}

class Playlist {
  private head: ListNode<Track> | null = null;
  private current: ListNode<Track> | null = null;
  private size = 0;
  private loop: 'none' | 'single' | 'all' = 'none';

  /**
   * 添加歌曲到末尾
   */
  add(track: Track): void {
    const newNode = new ListNode(track);

    if (!this.head) {
      this.head = newNode;
      this.current = newNode;
    } else {
      // 找到末尾
      let tail = this.head;
      while (tail.next) {
        tail = tail.next;
      }
      tail.next = newNode;
    }

    this.size++;
  }

  /**
   * 播放指定歌曲
   */
  play(trackId: string): Track | null {
    let node = this.head;
    while (node) {
      if (node.val.id === trackId) {
        this.current = node;
        return node.val;
      }
      node = node.next;
    }
    return null;
  }

  /**
   * 下一首
   */
  next(): Track | null {
    if (!this.current) return null;

    if (this.loop === 'single') {
      return this.current.val;
    }

    if (this.current.next) {
      this.current = this.current.next;
    } else if (this.loop === 'all') {
      this.current = this.head;
    } else {
      return null; // 播放结束
    }

    return this.current?.val ?? null;
  }

  /**
   * 上一首
   */
  prev(): Track | null {
    if (!this.current || !this.head) return null;

    // 找到前一个节点
    if (this.current === this.head) {
      if (this.loop === 'all') {
        // 找到最后一个
        let tail = this.head;
        while (tail.next) {
          tail = tail.next;
        }
        this.current = tail;
      }
    } else {
      let node = this.head;
      while (node.next && node.next !== this.current) {
        node = node.next;
      }
      this.current = node;
    }

    return this.current.val;
  }

  /**
   * 设置循环模式
   */
  setLoop(mode: 'none' | 'single' | 'all'): void {
    this.loop = mode;
  }

  /**
   * 获取当前歌曲
   */
  getCurrent(): Track | null {
    return this.current?.val ?? null;
  }
}

// ============================================================
// 6. 消息链（责任链模式）
// ============================================================

/**
 * 📝 业务场景：中间件/拦截器
 *
 * 场景描述：
 * - 请求经过多个处理器
 * - 每个处理器决定是否继续传递
 */
type NextFunction = () => void;
type Handler<T> = (data: T, next: NextFunction) => void;

class MiddlewareChain<T> {
  private head: ListNode<Handler<T>> | null = null;
  private tail: ListNode<Handler<T>> | null = null;

  /**
   * 添加中间件
   */
  use(handler: Handler<T>): this {
    const node = new ListNode(handler);

    if (!this.head) {
      this.head = this.tail = node;
    } else {
      this.tail!.next = node;
      this.tail = node;
    }

    return this;
  }

  /**
   * 执行中间件链
   */
  execute(data: T): void {
    const dispatch = (node: ListNode<Handler<T>> | null): void => {
      if (!node) return;

      const handler = node.val;
      handler(data, () => dispatch(node.next));
    };

    dispatch(this.head);
  }
}

// 使用示例
interface RequestContext {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: unknown;
}

const middlewares = new MiddlewareChain<RequestContext>();

// 添加日志中间件
middlewares.use((ctx, next) => {
  console.log(`${ctx.method} ${ctx.url}`);
  next();
});

// 添加认证中间件
middlewares.use((ctx, next) => {
  if (ctx.headers['authorization']) {
    next();
  } else {
    console.log('Unauthorized');
  }
});

// 执行
// middlewares.execute({
//   url: '/api/users',
//   method: 'GET',
//   headers: { authorization: 'Bearer xxx' }
// });

// ============================================================
// 导出
// ============================================================

export {
  ListNode,
  DoublyListNode,
  UndoRedoManager,
  BrowserHistory,
  PriorityTaskQueue,
  createFiber,
  walkFiber,
  Playlist,
  MiddlewareChain,
};

