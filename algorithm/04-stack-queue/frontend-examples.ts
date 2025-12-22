/**
 * ============================================================
 * 📚 栈与队列 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示栈与队列在前端实际业务中的应用
 */

// ============================================================
// 1. 撤销/重做功能（两个栈）
// ============================================================

/**
 * 📝 业务场景：编辑器撤销重做
 *
 * 场景描述：
 * - 用户执行操作后可以撤销
 * - 撤销后可以重做
 * - 新操作会清空重做栈
 */
class UndoRedoStack<T> {
  private undoStack: T[] = [];
  private redoStack: T[] = [];
  private current: T;

  constructor(initialState: T) {
    this.current = initialState;
  }

  /**
   * 执行新操作
   */
  execute(newState: T): void {
    this.undoStack.push(this.current);
    this.current = newState;
    this.redoStack = []; // 清空重做栈
  }

  /**
   * 撤销
   */
  undo(): T | null {
    if (this.undoStack.length === 0) return null;

    this.redoStack.push(this.current);
    this.current = this.undoStack.pop()!;
    return this.current;
  }

  /**
   * 重做
   */
  redo(): T | null {
    if (this.redoStack.length === 0) return null;

    this.undoStack.push(this.current);
    this.current = this.redoStack.pop()!;
    return this.current;
  }

  /**
   * 获取当前状态
   */
  getCurrentState(): T {
    return this.current;
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  canRedo(): boolean {
    return this.redoStack.length > 0;
  }
}

// 使用示例
const editor = new UndoRedoStack<string>('');
editor.execute('Hello');
editor.execute('Hello World');
editor.undo(); // 'Hello'
editor.redo(); // 'Hello World'

// ============================================================
// 2. 括号/标签匹配检查
// ============================================================

/**
 * 📝 业务场景：代码编辑器语法检查
 *
 * 场景描述：
 * - 检查 HTML 标签是否正确闭合
 * - 检查括号是否匹配
 */
interface MatchResult {
  valid: boolean;
  error?: {
    message: string;
    position: number;
  };
}

function checkBrackets(code: string): MatchResult {
  const stack: { char: string; pos: number }[] = [];
  const pairs: Record<string, string> = {
    ')': '(',
    ']': '[',
    '}': '{',
  };
  const openBrackets = new Set(['(', '[', '{']);

  for (let i = 0; i < code.length; i++) {
    const char = code[i];

    if (openBrackets.has(char)) {
      stack.push({ char, pos: i });
    } else if (char in pairs) {
      if (stack.length === 0) {
        return {
          valid: false,
          error: { message: `多余的右括号 '${char}'`, position: i },
        };
      }

      const top = stack.pop()!;
      if (top.char !== pairs[char]) {
        return {
          valid: false,
          error: {
            message: `括号不匹配: '${top.char}' 与 '${char}'`,
            position: i,
          },
        };
      }
    }
  }

  if (stack.length > 0) {
    const unclosed = stack.pop()!;
    return {
      valid: false,
      error: {
        message: `未闭合的括号 '${unclosed.char}'`,
        position: unclosed.pos,
      },
    };
  }

  return { valid: true };
}

/**
 * 检查 HTML 标签是否正确闭合
 */
function checkHtmlTags(html: string): MatchResult {
  const stack: { tag: string; pos: number }[] = [];
  const selfClosingTags = new Set([
    'br',
    'hr',
    'img',
    'input',
    'meta',
    'link',
  ]);

  const tagRegex = /<\/?([a-zA-Z][a-zA-Z0-9]*)[^>]*\/?>/g;
  let match;

  while ((match = tagRegex.exec(html)) !== null) {
    const fullTag = match[0];
    const tagName = match[1].toLowerCase();
    const pos = match.index;

    // 跳过自闭合标签
    if (selfClosingTags.has(tagName) || fullTag.endsWith('/>')) {
      continue;
    }

    if (fullTag.startsWith('</')) {
      // 闭合标签
      if (stack.length === 0) {
        return {
          valid: false,
          error: { message: `多余的闭合标签 </${tagName}>`, position: pos },
        };
      }

      const top = stack.pop()!;
      if (top.tag !== tagName) {
        return {
          valid: false,
          error: {
            message: `标签不匹配: <${top.tag}> 与 </${tagName}>`,
            position: pos,
          },
        };
      }
    } else {
      // 开始标签
      stack.push({ tag: tagName, pos });
    }
  }

  if (stack.length > 0) {
    const unclosed = stack.pop()!;
    return {
      valid: false,
      error: {
        message: `未闭合的标签 <${unclosed.tag}>`,
        position: unclosed.pos,
      },
    };
  }

  return { valid: true };
}

// ============================================================
// 3. 任务队列与限流
// ============================================================

/**
 * 📝 业务场景：API 请求限流
 *
 * 场景描述：
 * - 限制并发请求数量
 * - 超出限制的请求排队等待
 */
class RequestQueue<T> {
  private queue: (() => Promise<T>)[] = [];
  private running = 0;
  private maxConcurrent: number;

  constructor(maxConcurrent = 3) {
    this.maxConcurrent = maxConcurrent;
  }

  /**
   * 添加请求到队列
   */
  async add(requestFn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      const task = async () => {
        try {
          const result = await requestFn();
          resolve(result);
        } catch (error) {
          reject(error);
        } finally {
          this.running--;
          this.processNext();
        }
      };

      this.queue.push(task);
      this.processNext();
    });
  }

  private processNext(): void {
    while (this.running < this.maxConcurrent && this.queue.length > 0) {
      const task = this.queue.shift()!;
      this.running++;
      task();
    }
  }

  /**
   * 获取队列状态
   */
  getStatus(): { pending: number; running: number } {
    return {
      pending: this.queue.length,
      running: this.running,
    };
  }
}

// 使用示例
const requestQueue = new RequestQueue(3);

async function fetchData(id: number): Promise<string> {
  return requestQueue.add(async () => {
    const response = await fetch(`/api/data/${id}`);
    return response.json();
  });
}

// ============================================================
// 4. 消息通知队列
// ============================================================

/**
 * 📝 业务场景：Toast 通知
 *
 * 场景描述：
 * - 通知按顺序显示
 * - 每个通知显示固定时间
 * - 支持手动关闭
 */
interface Toast {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  duration?: number;
}

class ToastQueue {
  private queue: Toast[] = [];
  private current: Toast | null = null;
  private timer: NodeJS.Timeout | null = null;
  private onShow?: (toast: Toast) => void;
  private onHide?: (toast: Toast) => void;

  constructor(options?: {
    onShow?: (toast: Toast) => void;
    onHide?: (toast: Toast) => void;
  }) {
    this.onShow = options?.onShow;
    this.onHide = options?.onHide;
  }

  /**
   * 添加通知
   */
  show(toast: Omit<Toast, 'id'>): string {
    const id = Math.random().toString(36).substring(2);
    const newToast: Toast = { ...toast, id, duration: toast.duration ?? 3000 };

    this.queue.push(newToast);
    this.processNext();

    return id;
  }

  /**
   * 关闭当前通知
   */
  close(): void {
    if (!this.current) return;

    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    this.onHide?.(this.current);
    this.current = null;
    this.processNext();
  }

  private processNext(): void {
    if (this.current || this.queue.length === 0) return;

    this.current = this.queue.shift()!;
    this.onShow?.(this.current);

    if (this.current.duration && this.current.duration > 0) {
      this.timer = setTimeout(() => this.close(), this.current.duration);
    }
  }
}

// 使用示例
const toastQueue = new ToastQueue({
  onShow: (toast) => console.log('显示:', toast.message),
  onHide: (toast) => console.log('隐藏:', toast.message),
});

// toastQueue.show({ message: '操作成功', type: 'success' });

// ============================================================
// 5. 路径解析（栈处理 ..）
// ============================================================

/**
 * 📝 业务场景：URL 路径标准化
 *
 * 场景描述：
 * - 处理 . 和 .. 等特殊路径
 * - 合并重复的 /
 */
function normalizePath(path: string): string {
  const stack: string[] = [];
  const parts = path.split('/').filter((p) => p && p !== '.');

  for (const part of parts) {
    if (part === '..') {
      if (stack.length > 0 && stack[stack.length - 1] !== '..') {
        stack.pop();
      } else if (!path.startsWith('/')) {
        stack.push('..');
      }
    } else {
      stack.push(part);
    }
  }

  const result = stack.join('/');
  return path.startsWith('/') ? '/' + result : result || '.';
}

// 使用示例
// normalizePath('/a/b/../c/./d') => '/a/c/d'
// normalizePath('a/b/../c') => 'a/c'

// ============================================================
// 6. 表达式计算器
// ============================================================

/**
 * 📝 业务场景：简易计算器/公式引擎
 *
 * 场景描述：
 * - 支持 +、-、*、/ 和括号
 * - 正确处理优先级
 */
function calculate(expression: string): number {
  const tokens = tokenize(expression);
  const postfix = infixToPostfix(tokens);
  return evaluatePostfix(postfix);
}

function tokenize(expr: string): string[] {
  const tokens: string[] = [];
  let num = '';

  for (const char of expr) {
    if (/\d/.test(char)) {
      num += char;
    } else {
      if (num) {
        tokens.push(num);
        num = '';
      }
      if (char !== ' ') {
        tokens.push(char);
      }
    }
  }

  if (num) tokens.push(num);
  return tokens;
}

function infixToPostfix(tokens: string[]): string[] {
  const output: string[] = [];
  const stack: string[] = [];
  const precedence: Record<string, number> = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
  };

  for (const token of tokens) {
    if (/\d+/.test(token)) {
      output.push(token);
    } else if (token === '(') {
      stack.push(token);
    } else if (token === ')') {
      while (stack.length && stack[stack.length - 1] !== '(') {
        output.push(stack.pop()!);
      }
      stack.pop(); // 弹出 '('
    } else if (token in precedence) {
      while (
        stack.length &&
        stack[stack.length - 1] in precedence &&
        precedence[stack[stack.length - 1]] >= precedence[token]
      ) {
        output.push(stack.pop()!);
      }
      stack.push(token);
    }
  }

  while (stack.length) {
    output.push(stack.pop()!);
  }

  return output;
}

function evaluatePostfix(postfix: string[]): number {
  const stack: number[] = [];

  for (const token of postfix) {
    if (/\d+/.test(token)) {
      stack.push(parseInt(token));
    } else {
      const b = stack.pop()!;
      const a = stack.pop()!;
      switch (token) {
        case '+':
          stack.push(a + b);
          break;
        case '-':
          stack.push(a - b);
          break;
        case '*':
          stack.push(a * b);
          break;
        case '/':
          stack.push(Math.trunc(a / b));
          break;
      }
    }
  }

  return stack[0];
}

// 使用示例
// calculate('3 + 4 * 2') => 11
// calculate('(3 + 4) * 2') => 14

// ============================================================
// 7. DOM 层序遍历（队列实现 BFS）
// ============================================================

/**
 * 📝 业务场景：DOM 树遍历
 *
 * 场景描述：
 * - 广度优先遍历 DOM 树
 * - 按层级收集节点
 */
function traverseDOMByLevel(root: Element): Element[][] {
  const result: Element[][] = [];
  const queue: Element[] = [root];

  while (queue.length > 0) {
    const levelSize = queue.length;
    const currentLevel: Element[] = [];

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift()!;
      currentLevel.push(node);

      // 子节点入队
      for (const child of Array.from(node.children)) {
        queue.push(child);
      }
    }

    result.push(currentLevel);
  }

  return result;
}

/**
 * 查找特定元素（BFS）
 */
function findElement(
  root: Element,
  predicate: (el: Element) => boolean
): Element | null {
  const queue: Element[] = [root];

  while (queue.length > 0) {
    const node = queue.shift()!;

    if (predicate(node)) {
      return node;
    }

    for (const child of Array.from(node.children)) {
      queue.push(child);
    }
  }

  return null;
}

// ============================================================
// 8. 事件循环模拟
// ============================================================

/**
 * 📝 业务场景：理解 JavaScript 事件循环
 *
 * 场景描述：
 * - 模拟微任务和宏任务队列
 * - 理解执行顺序
 */
class EventLoopSimulator {
  private macroTaskQueue: (() => void)[] = [];
  private microTaskQueue: (() => void)[] = [];

  /**
   * 添加宏任务（类似 setTimeout）
   */
  addMacroTask(task: () => void): void {
    this.macroTaskQueue.push(task);
  }

  /**
   * 添加微任务（类似 Promise.then）
   */
  addMicroTask(task: () => void): void {
    this.microTaskQueue.push(task);
  }

  /**
   * 执行一轮事件循环
   */
  tick(): void {
    // 1. 执行所有微任务
    while (this.microTaskQueue.length > 0) {
      const task = this.microTaskQueue.shift()!;
      task();
    }

    // 2. 执行一个宏任务
    if (this.macroTaskQueue.length > 0) {
      const task = this.macroTaskQueue.shift()!;
      task();
    }
  }

  /**
   * 运行直到队列清空
   */
  run(): void {
    while (
      this.macroTaskQueue.length > 0 ||
      this.microTaskQueue.length > 0
    ) {
      this.tick();
    }
  }
}

// ============================================================
// 导出
// ============================================================

export {
  UndoRedoStack,
  checkBrackets,
  checkHtmlTags,
  RequestQueue,
  ToastQueue,
  normalizePath,
  calculate,
  traverseDOMByLevel,
  findElement,
  EventLoopSimulator,
};

