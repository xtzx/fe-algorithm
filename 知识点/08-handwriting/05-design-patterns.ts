/**
 * ============================================================
 * 📚 设计模式
 * ============================================================
 *
 * 面试考察重点：
 * 1. 单例模式
 * 2. 工厂模式
 * 3. 代理模式
 * 4. 策略模式
 * 5. 装饰器模式
 */

// ============================================================
// 1. 单例模式（Singleton）
// ============================================================

/**
 * 📊 单例模式
 *
 * 保证一个类只有一个实例
 *
 * 场景：
 * - 全局状态管理
 * - 配置对象
 * - 数据库连接
 */

// 基础单例
class Singleton {
  private static instance: Singleton;
  private data: any;

  private constructor() {
    this.data = {};
  }

  static getInstance(): Singleton {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }

  getData(): any {
    return this.data;
  }

  setData(data: any): void {
    this.data = data;
  }
}

// 懒加载单例（使用闭包）
const createSingleton = <T>(createInstance: () => T): (() => T) => {
  let instance: T | null = null;

  return () => {
    if (instance === null) {
      instance = createInstance();
    }
    return instance;
  };
};

// 使用示例
const getLogger = createSingleton(() => ({
  log: (msg: string) => console.log(`[LOG] ${msg}`),
  error: (msg: string) => console.error(`[ERROR] ${msg}`),
}));

// ============================================================
// 2. 工厂模式（Factory）
// ============================================================

/**
 * 📊 工厂模式
 *
 * 封装对象创建过程
 *
 * 场景：
 * - 创建复杂对象
 * - 根据条件创建不同对象
 */

// 简单工厂
interface Button {
  render(): string;
}

class WindowsButton implements Button {
  render(): string {
    return '<button class="windows">Windows Button</button>';
  }
}

class MacButton implements Button {
  render(): string {
    return '<button class="mac">Mac Button</button>';
  }
}

class ButtonFactory {
  static create(os: 'windows' | 'mac'): Button {
    switch (os) {
      case 'windows':
        return new WindowsButton();
      case 'mac':
        return new MacButton();
      default:
        throw new Error(`Unknown OS: ${os}`);
    }
  }
}

// 抽象工厂
interface GUIFactory {
  createButton(): Button;
  createCheckbox(): Checkbox;
}

interface Checkbox {
  render(): string;
}

class WindowsCheckbox implements Checkbox {
  render(): string {
    return '<input type="checkbox" class="windows">';
  }
}

class MacCheckbox implements Checkbox {
  render(): string {
    return '<input type="checkbox" class="mac">';
  }
}

class WindowsFactory implements GUIFactory {
  createButton(): Button {
    return new WindowsButton();
  }
  createCheckbox(): Checkbox {
    return new WindowsCheckbox();
  }
}

class MacFactory implements GUIFactory {
  createButton(): Button {
    return new MacButton();
  }
  createCheckbox(): Checkbox {
    return new MacCheckbox();
  }
}

// ============================================================
// 3. 代理模式（Proxy）
// ============================================================

/**
 * 📊 代理模式
 *
 * 为对象提供代理以控制访问
 *
 * 场景：
 * - 虚拟代理（懒加载）
 * - 缓存代理
 * - 保护代理（权限控制）
 */

// 虚拟代理（图片懒加载）
class ImageProxy {
  private realImage: HTMLImageElement | null = null;
  private placeholder: HTMLImageElement;

  constructor(private src: string) {
    this.placeholder = new Image();
    this.placeholder.src = 'loading.gif';
  }

  getImage(): HTMLImageElement {
    return this.realImage || this.placeholder;
  }

  load(): void {
    if (!this.realImage) {
      this.realImage = new Image();
      this.realImage.onload = () => {
        console.log('Image loaded');
      };
      this.realImage.src = this.src;
    }
  }
}

// 缓存代理
function createCacheProxy<T extends (...args: any[]) => any>(fn: T): T {
  const cache = new Map<string, ReturnType<T>>();

  return function(...args: Parameters<T>): ReturnType<T> {
    const key = JSON.stringify(args);

    if (cache.has(key)) {
      console.log('Cache hit:', key);
      return cache.get(key)!;
    }

    const result = fn(...args);
    cache.set(key, result);
    return result;
  } as T;
}

// 使用 ES6 Proxy
function createReactiveProxy<T extends object>(
  target: T,
  onChange: (prop: string | symbol, value: any) => void
): T {
  return new Proxy(target, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      // 深层代理
      if (typeof value === 'object' && value !== null) {
        return createReactiveProxy(value, onChange);
      }
      return value;
    },
    set(target, prop, value, receiver) {
      const result = Reflect.set(target, prop, value, receiver);
      onChange(prop, value);
      return result;
    },
  });
}

// ============================================================
// 4. 策略模式（Strategy）
// ============================================================

/**
 * 📊 策略模式
 *
 * 定义一系列算法，使它们可以互换
 *
 * 场景：
 * - 表单验证
 * - 支付方式
 * - 排序算法
 */

// 表单验证策略
interface ValidationStrategy {
  validate(value: string): boolean;
  message: string;
}

const strategies: Record<string, ValidationStrategy> = {
  required: {
    validate: (value) => value.trim().length > 0,
    message: '此字段为必填项',
  },
  email: {
    validate: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
    message: '请输入有效的邮箱地址',
  },
  minLength: {
    validate: (value) => value.length >= 6,
    message: '最少需要 6 个字符',
  },
  phone: {
    validate: (value) => /^1[3-9]\d{9}$/.test(value),
    message: '请输入有效的手机号',
  },
};

class Validator {
  private rules: Array<{ field: string; strategy: string }> = [];

  addRule(field: string, strategy: string): this {
    this.rules.push({ field, strategy });
    return this;
  }

  validate(data: Record<string, string>): { valid: boolean; errors: Record<string, string> } {
    const errors: Record<string, string> = {};

    for (const rule of this.rules) {
      const value = data[rule.field] || '';
      const strategy = strategies[rule.strategy];

      if (strategy && !strategy.validate(value)) {
        errors[rule.field] = strategy.message;
      }
    }

    return {
      valid: Object.keys(errors).length === 0,
      errors,
    };
  }
}

// 使用示例
const validator = new Validator()
  .addRule('username', 'required')
  .addRule('email', 'email')
  .addRule('password', 'minLength');

// ============================================================
// 5. 装饰器模式（Decorator）
// ============================================================

/**
 * 📊 装饰器模式
 *
 * 动态添加功能而不改变原有结构
 *
 * 场景：
 * - 日志记录
 * - 性能监控
 * - 权限检查
 */

// 函数装饰器
function logDecorator<T extends (...args: any[]) => any>(fn: T): T {
  return function(...args: Parameters<T>): ReturnType<T> {
    console.log(`Calling ${fn.name} with:`, args);
    const result = fn(...args);
    console.log(`Result:`, result);
    return result;
  } as T;
}

function measureTime<T extends (...args: any[]) => any>(fn: T): T {
  return function(...args: Parameters<T>): ReturnType<T> {
    const start = performance.now();
    const result = fn(...args);
    const end = performance.now();
    console.log(`${fn.name} took ${end - start}ms`);
    return result;
  } as T;
}

// TypeScript 装饰器
function Log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const original = descriptor.value;

  descriptor.value = function(...args: any[]) {
    console.log(`[${propertyKey}] called with:`, args);
    const result = original.apply(this, args);
    console.log(`[${propertyKey}] returned:`, result);
    return result;
  };

  return descriptor;
}

function Debounce(delay: number) {
  return function(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    let timeoutId: ReturnType<typeof setTimeout>;

    descriptor.value = function(...args: any[]) {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        original.apply(this, args);
      }, delay);
    };

    return descriptor;
  };
}

// 类装饰器示例
class ExampleClass {
  // @Log
  add(a: number, b: number): number {
    return a + b;
  }

  // @Debounce(300)
  search(query: string): void {
    console.log('Searching:', query);
  }
}

// ============================================================
// 6. 其他常用模式
// ============================================================

/**
 * 📊 适配器模式
 */

interface OldAPI {
  request(url: string, callback: (data: any) => void): void;
}

interface NewAPI {
  fetch(url: string): Promise<any>;
}

class APIAdapter implements NewAPI {
  constructor(private oldAPI: OldAPI) {}

  fetch(url: string): Promise<any> {
    return new Promise((resolve) => {
      this.oldAPI.request(url, (data) => {
        resolve(data);
      });
    });
  }
}

/**
 * 📊 命令模式
 */

interface Command {
  execute(): void;
  undo(): void;
}

class TextEditor {
  private content = '';

  getContent(): string {
    return this.content;
  }

  insert(text: string, position: number): void {
    this.content = this.content.slice(0, position) + text + this.content.slice(position);
  }

  delete(position: number, length: number): string {
    const deleted = this.content.slice(position, position + length);
    this.content = this.content.slice(0, position) + this.content.slice(position + length);
    return deleted;
  }
}

class InsertCommand implements Command {
  private deletedText = '';

  constructor(
    private editor: TextEditor,
    private text: string,
    private position: number
  ) {}

  execute(): void {
    this.editor.insert(this.text, this.position);
  }

  undo(): void {
    this.editor.delete(this.position, this.text.length);
  }
}

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 单例模式的线程安全
 *    - JS 单线程，但要注意异步
 *
 * 2. 过度使用设计模式
 *    - 简单问题不需要复杂模式
 *    - KISS 原则
 *
 * 3. 策略模式的策略选择
 *    - 策略太多时考虑配置化
 *
 * 4. 装饰器的执行顺序
 *    - 从下到上装饰
 *    - 从上到下执行
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 单例模式的应用场景？
 * A:
 *    - Vuex/Redux 的 store
 *    - 日志记录器
 *    - 配置管理
 *    - 数据库连接池
 *
 * Q2: 工厂模式和构造函数的区别？
 * A:
 *    - 工厂可以返回不同类型
 *    - 工厂可以使用缓存
 *    - 工厂更灵活
 *
 * Q3: 代理模式和装饰器模式的区别？
 * A:
 *    - 代理模式：控制访问
 *    - 装饰器模式：增强功能
 *    - 代理通常不改变接口
 *
 * Q4: 前端常用哪些设计模式？
 * A:
 *    - 单例：全局状态
 *    - 观察者/发布订阅：事件处理
 *    - 策略：表单验证
 *    - 代理：Vue 响应式
 *    - 装饰器：HOC、注解
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：请求缓存
 */

const requestCacheExample = `
const cachedFetch = createCacheProxy(async (url: string) => {
  const response = await fetch(url);
  return response.json();
});

// 相同请求会使用缓存
await cachedFetch('/api/user/1');
await cachedFetch('/api/user/1'); // Cache hit
`;

/**
 * 🏢 场景 2：表单验证
 */

const formValidationExample = `
const validator = new Validator()
  .addRule('username', 'required')
  .addRule('username', 'minLength')
  .addRule('email', 'email')
  .addRule('phone', 'phone');

const result = validator.validate({
  username: 'tom',
  email: 'tom@example.com',
  phone: '13800138000',
});

if (!result.valid) {
  console.log('Errors:', result.errors);
}
`;

/**
 * 🏢 场景 3：撤销/重做
 */

const undoRedoCommandExample = `
class CommandManager {
  private history: Command[] = [];
  private current = -1;

  execute(command: Command) {
    // 清除 redo 历史
    this.history = this.history.slice(0, this.current + 1);
    command.execute();
    this.history.push(command);
    this.current++;
  }

  undo() {
    if (this.current >= 0) {
      this.history[this.current].undo();
      this.current--;
    }
  }

  redo() {
    if (this.current < this.history.length - 1) {
      this.current++;
      this.history[this.current].execute();
    }
  }
}
`;

export {
  Singleton,
  createSingleton,
  ButtonFactory,
  WindowsFactory,
  MacFactory,
  ImageProxy,
  createCacheProxy,
  createReactiveProxy,
  strategies,
  Validator,
  logDecorator,
  measureTime,
  Log,
  Debounce,
  APIAdapter,
  TextEditor,
  InsertCommand,
  requestCacheExample,
  formValidationExample,
  undoRedoCommandExample,
};

