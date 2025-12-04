/**
 * ============================================================
 * 📚 运行时性能优化
 * ============================================================
 *
 * 面试考察重点：
 * 1. JavaScript 执行优化
 * 2. 内存优化
 * 3. 防抖节流
 * 4. Web Worker 使用
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 运行时性能优化的目标
 *
 * 目标：保持主线程响应性，避免长任务阻塞
 *
 * 长任务定义：执行时间 > 50ms 的任务
 * 影响：用户交互无响应，页面卡顿
 *
 * 优化策略：
 * 1. 减少计算量
 * 2. 分片执行
 * 3. 移到 Worker
 * 4. 优化数据结构
 */

// ============================================================
// 2. 防抖与节流
// ============================================================

/**
 * 📊 防抖（Debounce）
 *
 * 【定义】事件触发后延迟执行，期间再次触发则重新计时
 * 【场景】搜索框输入、窗口 resize、表单验证
 * 【效果】只执行最后一次
 *
 * 时间线：
 * 触发: ─●─●─●────────────────────────
 * 执行: ─────────────────────────●────
 *                              延迟后执行
 */

// 防抖实现（完整版）
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  options: {
    leading?: boolean;  // 是否在开始时执行
    trailing?: boolean; // 是否在结束时执行
    maxWait?: number;   // 最大等待时间
  } = {}
): T & { cancel: () => void; flush: () => void } {
  const { leading = false, trailing = true, maxWait } = options;

  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastArgs: any[] | null = null;
  let lastThis: any = null;
  let lastCallTime: number | undefined;
  let lastInvokeTime = 0;
  let result: any;

  function invokeFunc(time: number) {
    const args = lastArgs;
    const thisArg = lastThis;
    lastArgs = lastThis = null;
    lastInvokeTime = time;
    result = func.apply(thisArg, args!);
    return result;
  }

  function shouldInvoke(time: number) {
    const timeSinceLastCall = lastCallTime === undefined ? 0 : time - lastCallTime;
    const timeSinceLastInvoke = time - lastInvokeTime;

    return (
      lastCallTime === undefined ||
      timeSinceLastCall >= wait ||
      timeSinceLastCall < 0 ||
      (maxWait !== undefined && timeSinceLastInvoke >= maxWait)
    );
  }

  function leadingEdge(time: number) {
    lastInvokeTime = time;
    timeoutId = setTimeout(timerExpired, wait);
    return leading ? invokeFunc(time) : result;
  }

  function trailingEdge(time: number) {
    timeoutId = null;
    if (trailing && lastArgs) {
      return invokeFunc(time);
    }
    lastArgs = lastThis = null;
    return result;
  }

  function timerExpired() {
    const time = Date.now();
    if (shouldInvoke(time)) {
      return trailingEdge(time);
    }
    const timeSinceLastCall = time - (lastCallTime || 0);
    const timeSinceLastInvoke = time - lastInvokeTime;
    const timeWaiting = wait - timeSinceLastCall;
    const remainingWait = maxWait !== undefined
      ? Math.min(timeWaiting, maxWait - timeSinceLastInvoke)
      : timeWaiting;

    timeoutId = setTimeout(timerExpired, remainingWait);
  }

  function debounced(this: any, ...args: any[]) {
    const time = Date.now();
    const isInvoking = shouldInvoke(time);

    lastArgs = args;
    lastThis = this;
    lastCallTime = time;

    if (isInvoking) {
      if (timeoutId === null) {
        return leadingEdge(time);
      }
      if (maxWait !== undefined) {
        timeoutId = setTimeout(timerExpired, wait);
        return invokeFunc(time);
      }
    }
    if (timeoutId === null) {
      timeoutId = setTimeout(timerExpired, wait);
    }
    return result;
  }

  debounced.cancel = function() {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
    lastInvokeTime = 0;
    lastArgs = lastCallTime = lastThis = timeoutId = null;
  };

  debounced.flush = function() {
    if (timeoutId !== null) {
      return trailingEdge(Date.now());
    }
    return result;
  };

  return debounced as T & { cancel: () => void; flush: () => void };
}

/**
 * 📊 节流（Throttle）
 *
 * 【定义】固定时间间隔内只执行一次
 * 【场景】滚动事件、mousemove、拖拽
 * 【效果】固定频率执行
 *
 * 时间线：
 * 触发: ─●─●─●─●─●─●─●─●─●─●─●─
 * 执行: ─●───────●───────●─────
 *       固定间隔执行
 */

// 节流实现（完整版）
function throttle<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  options: {
    leading?: boolean;
    trailing?: boolean;
  } = {}
): T & { cancel: () => void } {
  const { leading = true, trailing = true } = options;

  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastArgs: any[] | null = null;
  let lastThis: any = null;
  let lastTime = 0;

  function invokeFunc() {
    const args = lastArgs;
    const thisArg = lastThis;
    lastArgs = lastThis = null;
    lastTime = Date.now();
    func.apply(thisArg, args!);
  }

  function throttled(this: any, ...args: any[]) {
    const now = Date.now();

    // 第一次调用且 leading 为 false
    if (!lastTime && !leading) {
      lastTime = now;
    }

    const remaining = wait - (now - lastTime);
    lastArgs = args;
    lastThis = this;

    if (remaining <= 0 || remaining > wait) {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      invokeFunc();
    } else if (!timeoutId && trailing) {
      timeoutId = setTimeout(() => {
        timeoutId = null;
        lastTime = leading ? Date.now() : 0;
        invokeFunc();
      }, remaining);
    }
  }

  throttled.cancel = function() {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    lastTime = 0;
    timeoutId = lastArgs = lastThis = null;
  };

  return throttled as T & { cancel: () => void };
}

/**
 * 💡 面试追问：防抖和节流如何选择？
 *
 * 防抖：
 * - 只关心最终结果
 * - 搜索框输入、窗口 resize
 *
 * 节流：
 * - 需要固定频率响应
 * - 滚动加载、拖拽、游戏循环
 *
 * ⚠️ 注意：
 * - 防抖有最大等待时间（maxWait）可以兼具两者特点
 * - lodash 的 throttle 实际是带 maxWait 的 debounce
 */

// ============================================================
// 3. 任务分片
// ============================================================

/**
 * 📊 长任务分片执行
 *
 * 问题：长任务阻塞主线程
 * 解决：将任务分成小块，每块执行后让出主线程
 */

// 使用 requestIdleCallback 分片
function processLargeArray<T>(
  items: T[],
  process: (item: T) => void,
  onComplete?: () => void
) {
  const queue = [...items];

  function processChunk(deadline: IdleDeadline) {
    // 在空闲时间内处理尽可能多的任务
    while (queue.length > 0 && deadline.timeRemaining() > 0) {
      const item = queue.shift()!;
      process(item);
    }

    if (queue.length > 0) {
      // 还有任务，继续调度
      requestIdleCallback(processChunk);
    } else {
      // 完成
      onComplete?.();
    }
  }

  requestIdleCallback(processChunk);
}

// 使用 scheduler.yield()（实验性 API）
async function processWithYield<T>(
  items: T[],
  process: (item: T) => void,
  chunkSize = 100
) {
  for (let i = 0; i < items.length; i += chunkSize) {
    // 处理一批
    const chunk = items.slice(i, i + chunkSize);
    chunk.forEach(process);

    // 让出主线程
    // @ts-ignore
    if (typeof scheduler !== 'undefined' && scheduler.yield) {
      // @ts-ignore
      await scheduler.yield();
    } else {
      // 降级方案
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
}

// 时间切片实现（React 类似思路）
function timeSlicing<T>(
  items: T[],
  process: (item: T) => void,
  options: {
    yieldInterval?: number; // 让出间隔（ms）
    onProgress?: (processed: number, total: number) => void;
    onComplete?: () => void;
  } = {}
) {
  const { yieldInterval = 5, onProgress, onComplete } = options;
  const queue = [...items];
  const total = items.length;
  let processed = 0;

  function processChunk() {
    const start = performance.now();

    while (queue.length > 0) {
      // 检查是否需要让出
      if (performance.now() - start >= yieldInterval) {
        // 使用 MessageChannel 创建宏任务
        const channel = new MessageChannel();
        channel.port1.onmessage = processChunk;
        channel.port2.postMessage(null);
        return;
      }

      const item = queue.shift()!;
      process(item);
      processed++;
      onProgress?.(processed, total);
    }

    onComplete?.();
  }

  processChunk();
}

// ============================================================
// 4. Web Worker
// ============================================================

/**
 * 📊 Web Worker 使用场景
 *
 * 1. CPU 密集计算
 *    - 数据处理、图像处理
 *    - 加密/解密
 *    - 复杂算法
 *
 * 2. 大数据处理
 *    - 大文件解析
 *    - 大数组排序/过滤
 *
 * 3. 后台任务
 *    - 数据同步
 *    - 预计算
 */

// Worker 封装类
class TaskWorker {
  private worker: Worker;
  private taskId = 0;
  private pending = new Map<number, {
    resolve: (value: any) => void;
    reject: (reason: any) => void;
  }>();

  constructor(workerScript: string) {
    this.worker = new Worker(workerScript);

    this.worker.onmessage = (e) => {
      const { id, result, error } = e.data;
      const task = this.pending.get(id);

      if (task) {
        if (error) {
          task.reject(new Error(error));
        } else {
          task.resolve(result);
        }
        this.pending.delete(id);
      }
    };
  }

  run<T>(type: string, data: any): Promise<T> {
    return new Promise((resolve, reject) => {
      const id = this.taskId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, data });
    });
  }

  terminate() {
    this.worker.terminate();
  }
}

// Worker 脚本示例
const workerScript = `
// worker.js
self.onmessage = function(e) {
  const { id, type, data } = e.data;

  try {
    let result;

    switch (type) {
      case 'sort':
        result = data.slice().sort((a, b) => a - b);
        break;
      case 'filter':
        result = data.filter(item => item > 0);
        break;
      case 'compute':
        // 复杂计算
        result = heavyComputation(data);
        break;
      default:
        throw new Error('Unknown task type');
    }

    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};

function heavyComputation(data) {
  // 模拟复杂计算
  let result = 0;
  for (let i = 0; i < data.length; i++) {
    result += Math.sqrt(data[i]) * Math.sin(data[i]);
  }
  return result;
}
`;

// 使用 Comlink 简化 Worker 通信
const comlinkExample = `
// worker.js
import * as Comlink from 'comlink';

const api = {
  async heavyTask(data) {
    // 复杂计算
    return result;
  }
};

Comlink.expose(api);

// main.js
import * as Comlink from 'comlink';

const worker = new Worker('./worker.js');
const api = Comlink.wrap(worker);

// 像调用本地函数一样使用
const result = await api.heavyTask(data);
`;

// ============================================================
// 5. 内存优化
// ============================================================

/**
 * 📊 内存优化策略
 *
 * 1. 避免内存泄漏
 * 2. 及时释放引用
 * 3. 使用对象池
 * 4. 优化数据结构
 */

// 对象池模式
class ObjectPool<T> {
  private pool: T[] = [];
  private factory: () => T;
  private reset: (obj: T) => void;
  private maxSize: number;

  constructor(
    factory: () => T,
    reset: (obj: T) => void,
    maxSize = 100
  ) {
    this.factory = factory;
    this.reset = reset;
    this.maxSize = maxSize;
  }

  acquire(): T {
    if (this.pool.length > 0) {
      return this.pool.pop()!;
    }
    return this.factory();
  }

  release(obj: T) {
    if (this.pool.length < this.maxSize) {
      this.reset(obj);
      this.pool.push(obj);
    }
  }

  clear() {
    this.pool = [];
  }
}

// 使用示例：粒子系统
interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
}

const particlePool = new ObjectPool<Particle>(
  // factory
  () => ({ x: 0, y: 0, vx: 0, vy: 0, life: 0 }),
  // reset
  (p) => { p.x = 0; p.y = 0; p.vx = 0; p.vy = 0; p.life = 0; }
);

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 防抖节流参数设置不当
 *    - 时间太长：响应迟钝
 *    - 时间太短：效果不明显
 *    - 建议：150-300ms
 *
 * 2. Web Worker 滥用
 *    - 通信开销可能抵消收益
 *    - 小任务不值得用 Worker
 *    - 建议：> 50ms 的任务再考虑
 *
 * 3. requestIdleCallback 不可靠
 *    - 可能长时间不被调用
 *    - 需要设置 timeout
 *
 * 4. 忘记清理定时器
 *    - 组件卸载时未清理
 *    - 导致内存泄漏和逻辑错误
 *
 * 5. 闭包持有大对象
 *    - 事件处理器引用大数据
 *    - 应该及时解除引用
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何检测长任务？
 * A:
 * - PerformanceObserver 监听 longtask
 * - Chrome DevTools Performance 面板
 * - Long Tasks API
 *
 * Q2: requestIdleCallback 和 setTimeout 的区别？
 * A:
 * - requestIdleCallback 在浏览器空闲时执行
 * - setTimeout 是固定延迟
 * - requestIdleCallback 适合低优先级任务
 * - 兼容性：Safari 不支持
 *
 * Q3: Web Worker 有什么限制？
 * A:
 * - 无法访问 DOM
 * - 无法访问 window、document
 * - 数据通过消息传递（结构化克隆）
 * - 同源限制
 *
 * Q4: 如何优化大列表过滤？
 * A:
 * 1. 防抖输入
 * 2. Web Worker 处理
 * 3. 虚拟滚动
 * 4. 增量搜索
 * 5. 索引预处理
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：搜索框优化
 *
 * 问题：输入时 API 请求过多
 *
 * 解决：
 * 1. 防抖 300ms
 * 2. 取消之前的请求
 * 3. 本地缓存结果
 */
const searchWithDebounce = `
const debouncedSearch = debounce(async (query) => {
  // 取消之前的请求
  controller?.abort();
  controller = new AbortController();

  try {
    const result = await fetch('/api/search?q=' + query, {
      signal: controller.signal
    });
    setResults(await result.json());
  } catch (e) {
    if (e.name !== 'AbortError') throw e;
  }
}, 300);
`;

/**
 * 🏢 场景 2：大数据处理
 *
 * 问题：处理 10 万条数据卡顿
 *
 * 解决：
 * 1. Web Worker 处理
 * 2. 分页加载
 * 3. 虚拟滚动显示
 */

/**
 * 🏢 场景 3：实时计算
 *
 * 问题：频繁计算导致卡顿
 *
 * 解决：
 * 1. 节流计算频率
 * 2. 缓存计算结果
 * 3. 异步计算
 */

export {
  debounce,
  throttle,
  processLargeArray,
  processWithYield,
  timeSlicing,
  TaskWorker,
  workerScript,
  comlinkExample,
  ObjectPool,
  particlePool,
  searchWithDebounce,
};

