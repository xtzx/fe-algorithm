/**
 * ============================================================
 * 📚 Phase 6: 调度机制
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 Scheduler 的作用
 * 2. 掌握优先级机制
 * 3. 理解时间切片
 * 4. 理解 Lane 模型
 *
 * 📁 源码位置：
 * - packages/scheduler/src/forks/Scheduler.js
 * - packages/react-reconciler/src/ReactFiberLane.js
 *
 * ⏱️ 预计时间：4 小时
 * 🔥 面试权重：⭐⭐⭐
 */

// ============================================================
// 1. Scheduler 概述
// ============================================================

/**
 * 📊 Scheduler 的作用
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        Scheduler                                │
 * │                                                                 │
 * │   ┌─────────────────────────────────────────────────────────┐  │
 * │   │                    任务队列                              │  │
 * │   │                                                         │  │
 * │   │   高优先级 ─────►  中优先级 ─────►  低优先级              │  │
 * │   │   (立即执行)       (5ms)           (10s)                │  │
 * │   └─────────────────────────────────────────────────────────┘  │
 * │                          │                                      │
 * │                          ▼                                      │
 * │   ┌─────────────────────────────────────────────────────────┐  │
 * │   │                   时间切片                               │  │
 * │   │                                                         │  │
 * │   │   每帧只执行一部分任务，避免阻塞主线程                    │  │
 * │   │   默认 5ms 一个时间片                                   │  │
 * │   └─────────────────────────────────────────────────────────┘  │
 * │                          │                                      │
 * │                          ▼                                      │
 * │   ┌─────────────────────────────────────────────────────────┐  │
 * │   │                   任务中断与恢复                          │  │
 * │   │                                                         │  │
 * │   │   高优先级任务可以打断低优先级任务                        │  │
 * │   │   低优先级任务可以从中断点恢复                            │  │
 * │   └─────────────────────────────────────────────────────────┘  │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 * ```
 */

// ============================================================
// 2. 优先级
// ============================================================

/**
 * 📊 Scheduler 优先级
 *
 * 源码位置：packages/scheduler/src/SchedulerPriorities.js
 */

const PriorityLevels = {
  ImmediatePriority: 1,    // 立即执行（同步）
  UserBlockingPriority: 2, // 用户交互（250ms）
  NormalPriority: 3,       // 普通（5s）
  LowPriority: 4,          // 低优先级（10s）
  IdlePriority: 5,         // 空闲（永不过期）
};

// 不同优先级的过期时间
const IMMEDIATE_PRIORITY_TIMEOUT = -1;    // 立即过期
const USER_BLOCKING_PRIORITY_TIMEOUT = 250;
const NORMAL_PRIORITY_TIMEOUT = 5000;
const LOW_PRIORITY_TIMEOUT = 10000;
const IDLE_PRIORITY_TIMEOUT = 1073741823; // 最大 32 位整数

// ============================================================
// 3. 任务调度
// ============================================================

/**
 * 📊 任务数据结构
 */

interface Task {
  id: number;
  callback: ((didTimeout: boolean) => any) | null;
  priorityLevel: number;
  startTime: number;
  expirationTime: number;
  sortIndex: number;
}

// 任务队列（小顶堆）
let taskQueue: Task[] = [];       // 已过期任务
let timerQueue: Task[] = [];      // 未过期任务
let taskIdCounter = 0;
let currentTask: Task | null = null;

// 简化版 scheduleCallback
function scheduleCallback(
  priorityLevel: number,
  callback: (didTimeout: boolean) => any,
  options?: { delay?: number }
): Task {
  const currentTime = performance.now();

  // 计算开始时间
  let startTime = currentTime;
  if (options && options.delay && options.delay > 0) {
    startTime = currentTime + options.delay;
  }

  // 计算过期时间
  let timeout: number;
  switch (priorityLevel) {
    case PriorityLevels.ImmediatePriority:
      timeout = IMMEDIATE_PRIORITY_TIMEOUT;
      break;
    case PriorityLevels.UserBlockingPriority:
      timeout = USER_BLOCKING_PRIORITY_TIMEOUT;
      break;
    case PriorityLevels.IdlePriority:
      timeout = IDLE_PRIORITY_TIMEOUT;
      break;
    case PriorityLevels.LowPriority:
      timeout = LOW_PRIORITY_TIMEOUT;
      break;
    default:
      timeout = NORMAL_PRIORITY_TIMEOUT;
      break;
  }

  const expirationTime = startTime + timeout;

  // 创建任务
  const newTask: Task = {
    id: taskIdCounter++,
    callback,
    priorityLevel,
    startTime,
    expirationTime,
    sortIndex: -1,
  };

  if (startTime > currentTime) {
    // 延迟任务，加入 timerQueue
    newTask.sortIndex = startTime;
    push(timerQueue, newTask);
    // 设置定时器
  } else {
    // 立即任务，加入 taskQueue
    newTask.sortIndex = expirationTime;
    push(taskQueue, newTask);
    // 请求调度
    requestHostCallback(flushWork);
  }

  return newTask;
}

// ============================================================
// 4. 时间切片
// ============================================================

/**
 * 📊 时间切片原理
 *
 * ```
 * 一帧时间（约 16.6ms）
 * ┌────────────────────────────────────────────────────────────────┐
 * │                                                                │
 * │  JS 执行  │  样式计算  │  布局  │  绘制  │  空闲               │
 * │  (5ms)   │           │       │       │                       │
 * │          │           │       │       │                       │
 * └──────────┴───────────┴───────┴───────┴───────────────────────┘
 *
 * Scheduler 默认每个时间片 5ms
 * 执行 5ms 后检查是否需要让出
 * ```
 */

// 时间片长度
const frameYieldMs = 5;
let frameDeadline = 0;

// 是否应该让出
function shouldYieldToHost(): boolean {
  const currentTime = performance.now();
  return currentTime >= frameDeadline;
}

// 请求调度（使用 MessageChannel）
let scheduledHostCallback: ((hasTimeRemaining: boolean, currentTime: number) => boolean) | null = null;
const channel = typeof MessageChannel !== 'undefined' ? new MessageChannel() : null;

function requestHostCallback(
  callback: (hasTimeRemaining: boolean, currentTime: number) => boolean
) {
  scheduledHostCallback = callback;
  if (channel) {
    channel.port1.postMessage(null);
  }
}

// MessageChannel 回调
if (channel) {
  channel.port2.onmessage = () => {
    if (scheduledHostCallback !== null) {
      const currentTime = performance.now();
      // 设置 deadline
      frameDeadline = currentTime + frameYieldMs;
      // 执行任务
      const hasMoreWork = scheduledHostCallback(true, currentTime);
      if (hasMoreWork) {
        // 还有任务，继续调度
        channel.port1.postMessage(null);
      } else {
        scheduledHostCallback = null;
      }
    }
  };
}

// 工作循环
function flushWork(
  hasTimeRemaining: boolean,
  initialTime: number
): boolean {
  let currentTime = initialTime;

  // 执行任务
  currentTask = peek(taskQueue);

  while (currentTask !== null) {
    // 检查是否需要让出
    if (currentTask.expirationTime > currentTime && shouldYieldToHost()) {
      break;
    }

    const callback = currentTask.callback;
    if (callback !== null) {
      currentTask.callback = null;
      const didTimeout = currentTask.expirationTime <= currentTime;
      // 执行任务
      const continuationCallback = callback(didTimeout);

      if (typeof continuationCallback === 'function') {
        // 任务没完成，更新 callback
        currentTask.callback = continuationCallback;
      } else {
        // 任务完成，移除
        pop(taskQueue);
      }
    } else {
      pop(taskQueue);
    }

    currentTask = peek(taskQueue);
  }

  // 返回是否还有任务
  return currentTask !== null;
}

// ============================================================
// 5. Lane 模型
// ============================================================

/**
 * 📊 Lane 模型
 *
 * React 18 使用 Lane 模型管理优先级
 * Lane 是一个 31 位的二进制数，每一位代表一个优先级
 *
 * ```
 * SyncLane             = 0b0000000000000000000000000000001;
 * InputContinuousLane  = 0b0000000000000000000000000000100;
 * DefaultLane          = 0b0000000000000000000000000010000;
 * TransitionLane       = 0b0000000000000000000000001000000;
 * IdleLane             = 0b0100000000000000000000000000000;
 * ```
 *
 * 优势：
 * - 可以用位运算快速合并/判断优先级
 * - 支持批量处理同一优先级的更新
 */

const Lanes = {
  NoLane: 0b0000000000000000000000000000000,
  SyncLane: 0b0000000000000000000000000000001,
  InputContinuousLane: 0b0000000000000000000000000000100,
  DefaultLane: 0b0000000000000000000000000010000,
  TransitionLane1: 0b0000000000000000000000001000000,
  IdleLane: 0b0100000000000000000000000000000,
};

// 合并 Lane
function mergeLanes(a: number, b: number): number {
  return a | b;
}

// 判断是否包含 Lane
function includesSomeLane(set: number, subset: number): boolean {
  return (set & subset) !== 0;
}

// ============================================================
// 6. 💡 面试题
// ============================================================

/**
 * 💡 Q1: React 的调度机制是什么？
 *
 * A: React 使用 Scheduler 进行任务调度：
 *    1. 优先级调度：不同任务有不同优先级
 *    2. 时间切片：每帧只执行一部分任务
 *    3. 可中断：高优先级任务可以打断低优先级
 *
 * 💡 Q2: 什么是时间切片？
 *
 * A: 将长任务切分成多个小任务，每帧执行一个时间片（5ms），
 *    然后检查是否有更高优先级的任务，避免阻塞主线程。
 *
 * 💡 Q3: React 使用什么 API 实现调度？
 *
 * A: 主要使用 MessageChannel：
 *    - 不使用 setTimeout（最小 4ms 延迟）
 *    - 不使用 requestAnimationFrame（与帧率绑定）
 *    - MessageChannel 是宏任务，能被 JS 引擎优化
 *
 * 💡 Q4: 什么是 Lane 模型？
 *
 * A: Lane 是 React 18 的优先级模型：
 *    - 使用 31 位二进制数表示优先级
 *    - 可以用位运算快速处理
 *    - 支持批量更新
 */

// ============================================================
// 7. 辅助函数（小顶堆）
// ============================================================

function push(heap: Task[], node: Task) {
  const index = heap.length;
  heap.push(node);
  siftUp(heap, node, index);
}

function peek(heap: Task[]): Task | null {
  return heap.length === 0 ? null : heap[0];
}

function pop(heap: Task[]): Task | null {
  if (heap.length === 0) return null;
  const first = heap[0];
  const last = heap.pop()!;
  if (last !== first) {
    heap[0] = last;
    siftDown(heap, last, 0);
  }
  return first;
}

function siftUp(heap: Task[], node: Task, i: number) {
  let index = i;
  while (index > 0) {
    const parentIndex = (index - 1) >>> 1;
    const parent = heap[parentIndex];
    if (compare(parent, node) > 0) {
      heap[parentIndex] = node;
      heap[index] = parent;
      index = parentIndex;
    } else {
      return;
    }
  }
}

function siftDown(heap: Task[], node: Task, i: number) {
  let index = i;
  const length = heap.length;
  const halfLength = length >>> 1;
  while (index < halfLength) {
    const leftIndex = (index + 1) * 2 - 1;
    const left = heap[leftIndex];
    const rightIndex = leftIndex + 1;
    const right = heap[rightIndex];

    if (compare(left, node) < 0) {
      if (rightIndex < length && compare(right, left) < 0) {
        heap[index] = right;
        heap[rightIndex] = node;
        index = rightIndex;
      } else {
        heap[index] = left;
        heap[leftIndex] = node;
        index = leftIndex;
      }
    } else if (rightIndex < length && compare(right, node) < 0) {
      heap[index] = right;
      heap[rightIndex] = node;
      index = rightIndex;
    } else {
      return;
    }
  }
}

function compare(a: Task, b: Task): number {
  const diff = a.sortIndex - b.sortIndex;
  return diff !== 0 ? diff : a.id - b.id;
}

// ============================================================
// 8. 📖 源码阅读指南
// ============================================================

/**
 * 📖 阅读顺序：
 *
 * 1. packages/scheduler/src/forks/Scheduler.js
 *    - unstable_scheduleCallback（调度入口）
 *    - workLoop（工作循环）
 *    - shouldYieldToHost（让出判断）
 *
 * 2. packages/scheduler/src/SchedulerMinHeap.js
 *    - 小顶堆实现
 *
 * 3. packages/react-reconciler/src/ReactFiberLane.js
 *    - Lane 定义和操作
 */

// ============================================================
// 9. ✅ 学习检查
// ============================================================

/**
 * ✅ 完成以下任务：
 *
 * - [ ] 理解 Scheduler 的作用
 * - [ ] 理解优先级机制
 * - [ ] 理解时间切片原理
 * - [ ] 理解 Lane 模型
 * - [ ] 阅读源码：Scheduler.js
 */

export {
  PriorityLevels,
  Lanes,
  scheduleCallback,
  shouldYieldToHost,
  mergeLanes,
  includesSomeLane,
};

