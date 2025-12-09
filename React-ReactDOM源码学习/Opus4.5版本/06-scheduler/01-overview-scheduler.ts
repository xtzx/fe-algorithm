/**
 * ============================================================
 * 📚 Phase 6: Scheduler 调度机制 - Part 1: 概览
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/scheduler/src/forks/Scheduler.js      （调度器核心）
 * - packages/scheduler/src/SchedulerMinHeap.js     （最小堆实现）
 * - packages/scheduler/src/SchedulerPriorities.js  （优先级定义）
 * - packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 *
 * ⏱️ 预计时间：3-4 小时
 * 🎯 面试权重：⭐⭐⭐⭐
 */

// ============================================================
// Part 1: 为什么需要 Scheduler？
// ============================================================

/**
 * 📊 问题背景：JavaScript 单线程的困境
 *
 * 浏览器的主线程负责：
 * 1. JavaScript 执行
 * 2. 样式计算
 * 3. 布局（Layout）
 * 4. 绘制（Paint）
 * 5. 用户交互响应
 *
 * 如果 React 渲染阻塞主线程太久会怎样？
 */

const problemWithoutScheduler = `
📊 没有 Scheduler 的问题

假设有一个很大的组件树需要渲染（10000 个节点）：

┌──────────────────────────────────────────────────────────────────────┐
│ 主线程时间轴                                                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ├─────────────────────────────────────┤                              │
│ │     React 渲染 (300ms)              │                              │
│ └─────────────────────────────────────┘                              │
│                                        ↑                             │
│                                        │                             │
│                               用户点击事件（被阻塞！）                 │
│                               输入响应延迟 > 100ms                    │
│                               用户感知「卡顿」                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

问题：
- 长任务阻塞主线程
- 用户交互无响应
- 掉帧（低于 60fps）
- 体验差
`;

/**
 * 📊 Scheduler 的解决方案：时间切片（Time Slicing）
 */

const schedulerSolution = `
📊 Scheduler 的解决方案

核心思想：把长任务拆成多个小任务，每个小任务执行完后检查是否需要让出主线程

┌──────────────────────────────────────────────────────────────────────┐
│ 主线程时间轴（使用 Scheduler 后）                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤               │
│ │ 渲染 │ │ 渲染 │ │ 渲染 │ │ 渲染 │ │ 渲染 │ │ 渲染 │               │
│ │ 5ms  │ │ 5ms  │ │ 5ms  │ │ 5ms  │ │ 5ms  │ │ 5ms  │               │
│ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘               │
│          ↑       ↑                                                   │
│          │       │                                                   │
│          │       └── 浏览器绘制                                       │
│          │                                                           │
│          └── 用户点击（立即响应！）                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Scheduler 做了什么：
1. 把渲染工作拆成小块（每块 ~5ms）
2. 每块执行完检查 shouldYieldToHost()
3. 如果有更高优先级任务（用户输入），让出主线程
4. 等浏览器空闲时继续未完成的工作
`;

// ============================================================
// Part 2: Scheduler 的核心概念
// ============================================================

/**
 * 📊 核心概念 1：优先级（Priority）
 *
 * 📁 源码位置: packages/scheduler/src/SchedulerPriorities.js
 */

const schedulerPriorities = `
📊 Scheduler 优先级

┌─────────────────────────────────────────────────────────────────────────┐
│ 优先级名称              │ 值 │ 超时时间        │ 使用场景              │
├─────────────────────────┼────┼─────────────────┼───────────────────────┤
│ ImmediatePriority       │ 1  │ -1ms（立即）    │ 同步任务、用户输入     │
│ UserBlockingPriority    │ 2  │ 250ms           │ 用户交互（点击等）     │
│ NormalPriority          │ 3  │ 5000ms          │ 普通更新               │
│ LowPriority             │ 4  │ 10000ms         │ 数据预加载             │
│ IdlePriority            │ 5  │ maxInt（永不）  │ 空闲时执行             │
└─────────────────────────┴────┴─────────────────┴───────────────────────┘

// 源码定义
export const NoPriority = 0;
export const ImmediatePriority = 1;
export const UserBlockingPriority = 2;
export const NormalPriority = 3;
export const LowPriority = 4;
export const IdlePriority = 5;

超时时间的作用：
- expirationTime = currentTime + timeout
- 过期的任务会被强制执行，防止饥饿
- 高优先级任务超时时间短，会更快过期
`;

/**
 * 📊 核心概念 2：任务队列（Task Queue）
 *
 * Scheduler 维护两个队列：
 * 1. taskQueue: 已经可以执行的任务
 * 2. timerQueue: 延迟任务（还没到开始时间）
 */

const taskQueues = `
📊 两个任务队列

┌─────────────────────────────────────────────────────────────────────────┐
│                         taskQueue（立即执行队列）                        │
│                                                                         │
│  存储：已经到开始时间、可以执行的任务                                    │
│  排序：按 expirationTime（过期时间）排序，使用最小堆                      │
│  特点：堆顶是最快过期的任务                                              │
│                                                                         │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                    │
│  │ T1  │  │ T2  │  │ T3  │  │ T4  │  ...                               │
│  │exp:5│  │exp:8│  │exp:9│  │exp:12│                                   │
│  └─────┘  └─────┘  └─────┘  └─────┘                                    │
│     ↑                                                                   │
│     堆顶（下一个执行）                                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        timerQueue（延迟任务队列）                        │
│                                                                         │
│  存储：还没到开始时间的延迟任务                                          │
│  排序：按 startTime（开始时间）排序                                      │
│  转移：到了 startTime 后，移动到 taskQueue                              │
│                                                                         │
│  ┌─────┐  ┌─────┐  ┌─────┐                                             │
│  │ T5  │  │ T6  │  │ T7  │                                             │
│  │st:20│  │st:50│  │st:100│                                            │
│  └─────┘  └─────┘  └─────┘                                             │
│     ↑                                                                   │
│     堆顶（最快开始的延迟任务）                                           │
└─────────────────────────────────────────────────────────────────────────┘
`;

/**
 * 📊 核心概念 3：时间切片（Time Slicing）
 */

const timeSlicing = `
📊 时间切片机制

关键变量:
- frameInterval: 每个时间片的长度，默认 5ms
- startTime: 当前时间片的开始时间

shouldYieldToHost() 的判断逻辑:

function shouldYieldToHost() {
  const timeElapsed = getCurrentTime() - startTime;

  // 1. 如果执行时间 < 5ms，继续执行
  if (timeElapsed < frameInterval) {
    return false;  // 不让出
  }

  // 2. 如果有待处理的用户输入，让出
  if (isInputPending()) {
    return true;   // 让出
  }

  // 3. 执行时间过长，强制让出
  if (timeElapsed > maxInterval) {
    return true;   // 让出
  }

  return true;     // 默认让出
}

时间阈值:
- frameYieldMs = 5ms      （普通让出阈值）
- continuousYieldMs = 50ms （连续输入让出阈值）
- maxYieldMs = 300ms       （最大阈值，强制让出）
`;

/**
 * 📊 核心概念 4：MessageChannel 调度
 */

const messageChannelScheduling = `
📊 为什么用 MessageChannel 而不是 setTimeout?

setTimeout 的问题:
- 最小延迟 4ms（浏览器限制）
- 在后台标签页会被节流
- 不够精确

MessageChannel 的优势:
- 没有 4ms 限制
- 在宏任务队列中，不会阻塞渲染
- 可以被更高优先级任务打断

📁 源码位置: packages/scheduler/src/forks/Scheduler.js 第 566-580 行

// 调度机制的选择
let schedulePerformWorkUntilDeadline;

if (typeof localSetImmediate === 'function') {
  // Node.js 环境
  schedulePerformWorkUntilDeadline = () => {
    localSetImmediate(performWorkUntilDeadline);
  };
} else if (typeof MessageChannel !== 'undefined') {
  // 浏览器环境（首选）
  const channel = new MessageChannel();
  const port = channel.port2;
  channel.port1.onmessage = performWorkUntilDeadline;
  schedulePerformWorkUntilDeadline = () => {
    port.postMessage(null);
  };
} else {
  // 降级方案
  schedulePerformWorkUntilDeadline = () => {
    localSetTimeout(performWorkUntilDeadline, 0);
  };
}
`;

// ============================================================
// Part 3: Scheduler 与 React 的关系
// ============================================================

/**
 * 📊 Scheduler 在 React 架构中的位置
 */

const schedulerInReact = `
📊 Scheduler 在 React 架构中的位置

┌─────────────────────────────────────────────────────────────────────────┐
│                           React 应用                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   用户交互 ──→ 触发更新 ──→ scheduleUpdateOnFiber                       │
│                               │                                         │
│                               ▼                                         │
│                     ensureRootIsScheduled                               │
│                               │                                         │
│               ┌───────────────┼───────────────┐                         │
│               ▼               ▼               ▼                         │
│           同步更新         并发更新       延迟更新                       │
│         (SyncLane)    (DefaultLane)   (TransitionLane)                  │
│               │               │               │                         │
│               ▼               ▼               ▼                         │
│         微任务执行      Scheduler 调度   Scheduler 调度                  │
│                               │               │                         │
│                               ▼               ▼                         │
│                     scheduleCallback(priority, callback)                │
│                               │                                         │
│                               ▼                                         │
│                     ┌─────────────────┐                                 │
│                     │   Scheduler     │                                 │
│                     │   ┌─────────┐   │                                 │
│                     │   │taskQueue│   │                                 │
│                     │   │timerQ   │   │                                 │
│                     │   └─────────┘   │                                 │
│                     └─────────────────┘                                 │
│                               │                                         │
│                               ▼                                         │
│                  performConcurrentWorkOnRoot                            │
│                               │                                         │
│                               ▼                                         │
│                     workLoopConcurrent                                  │
│                     (可中断的渲染循环)                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

关键点:
1. React Reconciler 不直接执行渲染，而是通过 Scheduler 调度
2. Scheduler 决定何时执行、执行多久
3. 渲染过程可以被打断，让出主线程给更重要的任务
`;

// ============================================================
// Part 4: 面试要点
// ============================================================

const interviewPoints = `
💡 面试要点

Q1: 为什么 React 需要自己实现 Scheduler，而不用 requestIdleCallback?
A:
   1. requestIdleCallback 兼容性差（Safari 不支持）
   2. 一帧内只执行一次，频率太低
   3. 无法控制优先级
   4. 后台标签页不执行

Q2: Scheduler 的核心思想是什么？
A: 时间切片 + 优先级调度
   - 把长任务拆成小块（~5ms）
   - 每块执行完检查是否让出
   - 高优先级任务可以打断低优先级

Q3: scheduleCallback 和 scheduleSyncCallback 有什么区别？
A:
   - scheduleCallback: 放入 Scheduler 的 taskQueue，异步执行
   - scheduleSyncCallback: 放入内部队列，在微任务中同步执行

Q4: 什么情况下任务会被打断？
A:
   1. 时间片用完（执行时间 > 5ms）
   2. 有用户输入待处理（isInputPending）
   3. 有更高优先级任务
   4. 任务已过期需要让给其他过期任务
`;

export {
  problemWithoutScheduler,
  schedulerSolution,
  schedulerPriorities,
  taskQueues,
  timeSlicing,
  messageChannelScheduling,
  schedulerInReact,
  interviewPoints,
};

