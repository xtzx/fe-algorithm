/**
 * ============================================================
 * 📚 Phase 6: Scheduler 调度机制 - Part 4: 源码导航与重要函数
 * ============================================================
 *
 * 本文件提供源码阅读指南，标注关键函数的位置和作用
 */

// ============================================================
// Part 1: 源码文件结构
// ============================================================

const sourceCodeStructure = `
📁 Scheduler 相关源码结构

packages/
├── scheduler/
│   └── src/
│       ├── forks/
│       │   └── Scheduler.js        ⭐ 核心！调度器主逻辑
│       ├── SchedulerMinHeap.js     ⭐ 最小堆实现
│       ├── SchedulerPriorities.js  优先级常量定义
│       └── SchedulerFeatureFlags.js 特性开关
│
└── react-reconciler/
    └── src/
        ├── ReactFiberWorkLoop.new.js  ⭐ 核心！React 调度入口
        ├── ReactFiberLane.new.js       Lane 优先级模型
        ├── ReactEventPriorities.new.js 事件优先级
        └── ReactFiberSyncTaskQueue.new.js 同步任务队列

阅读顺序建议:
1. SchedulerPriorities.js    - 理解优先级定义
2. SchedulerMinHeap.js       - 理解数据结构
3. Scheduler.js              - 理解调度核心
4. ReactFiberWorkLoop.new.js - 理解 React 如何使用 Scheduler
`;

// ============================================================
// Part 2: Scheduler.js 关键函数
// ============================================================

/**
 * 📁 packages/scheduler/src/forks/Scheduler.js
 */

const schedulerJsFunctions = `
📁 packages/scheduler/src/forks/Scheduler.js

┌─────────────────────────────────────────────────────────────────────────────┐
│ 函数名                    │ 行号      │ 作用                                │
├───────────────────────────┼───────────┼─────────────────────────────────────┤
│ unstable_scheduleCallback │ 308-388   │ ⭐ 核心！调度任务入口                │
│ unstable_cancelCallback   │ 406-419   │ 取消任务                            │
│ workLoop                  │ 189-244   │ ⭐ 核心！任务执行循环                │
│ flushWork                 │ 147-187   │ 刷新任务                            │
│ shouldYieldToHost         │ 440-483   │ ⭐ 判断是否让出主线程                │
│ advanceTimers             │ 106-128   │ 推进延迟任务                        │
│ handleTimeout             │ 130-145   │ 处理延迟任务到期                    │
│ requestHostCallback       │ 582-588   │ 请求宿主回调                        │
│ requestHostTimeout        │ 590-594   │ 请求延迟回调                        │
│ performWorkUntilDeadline  │ 515-548   │ 执行工作直到截止时间                 │
└───────────────────────────┴───────────┴─────────────────────────────────────┘

⭐ unstable_scheduleCallback (308-388)
────────────────────────────────────────
作用：创建任务并加入队列
输入：priorityLevel, callback, options
输出：Task 对象
关键逻辑：
  1. 计算 startTime（是否有 delay）
  2. 根据 priorityLevel 计算 timeout
  3. expirationTime = startTime + timeout
  4. 创建 Task 对象
  5. 有 delay → timerQueue，无 delay → taskQueue
  6. requestHostCallback(flushWork)

⭐ workLoop (189-244)
────────────────────────────────────────
作用：循环执行任务
关键逻辑：
  while (currentTask !== null) {
    if (未过期 && shouldYield) break;
    执行 callback
    if (返回函数) 保存 continuation
    else pop(taskQueue)
  }
  return 是否还有任务

⭐ shouldYieldToHost (440-483)
────────────────────────────────────────
作用：判断是否应该让出主线程
关键逻辑：
  1. timeElapsed < 5ms → 不让出
  2. needsPaint → 让出
  3. isInputPending → 让出
  4. timeElapsed > 300ms → 强制让出
`;

/**
 * 📊 unstable_scheduleCallback 详解
 */

const scheduleCallbackDetail = `
📊 unstable_scheduleCallback 详细解读

function unstable_scheduleCallback(priorityLevel, callback, options) {
  // 第 309 行: 获取当前时间
  var currentTime = getCurrentTime();

  // 第 311-321 行: 计算开始时间
  var startTime;
  if (typeof options === 'object' && options !== null) {
    var delay = options.delay;
    if (typeof delay === 'number' && delay > 0) {
      startTime = currentTime + delay;  // 延迟任务
    } else {
      startTime = currentTime;
    }
  } else {
    startTime = currentTime;
  }

  // 第 323-341 行: 根据优先级设置超时时间
  var timeout;
  switch (priorityLevel) {
    case ImmediatePriority:
      timeout = IMMEDIATE_PRIORITY_TIMEOUT;  // -1
      break;
    case UserBlockingPriority:
      timeout = USER_BLOCKING_PRIORITY_TIMEOUT;  // 250
      break;
    case IdlePriority:
      timeout = IDLE_PRIORITY_TIMEOUT;  // maxSigned31BitInt
      break;
    case LowPriority:
      timeout = LOW_PRIORITY_TIMEOUT;  // 10000
      break;
    case NormalPriority:
    default:
      timeout = NORMAL_PRIORITY_TIMEOUT;  // 5000
      break;
  }

  // 第 343 行: 计算过期时间
  var expirationTime = startTime + timeout;

  // 第 345-355 行: 创建任务
  var newTask = {
    id: taskIdCounter++,
    callback,
    priorityLevel,
    startTime,
    expirationTime,
    sortIndex: -1,
  };

  // 第 357-387 行: 入队
  if (startTime > currentTime) {
    // 延迟任务
    newTask.sortIndex = startTime;
    push(timerQueue, newTask);

    if (peek(taskQueue) === null && newTask === peek(timerQueue)) {
      // 这是唯一的任务，设置定时器
      if (isHostTimeoutScheduled) {
        cancelHostTimeout();
      } else {
        isHostTimeoutScheduled = true;
      }
      requestHostTimeout(handleTimeout, startTime - currentTime);
    }
  } else {
    // 立即任务
    newTask.sortIndex = expirationTime;
    push(taskQueue, newTask);

    if (!isHostCallbackScheduled && !isPerformingWork) {
      isHostCallbackScheduled = true;
      requestHostCallback(flushWork);
    }
  }

  return newTask;
}
`;

/**
 * 📊 workLoop 详解
 */

const workLoopDetail = `
📊 workLoop 详细解读

function workLoop(hasTimeRemaining, initialTime) {
  // 第 190 行: 初始化时间
  let currentTime = initialTime;

  // 第 191 行: 推进延迟任务（检查 timerQueue 中是否有到期的）
  advanceTimers(currentTime);

  // 第 192 行: 取堆顶任务
  currentTask = peek(taskQueue);

  // 第 193-233 行: 主循环
  while (
    currentTask !== null &&
    !(enableSchedulerDebugging && isSchedulerPaused)
  ) {
    // 第 197-203 行: ⭐ 关键判断
    if (
      currentTask.expirationTime > currentTime &&  // 任务未过期
      (!hasTimeRemaining || shouldYieldToHost())   // 但需要让出
    ) {
      // 让出主线程
      break;
    }

    // 第 204 行: 获取回调
    const callback = currentTask.callback;

    if (typeof callback === 'function') {
      // 第 206 行: 清空 callback（防止重复执行）
      currentTask.callback = null;

      // 第 207 行: 设置当前优先级
      currentPriorityLevel = currentTask.priorityLevel;

      // 第 208 行: 判断是否超时
      const didUserCallbackTimeout = currentTask.expirationTime <= currentTime;

      // 第 212 行: ⭐ 执行任务
      const continuationCallback = callback(didUserCallbackTimeout);

      // 第 213 行: 更新时间
      currentTime = getCurrentTime();

      // 第 214-227 行: 处理返回值
      if (typeof continuationCallback === 'function') {
        // 任务未完成，保存 continuation
        currentTask.callback = continuationCallback;
      } else {
        // 任务完成，移出队列
        if (currentTask === peek(taskQueue)) {
          pop(taskQueue);
        }
      }

      // 第 228 行: 再次检查延迟任务
      advanceTimers(currentTime);
    } else {
      // callback 为 null，任务被取消
      pop(taskQueue);
    }

    // 第 232 行: 取下一个任务
    currentTask = peek(taskQueue);
  }

  // 第 234-243 行: 返回是否还有任务
  if (currentTask !== null) {
    return true;
  } else {
    const firstTimer = peek(timerQueue);
    if (firstTimer !== null) {
      requestHostTimeout(handleTimeout, firstTimer.startTime - currentTime);
    }
    return false;
  }
}
`;

// ============================================================
// Part 3: ReactFiberWorkLoop.new.js 关键函数
// ============================================================

const reactFiberWorkLoopFunctions = `
📁 packages/react-reconciler/src/ReactFiberWorkLoop.new.js

┌─────────────────────────────────────────────────────────────────────────────┐
│ 函数名                      │ 行号       │ 作用                              │
├─────────────────────────────┼────────────┼───────────────────────────────────┤
│ scheduleUpdateOnFiber       │ 533-690    │ ⭐ 调度更新入口                    │
│ ensureRootIsScheduled       │ 696-825    │ ⭐ 确保根节点被调度                │
│ performConcurrentWorkOnRoot │ 829-1020   │ ⭐ 并发渲染入口                    │
│ performSyncWorkOnRoot       │ 1022-1150  │ 同步渲染入口                      │
│ renderRootConcurrent        │ 1748-1826  │ 并发渲染                          │
│ renderRootSync              │ 1680-1746  │ 同步渲染                          │
│ workLoopConcurrent          │ 1829-1834  │ ⭐ 可中断工作循环                  │
│ workLoopSync                │ 1823-1827  │ 同步工作循环                      │
│ performUnitOfWork           │ 1836-1867  │ 处理单个 Fiber                    │
│ flushPassiveEffects         │ 2369-2403  │ 执行 passive effects              │
└─────────────────────────────┴────────────┴───────────────────────────────────┘

⭐ scheduleUpdateOnFiber (533-690)
────────────────────────────────────────
作用：调度 Fiber 更新
调用时机：setState, forceUpdate 等
关键逻辑：
  1. markRootUpdated(root, lane) - 标记有更新
  2. ensureRootIsScheduled(root) - 确保被调度

⭐ ensureRootIsScheduled (696-825)
────────────────────────────────────────
作用：确保根节点在 Scheduler 中被调度
关键逻辑：
  1. markStarvedLanesAsExpired() - 标记饥饿的 lane
  2. getNextLanes() - 获取下一个要处理的 lanes
  3. 如果优先级相同，复用现有任务
  4. 如果是 SyncLane，用微任务调度
  5. 否则用 scheduleCallback 调度

⭐ performConcurrentWorkOnRoot (829-1020)
────────────────────────────────────────
作用：执行并发渲染
被谁调用：Scheduler 的 workLoop
关键逻辑：
  1. flushPassiveEffects() - 先执行 pending effects
  2. getNextLanes() - 获取要处理的 lanes
  3. shouldTimeSlice 判断是否走时间切片
  4. renderRootConcurrent 或 renderRootSync
  5. 如果未完成，返回 continuation
  6. 如果完成，commitRoot()

⭐ workLoopConcurrent (1829-1834)
────────────────────────────────────────
作用：可中断的工作循环
function workLoopConcurrent() {
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}
`;

/**
 * 📊 ensureRootIsScheduled 详解
 */

const ensureRootIsScheduledDetail = `
📊 ensureRootIsScheduled 详细解读

function ensureRootIsScheduled(root, currentTime) {
  // 第 697 行: 获取现有任务
  const existingCallbackNode = root.callbackNode;

  // 第 701 行: 标记饥饿的 lanes
  markStarvedLanesAsExpired(root, currentTime);

  // 第 704-707 行: 获取下一个要处理的 lanes
  const nextLanes = getNextLanes(
    root,
    root === workInProgressRoot ? workInProgressRootRenderLanes : NoLanes,
  );

  // 第 709-716 行: 没有任务，取消调度
  if (nextLanes === NoLanes) {
    if (existingCallbackNode !== null) {
      cancelCallback(existingCallbackNode);
    }
    root.callbackNode = null;
    root.callbackPriority = NoLane;
    return;
  }

  // 第 720 行: 获取最高优先级
  const newCallbackPriority = getHighestPriorityLane(nextLanes);

  // 第 723-750 行: ⭐ 关键！判断是否复用现有任务
  const existingCallbackPriority = root.callbackPriority;
  if (existingCallbackPriority === newCallbackPriority) {
    // 优先级相同，复用现有任务
    return;
  }

  // 第 752-755 行: 优先级不同，取消旧任务
  if (existingCallbackNode != null) {
    cancelCallback(existingCallbackNode);
  }

  // 第 758-821 行: 调度新任务
  let newCallbackNode;
  if (newCallbackPriority === SyncLane) {
    // 同步优先级：用微任务调度
    if (root.tag === LegacyRoot) {
      scheduleLegacySyncCallback(performSyncWorkOnRoot.bind(null, root));
    } else {
      scheduleSyncCallback(performSyncWorkOnRoot.bind(null, root));
    }
    if (supportsMicrotasks) {
      scheduleMicrotask(() => {
        if ((executionContext & (RenderContext | CommitContext)) === NoContext) {
          flushSyncCallbacks();
        }
      });
    }
    newCallbackNode = null;
  } else {
    // 其他优先级：用 Scheduler 调度
    let schedulerPriorityLevel;
    switch (lanesToEventPriority(nextLanes)) {
      case DiscreteEventPriority:
        schedulerPriorityLevel = ImmediateSchedulerPriority;
        break;
      case ContinuousEventPriority:
        schedulerPriorityLevel = UserBlockingSchedulerPriority;
        break;
      case DefaultEventPriority:
        schedulerPriorityLevel = NormalSchedulerPriority;
        break;
      case IdleEventPriority:
        schedulerPriorityLevel = IdleSchedulerPriority;
        break;
      default:
        schedulerPriorityLevel = NormalSchedulerPriority;
        break;
    }
    newCallbackNode = scheduleCallback(
      schedulerPriorityLevel,
      performConcurrentWorkOnRoot.bind(null, root),
    );
  }

  // 第 823-824 行: 保存任务引用
  root.callbackPriority = newCallbackPriority;
  root.callbackNode = newCallbackNode;
}
`;

// ============================================================
// Part 4: 其他相关文件
// ============================================================

const otherRelatedFiles = `
📁 其他相关文件

packages/scheduler/src/SchedulerMinHeap.js
────────────────────────────────────────
- push(heap, node)   - 插入并上浮
- peek(heap)         - 查看堆顶
- pop(heap)          - 弹出堆顶
- siftUp()           - 上浮
- siftDown()         - 下沉
- compare(a, b)      - 比较函数

packages/react-reconciler/src/ReactFiberLane.new.js
────────────────────────────────────────
- SyncLane, DefaultLane, TransitionLanes... - Lane 常量
- getNextLanes()           - 获取下一个要处理的 lanes
- getHighestPriorityLane() - 获取最高优先级 lane
- markRootUpdated()        - 标记根有更新
- markStarvedLanesAsExpired() - 标记饥饿的 lanes

packages/react-reconciler/src/ReactEventPriorities.new.js
────────────────────────────────────────
- DiscreteEventPriority      - 离散事件优先级
- ContinuousEventPriority    - 连续事件优先级
- DefaultEventPriority       - 默认优先级
- IdleEventPriority          - 空闲优先级
- lanesToEventPriority()     - Lane 转事件优先级
`;

// ============================================================
// Part 5: 与其他 Phase 的关联
// ============================================================

const relationWithOtherPhases = `
📊 Scheduler 与其他 Phase 的关联

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 2: Fiber 架构                                                        │
│  ─────────────────                                                          │
│  - Fiber 的链表结构支持可中断渲染                                            │
│  - workInProgress 保存当前处理位置                                          │
│  - alternate 支持双缓冲                                                     │
│                                                                             │
│  Scheduler 在 Fiber 中的介入点：                                            │
│  performUnitOfWork() → beginWork() / completeWork()                         │
│  每处理一个 Fiber 后检查 shouldYield()                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 3: 渲染流程                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  渲染入口点:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Scheduler.workLoop()                                                 │   │
│  │     └── performConcurrentWorkOnRoot()    ← Scheduler 调用这里        │   │
│  │             ├── renderRootConcurrent()                               │   │
│  │             │       └── workLoopConcurrent()                         │   │
│  │             │               └── performUnitOfWork()                  │   │
│  │             │                       ├── beginWork()                  │   │
│  │             │                       └── completeUnitOfWork()         │   │
│  │             └── commitRoot()                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Lane 模型                                                                  │
│  ─────────────────                                                          │
│                                                                             │
│  优先级映射链:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 用户事件 → EventPriority → Lane → SchedulerPriority → Task           │   │
│  │                                                                      │   │
│  │ onClick  → Discrete      → Sync → Immediate       → 立即执行         │   │
│  │ onScroll → Continuous    → Input → UserBlocking   → 250ms timeout   │   │
│  │ setState → Default       → Default → Normal       → 5000ms timeout  │   │
│  │ transition → Transition  → Transition → Normal    → 5000ms timeout  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 7: 并发特性                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  startTransition:                                                           │
│  - 将更新标记为 TransitionLane                                              │
│  - TransitionLane 对应 NormalPriority                                       │
│  - 可以被更高优先级打断                                                      │
│                                                                             │
│  useDeferredValue:                                                          │
│  - 内部使用 startTransition                                                 │
│  - 返回值在低优先级更新中延迟                                               │
│                                                                             │
│  Suspense + Concurrent Mode:                                                │
│  - 挂起时保存 workInProgress                                                │
│  - resolve 后重新调度                                                       │
│  - 可以显示 fallback 而不阻塞                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 6: 面试题总结
// ============================================================

const interviewQuestions = `
💡 Phase 6 面试题总结

Q1: Scheduler 解决了什么问题？
A: 解决长任务阻塞主线程的问题。
   通过时间切片把长任务拆成小块（~5ms），
   每块执行后检查是否让出主线程，
   保证用户交互不被阻塞。

Q2: Scheduler 有几种优先级？分别是什么？
A: 5 种
   - ImmediatePriority(1): 立即执行，-1ms 过期
   - UserBlockingPriority(2): 用户交互，250ms
   - NormalPriority(3): 普通更新，5000ms
   - LowPriority(4): 低优先级，10000ms
   - IdlePriority(5): 空闲执行，几乎不过期

Q3: 为什么用 MessageChannel 而不是 setTimeout？
A: - setTimeout 有 4ms 最小延迟
   - MessageChannel 没有这个限制
   - 在宏任务队列中，不阻塞渲染

Q4: 任务是如何被打断和恢复的？
A: 打断：shouldYieldToHost() 返回 true 时，workLoop 跳出
   恢复：callback 返回 continuation 函数，
        Scheduler 保存到 task.callback，
        下次执行时调用 continuation

Q5: Lane 和 Scheduler Priority 是什么关系？
A: Lane 是 React 内部优先级模型，
   需要通过 lanesToEventPriority 转换为 EventPriority，
   再映射到 Scheduler Priority。
   SyncLane → Immediate
   InputContinuousLane → UserBlocking
   DefaultLane → Normal
   IdleLane → Idle

Q6: ensureRootIsScheduled 什么时候会复用任务？
A: 当新的更新优先级与现有任务优先级相同时。
   这避免了高频更新导致任务堆积。

Q7: 同步更新（SyncLane）是怎么调度的？
A: 不走 Scheduler，而是：
   1. 放入内部同步队列（scheduleSyncCallback）
   2. 用微任务调度（scheduleMicrotask）
   3. 在微任务中执行（flushSyncCallbacks）

Q8: 什么情况下任务会过期？
A: expirationTime = startTime + timeout
   当 currentTime >= expirationTime 时任务过期。
   过期的任务会被强制执行，防止饥饿。

Q9: advanceTimers 的作用是什么？
A: 检查 timerQueue 中是否有到期的延迟任务，
   如果有，移动到 taskQueue。

Q10: React 如何保证 Fiber 树可以从中断处继续？
A: - workInProgress 保存当前 Fiber
   - workInProgressRoot 保存当前根
   - workInProgressRootRenderLanes 保存当前 lanes
   这些是模块级变量，中断后不会丢失。
`;

// ============================================================
// Part 7: 学习检查清单
// ============================================================

const learningChecklist = `
✅ Phase 6 学习检查

□ 核心概念
  □ 理解 Scheduler 解决的问题（长任务阻塞）
  □ 理解时间切片的原理
  □ 理解 5 种优先级及其超时时间
  □ 理解 taskQueue 和 timerQueue 的区别

□ 数据结构
  □ 理解 Task 的各个字段
  □ 理解最小堆的工作原理
  □ 理解 Lane 到 Scheduler Priority 的映射

□ 核心流程
  □ 能说清 scheduleCallback 的完整流程
  □ 能说清 workLoop 的执行逻辑
  □ 能说清 shouldYieldToHost 的判断逻辑
  □ 能说清任务的打断和恢复机制

□ React 集成
  □ 理解 scheduleUpdateOnFiber 的作用
  □ 理解 ensureRootIsScheduled 的作用
  □ 理解 performConcurrentWorkOnRoot 的作用
  □ 理解 workLoopConcurrent 的作用

□ 源码位置
  □ 能找到 Scheduler.js 的关键函数
  □ 能找到 ReactFiberWorkLoop.new.js 的关键函数
  □ 能说清各函数的调用关系

□ 实践
  □ 能解释 startTransition 如何利用 Scheduler
  □ 能解释高频更新为什么不会导致任务堆积
  □ 能用 React DevTools 观察调度行为
`;

export {
  sourceCodeStructure,
  schedulerJsFunctions,
  scheduleCallbackDetail,
  workLoopDetail,
  reactFiberWorkLoopFunctions,
  ensureRootIsScheduledDetail,
  otherRelatedFiles,
  relationWithOtherPhases,
  interviewQuestions,
  learningChecklist,
};

