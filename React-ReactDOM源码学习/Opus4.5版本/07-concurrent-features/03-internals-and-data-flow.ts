/**
 * ============================================================
 * 📚 Phase 7: 并发特性 - Part 3: 内部原理与数据流
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 * - packages/react-reconciler/src/ReactFiberLane.new.js
 * - packages/react-reconciler/src/ReactFiberHooks.new.js
 */

// ============================================================
// Part 1: 并发更新的完整流程
// ============================================================

/**
 * 📊 从用户交互到渲染完成的完整流程
 */

const concurrentUpdateFlow = `
📊 并发更新完整流程

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Phase 1: 更新产生                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  用户输入 onChange                                                          │
│       │                                                                     │
│       ▼                                                                     │
│  setInputValue(newValue)  ─────────────────────────────────────────┐       │
│       │                                                            │        │
│       │ 高优先级                                                    │        │
│       ▼                                                            ▼        │
│  requestUpdateLane()                                    startTransition     │
│       │                                                      │              │
│       │ 返回 SyncLane 或 DefaultLane                          │              │
│       │                                                      ▼              │
│       │                                        setSearchResults(filter())   │
│       │                                                      │              │
│       │                                                      │ 低优先级     │
│       │                                                      ▼              │
│       │                                         requestUpdateLane()         │
│       │                                                      │              │
│       │                                         返回 TransitionLane        │
│       │                                                      │              │
│       └──────────────────────┬───────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│                                                                             │
│  Phase 2: 调度更新                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│                    scheduleUpdateOnFiber(root, fiber, lane)                 │
│                              │                                              │
│                              ▼                                              │
│                    markRootUpdated(root, lane)                              │
│                    root.pendingLanes |= lane                                │
│                              │                                              │
│                              ▼                                              │
│                    ensureRootIsScheduled(root)                              │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│         SyncLane?      并发 Lane?        相同优先级?                        │
│              │               │               │                              │
│              ▼               ▼               ▼                              │
│        微任务执行      scheduleCallback    复用任务                         │
│                              │                                              │
│                              ▼                                              │
│                    Scheduler.push(taskQueue)                                │
│                                                                             │
│  Phase 3: 执行渲染                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│                    performConcurrentWorkOnRoot(root)                        │
│                              │                                              │
│                              ▼                                              │
│                    getNextLanes(root)  ← 获取要处理的 lanes                 │
│                              │                                              │
│                              ▼                                              │
│                    shouldTimeSlice = ?                                      │
│                    !includesBlockingLane(lanes) &&                          │
│                    !includesExpiredLane(lanes)                              │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼                                              │
│    shouldTimeSlice=true   shouldTimeSlice=false                             │
│              │               │                                              │
│              ▼               ▼                                              │
│    renderRootConcurrent  renderRootSync                                     │
│    (可中断)              (不可中断)                                          │
│              │                                                              │
│              ▼                                                              │
│                                                                             │
│  Phase 4: 工作循环                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│    workLoopConcurrent()                                                     │
│    while (workInProgress !== null && !shouldYield()) {                      │
│        performUnitOfWork(workInProgress);                                   │
│    }                                                                        │
│              │                                                              │
│              │  每个 Fiber 处理后检查：                                      │
│              │  - shouldYield()? → 时间片用完，让出                         │
│              │  - 有更高优先级? → 被打断                                    │
│              │                                                              │
│              ▼                                                              │
│                                                                             │
│  Phase 5: 提交或中断                                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│       ┌───────────────────────┬───────────────────────┐                     │
│       ▼                       ▼                       ▼                     │
│    完成渲染               被打断                  需要让出                  │
│       │                       │                       │                     │
│       ▼                       ▼                       ▼                     │
│  commitRoot()          返回 continuation         返回 continuation         │
│  更新 DOM              保存 workInProgress       下次继续                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: Lane 如何标记并发更新
// ============================================================

/**
 * 📊 更新如何被打上 Lane
 */

const laneAssignment = `
📊 更新如何被分配 Lane

📁 核心函数: packages/react-reconciler/src/ReactFiberWorkLoop.new.js

requestUpdateLane(fiber) 的逻辑：
─────────────────────────────────────

function requestUpdateLane(fiber) {
  // 1. 检查是否在 transition 中
  const isTransition = ReactCurrentBatchConfig.transition !== null;

  if (isTransition) {
    // ⭐ startTransition 内的更新
    return claimNextTransitionLane();
    // 返回 TransitionLane1 ~ TransitionLane16（循环分配）
  }

  // 2. 获取当前事件优先级
  const updateLane = getCurrentUpdatePriority();
  if (updateLane !== NoLane) {
    return updateLane;
  }

  // 3. 获取当前事件类型的优先级
  const eventLane = getCurrentEventPriority();
  return eventLane;
}

Lane 分配结果：
─────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ 场景                      │ Lane                    │ 是否可中断            │
├───────────────────────────┼─────────────────────────┼───────────────────────┤
│ ReactDOM.flushSync        │ SyncLane                │ ❌ 不可中断           │
│ onClick 点击              │ DiscreteEventPriority   │ ❌ 不可中断           │
│ onScroll 滚动             │ ContinuousEventPriority │ ❌ 不可中断           │
│ 普通 setState             │ DefaultLane             │ 取决于具体情况        │
│ startTransition 内        │ TransitionLane          │ ✅ 可中断             │
│ useDeferredValue 触发     │ TransitionLane          │ ✅ 可中断             │
│ Suspense retry            │ RetryLane               │ ✅ 可中断             │
└───────────────────────────┴─────────────────────────┴───────────────────────┘

claimNextTransitionLane 的实现：
─────────────────────────────────────

📁 packages/react-reconciler/src/ReactFiberLane.new.js 第 493-502 行

let nextTransitionLane = TransitionLane1;

function claimNextTransitionLane() {
  const lane = nextTransitionLane;
  nextTransitionLane <<= 1;  // 左移一位

  if ((nextTransitionLane & TransitionLanes) === NoLanes) {
    // 超出范围，回到第一个
    nextTransitionLane = TransitionLane1;
  }

  return lane;
}

// 这意味着每个 transition 会分配不同的 Lane
// 第1个: TransitionLane1 (0b0000000000000000000000001000000)
// 第2个: TransitionLane2 (0b0000000000000000000000010000000)
// ...
// 第16个: TransitionLane16
// 第17个: 循环回到 TransitionLane1
`;

// ============================================================
// Part 3: 如何决定是否可中断
// ============================================================

/**
 * 📊 中断判断机制
 */

const interruptionMechanism = `
📊 中断判断机制

📁 packages/react-reconciler/src/ReactFiberWorkLoop.new.js 第 877-883 行

performConcurrentWorkOnRoot 中的关键判断:
─────────────────────────────────────

const shouldTimeSlice =
  !includesBlockingLane(root, lanes) &&  // 不包含阻塞型 Lane
  !includesExpiredLane(root, lanes) &&   // 没有过期的 Lane
  !didTimeout;                           // Scheduler 没有超时

let exitStatus = shouldTimeSlice
  ? renderRootConcurrent(root, lanes)    // ⭐ 可中断
  : renderRootSync(root, lanes);         // 不可中断

includesBlockingLane 的实现:
─────────────────────────────────────

function includesBlockingLane(root, lanes) {
  // BlockingLane = SyncLane | InputContinuousLane
  // 如果包含这些 Lane，必须同步完成，不能中断

  if (allowConcurrentByDefault) {
    // 如果开启了"默认并发"，只有 SyncLane 阻塞
    return (lanes & SyncLane) !== NoLanes;
  }

  // 否则，SyncLane、InputContinuousLane、DefaultLane 都阻塞
  const SyncDefaultLanes = InputContinuousLane | DefaultLane;
  return (lanes & SyncDefaultLanes) !== NoLanes;
}

中断的两种情况:
─────────────────────────────────────

情况1: 时间片用完
  workLoopConcurrent() 中:
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }

  // shouldYield() 来自 Scheduler
  // 当执行时间 > 5ms 时返回 true

情况2: 更高优先级任务到来
  在 ensureRootIsScheduled 中:
  if (existingCallbackPriority !== newCallbackPriority) {
    // 新任务优先级不同
    cancelCallback(existingCallbackNode);  // 取消当前任务
    // 调度新任务...
  }
`;

// ============================================================
// Part 4: 被打断后的恢复机制
// ============================================================

/**
 * 📊 任务打断与恢复
 */

const interruptAndResume = `
📊 任务打断与恢复机制

被打断时保存的状态:
─────────────────────────────────────

// 模块级变量（ReactFiberWorkLoop.new.js）
let workInProgress: Fiber | null = null;        // 当前处理的 Fiber
let workInProgressRoot: FiberRoot | null = null; // 当前根节点
let workInProgressRootRenderLanes: Lanes = NoLanes; // 当前渲染的 lanes

// 被打断时：
// 1. workInProgress 保留当前位置
// 2. performConcurrentWorkOnRoot 返回 continuation
// 3. Scheduler 保存 continuation 到 task.callback

恢复执行:
─────────────────────────────────────

function performConcurrentWorkOnRoot(root, didTimeout) {
  // ...渲染逻辑...

  if (workInProgress !== null) {
    // ⭐ 渲染未完成（被打断了）
    // 返回自身作为 continuation
    return performConcurrentWorkOnRoot.bind(null, root);
  }

  // 渲染完成
  // ...commit 逻辑...
}

// 在 renderRootConcurrent 中:
function renderRootConcurrent(root, lanes) {
  // 检查是否可以继续之前的工作
  if (workInProgressRoot !== root ||
      workInProgressRootRenderLanes !== lanes) {
    // 不能继续，重新开始
    prepareFreshStack(root, lanes);
  } else {
    // ⭐ 可以继续！
    // workInProgress 还在，从上次位置继续
  }

  // 执行工作循环
  workLoopConcurrent();

  // 检查结果
  if (workInProgress !== null) {
    return RootInProgress;  // 未完成
  } else {
    return RootCompleted;   // 完成
  }
}

什么时候必须重新开始:
─────────────────────────────────────

1. 根节点变了
   workInProgressRoot !== root

2. 渲染的 lanes 变了
   workInProgressRootRenderLanes !== lanes

   例如：之前在渲染 TransitionLane，现在要渲染 SyncLane
   之前的工作无效，需要重新开始

3. 有更高优先级的更新插入
   新的更新可能影响之前的计算结果
   需要从头开始，包含新的更新
`;

// ============================================================
// Part 5: 伪代码级调度主循环
// ============================================================

/**
 * 📊 调度主循环伪代码
 */

const schedulingPseudoCode = `
📊 调度主循环伪代码

// ========================================
// 整体流程伪代码
// ========================================

function main调度循环() {
  while (true) {
    // 1. 获取最高优先级任务
    const task = Scheduler.peek(taskQueue);
    if (!task) {
      // 没有任务，等待
      break;
    }

    // 2. 检查时间片
    const startTime = getCurrentTime();

    // 3. 执行任务
    const callback = task.callback;  // = performConcurrentWorkOnRoot
    const continuation = callback(didTimeout);

    // 4. 检查结果
    if (typeof continuation === 'function') {
      // 任务未完成，保存 continuation
      task.callback = continuation;
    } else {
      // 任务完成，移出队列
      Scheduler.pop(taskQueue);
    }

    // 5. 检查是否需要让出
    if (shouldYield()) {
      break;  // 让出主线程，下个宏任务继续
    }
  }
}

// ========================================
// performConcurrentWorkOnRoot 伪代码
// ========================================

function performConcurrentWorkOnRoot(root, didTimeout) {
  // 1. 刷新 passive effects
  flushPassiveEffects();

  // 2. 获取要处理的 lanes
  const lanes = getNextLanes(root, workInProgressRootRenderLanes);

  // 3. 判断是否使用时间切片
  const shouldTimeSlice =
    !includesBlockingLane(lanes) &&
    !includesExpiredLane(lanes) &&
    !didTimeout;

  // 4. 渲染
  let exitStatus;
  if (shouldTimeSlice) {
    exitStatus = renderRootConcurrent(root, lanes);
  } else {
    exitStatus = renderRootSync(root, lanes);
  }

  // 5. 处理结果
  if (exitStatus === RootInProgress) {
    // 被打断，返回 continuation
    return performConcurrentWorkOnRoot.bind(null, root);
  }

  if (exitStatus === RootCompleted) {
    // 完成渲染，提交
    commitRoot(root);
  }

  // 6. 检查是否还有其他工作
  ensureRootIsScheduled(root);

  if (root.callbackNode === originalCallbackNode) {
    // 还是同一个任务，继续
    return performConcurrentWorkOnRoot.bind(null, root);
  }

  return null;
}

// ========================================
// renderRootConcurrent 伪代码
// ========================================

function renderRootConcurrent(root, lanes) {
  // 1. 检查是否可以复用之前的工作
  if (workInProgressRoot !== root ||
      workInProgressRootRenderLanes !== lanes) {
    // 不能复用，重新开始
    prepareFreshStack(root, lanes);
  }

  // 2. 工作循环（可中断）
  while (workInProgress !== null) {
    // 检查是否需要让出
    if (shouldYield()) {
      // 时间片用完，暂停
      return RootInProgress;
    }

    // 处理一个 Fiber
    performUnitOfWork(workInProgress);
  }

  // 3. 全部完成
  return RootCompleted;
}

// ========================================
// performUnitOfWork 伪代码
// ========================================

function performUnitOfWork(unitOfWork) {
  const current = unitOfWork.alternate;

  // 1. beginWork：处理当前 Fiber，返回子节点
  let next = beginWork(current, unitOfWork, renderLanes);

  if (next !== null) {
    // 有子节点，继续处理子节点
    workInProgress = next;
  } else {
    // 没有子节点，进入 completeWork
    completeUnitOfWork(unitOfWork);
  }
}
`;

// ============================================================
// Part 6: 关键数据结构
// ============================================================

/**
 * 📊 并发相关的关键数据结构
 */

const keyDataStructures = `
📊 并发相关的关键数据结构

┌─────────────────────────────────────────────────────────────────────────────┐
│                         FiberRoot 相关字段                                  │
│ 📁 packages/react-reconciler/src/ReactFiberRoot.new.js                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FiberRoot {                                                                │
│    // 待处理的 lanes（所有未完成更新的优先级集合）                           │
│    pendingLanes: Lanes;                                                     │
│                                                                             │
│    // 挂起的 lanes（因 Suspense 暂停）                                       │
│    suspendedLanes: Lanes;                                                   │
│                                                                             │
│    // 被 ping 的 lanes（Suspense resolve 后）                               │
│    pingedLanes: Lanes;                                                      │
│                                                                             │
│    // 过期的 lanes（需要同步执行）                                           │
│    expiredLanes: Lanes;                                                     │
│                                                                             │
│    // 当前 Scheduler 任务                                                   │
│    callbackNode: Task | null;                                               │
│                                                                             │
│    // 当前任务的优先级                                                       │
│    callbackPriority: Lane;                                                  │
│                                                                             │
│    // 每个 Lane 对应的过期时间                                               │
│    expirationTimes: Array<number>;                                          │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Update 相关字段                                     │
│ 📁 packages/react-reconciler/src/ReactFiberClassUpdateQueue.new.js         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Update {                                                                   │
│    // ⭐ 更新的优先级                                                        │
│    lane: Lane;                                                              │
│                                                                             │
│    // 更新类型（UpdateState, ReplaceState, ForceUpdate, CaptureUpdate）     │
│    tag: 0 | 1 | 2 | 3;                                                     │
│                                                                             │
│    // 更新的 payload（新状态或状态计算函数）                                 │
│    payload: any;                                                            │
│                                                                             │
│    // 更新完成后的回调                                                       │
│    callback: (() => mixed) | null;                                          │
│                                                                             │
│    // 链表指针                                                               │
│    next: Update<State> | null;                                              │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      ReactCurrentBatchConfig                                │
│ 📁 packages/react/src/ReactCurrentBatchConfig.js                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ReactCurrentBatchConfig = {                                                │
│    // ⭐ transition 标记                                                     │
│    // null = 普通更新                                                        │
│    // {} = 在 startTransition 内                                             │
│    transition: null | {},                                                   │
│  };                                                                         │
│                                                                             │
│  使用场景：                                                                  │
│  - startTransition 开始时设置 transition = {}                               │
│  - requestUpdateLane 检查 transition 是否为 null                            │
│  - 不为 null 时返回 TransitionLane                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 7: 面试题
// ============================================================

const interviewQuestions = `
💡 Part 3 面试题

Q1: 一个更新是如何被标记为"可中断"的？
A: 通过 Lane 判断。在 performConcurrentWorkOnRoot 中：
   shouldTimeSlice = !includesBlockingLane(lanes) && !includesExpiredLane(lanes)
   如果不包含阻塞型 Lane（Sync、Input）且没有过期，则可中断。
   startTransition 内的更新会分配 TransitionLane，属于可中断类型。

Q2: 更新被打断后如何恢复？
A:
   1. workInProgress 保留当前 Fiber 位置
   2. performConcurrentWorkOnRoot 返回 continuation（自身）
   3. Scheduler 保存到 task.callback
   4. 下次执行时，检查 workInProgressRoot === root 则继续
   5. 从 workInProgress 位置继续处理

Q3: TransitionLane 是如何分配的？
A: 通过 claimNextTransitionLane() 循环分配：
   - 有 16 个 TransitionLane（Lane1 ~ Lane16）
   - 每次调用返回当前 lane，然后左移一位
   - 超出范围后回到 Lane1
   - 这样不同 transition 有不同的 lane，可以独立追踪

Q4: 什么情况下必须重新开始渲染？
A:
   1. 根节点变了 (workInProgressRoot !== root)
   2. 渲染的 lanes 变了 (workInProgressRootRenderLanes !== lanes)
   3. 有更高优先级更新插入，需要包含新更新

Q5: ReactCurrentBatchConfig.transition 的作用？
A: 作为 startTransition 的标记。
   startTransition 开始时设置 transition = {}
   requestUpdateLane 检查这个值
   如果不为 null，返回 TransitionLane
   结束后恢复为之前的值
`;

export {
  concurrentUpdateFlow,
  laneAssignment,
  interruptionMechanism,
  interruptAndResume,
  schedulingPseudoCode,
  keyDataStructures,
  interviewQuestions,
};

