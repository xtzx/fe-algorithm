/**
 * ============================================================
 * 📚 Phase 6: Scheduler 调度机制 - Part 3: 真实案例与调度流程
 * ============================================================
 *
 * 本文件通过真实交互场景来讲解 Scheduler 的行为
 */

// ============================================================
// Part 1: 调度流程全景图
// ============================================================

/**
 * 📊 调度流程总览
 */

const schedulingFlowOverview = `
📊 调度流程全景图（从更新触发到执行完成）

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. 更新产生                                                                │
│  ────────────────────────────────────────────────────────────────────────── │
│     setState() / dispatch() / forceUpdate()                                 │
│                              │                                              │
│                              ▼                                              │
│                   scheduleUpdateOnFiber(root, fiber, lane)                  │
│                   📁 ReactFiberWorkLoop.new.js:533                          │
│                              │                                              │
│                              ▼                                              │
│                   markRootUpdated(root, lane)                               │
│                   root.pendingLanes |= lane                                 │
│                              │                                              │
│                              ▼                                              │
│  2. 调度决策                                                                │
│  ────────────────────────────────────────────────────────────────────────── │
│                   ensureRootIsScheduled(root)                               │
│                   📁 ReactFiberWorkLoop.new.js:696                          │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│         SyncLane?      并发 Lane?      无待处理?                            │
│              │               │               │                              │
│              ▼               ▼               ▼                              │
│        scheduleMicrotask  scheduleCallback   return                         │
│       (微任务同步执行)   (Scheduler调度)                                     │
│                              │                                              │
│                              ▼                                              │
│  3. Scheduler 接收任务                                                      │
│  ────────────────────────────────────────────────────────────────────────── │
│                   scheduleCallback(priority, callback)                      │
│                   📁 Scheduler.js:308                                       │
│                              │                                              │
│                              ▼                                              │
│                   创建 Task，计算 expirationTime                            │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼                                              │
│         有 delay?       无 delay?                                           │
│              │               │                                              │
│              ▼               ▼                                              │
│        push(timerQueue)  push(taskQueue)                                    │
│              │               │                                              │
│              │               ▼                                              │
│              │      requestHostCallback(flushWork)                          │
│              │               │                                              │
│              │               ▼                                              │
│              │      schedulePerformWorkUntilDeadline()                      │
│              │      (MessageChannel.postMessage)                            │
│              │                                                              │
│              ▼                                                              │
│        requestHostTimeout(handleTimeout, delay)                             │
│        (setTimeout)                                                         │
│                              │                                              │
│                              ▼                                              │
│  4. 浏览器空闲，执行任务                                                    │
│  ────────────────────────────────────────────────────────────────────────── │
│                   performWorkUntilDeadline()                                │
│                   📁 Scheduler.js:515                                       │
│                              │                                              │
│                              ▼                                              │
│                   flushWork(hasTimeRemaining, initialTime)                  │
│                   📁 Scheduler.js:147                                       │
│                              │                                              │
│                              ▼                                              │
│                   workLoop(hasTimeRemaining, initialTime)                   │
│                   📁 Scheduler.js:189                                       │
│                              │                                              │
│                              ▼                                              │
│  5. 工作循环                                                                │
│  ────────────────────────────────────────────────────────────────────────── │
│     while (currentTask !== null) {                                          │
│                              │                                              │
│         ┌────────────────────┴────────────────────┐                         │
│         ▼                                         ▼                         │
│     未过期 && shouldYieldToHost()?            已过期或有时间                 │
│         │                                         │                         │
│         ▼                                         ▼                         │
│       break (让出)                         执行 callback                    │
│         │                                         │                         │
│         │                     ┌───────────────────┴───────────────┐         │
│         │                     ▼                                   ▼         │
│         │              返回函数?                            返回 null?      │
│         │                     │                                   │         │
│         │                     ▼                                   ▼         │
│         │          任务未完成，继续调度                    任务完成，pop()   │
│         │          currentTask.callback = continuation                      │
│         │                                                                   │
│         └─────────────────────┬─────────────────────────────────────────┘   │
│                               ▼                                             │
│  6. 任务被打断/恢复                                                         │
│  ────────────────────────────────────────────────────────────────────────── │
│     return hasMoreWork (true/false)                                         │
│                               │                                             │
│               ┌───────────────┴───────────────┐                             │
│               ▼                               ▼                             │
│          hasMoreWork=true                hasMoreWork=false                  │
│               │                               │                             │
│               ▼                               ▼                             │
│     schedulePerformWorkUntilDeadline()    完成                              │
│     (再次调度，继续执行)                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: 工作循环伪代码
// ============================================================

/**
 * 📊 Scheduler workLoop 伪代码
 */

const workLoopPseudoCode = `
📊 workLoop 伪代码

function workLoop(hasTimeRemaining, initialTime) {
  let currentTime = initialTime;

  // 检查延迟任务是否到期
  advanceTimers(currentTime);

  // 获取最高优先级任务
  currentTask = peek(taskQueue);

  while (currentTask !== null) {
    // ⭐ 关键判断：是否让出主线程
    if (
      currentTask.expirationTime > currentTime &&  // 任务未过期
      (!hasTimeRemaining || shouldYieldToHost())   // 但时间片用完了
    ) {
      // 让出主线程，下次继续
      break;
    }

    // 获取任务回调
    const callback = currentTask.callback;

    if (typeof callback === 'function') {
      currentTask.callback = null;

      // 判断是否已超时
      const didUserCallbackTimeout = currentTask.expirationTime <= currentTime;

      // ⭐ 执行任务
      const continuationCallback = callback(didUserCallbackTimeout);

      currentTime = getCurrentTime();

      if (typeof continuationCallback === 'function') {
        // 任务返回函数 → 任务未完成，下次继续
        currentTask.callback = continuationCallback;
      } else {
        // 任务完成，移出队列
        if (currentTask === peek(taskQueue)) {
          pop(taskQueue);
        }
      }

      // 再次检查延迟任务
      advanceTimers(currentTime);
    } else {
      // callback 为 null，任务被取消
      pop(taskQueue);
    }

    // 获取下一个任务
    currentTask = peek(taskQueue);
  }

  // 返回是否还有任务
  if (currentTask !== null) {
    return true;   // 还有任务，需要再次调度
  } else {
    // 检查是否有延迟任务
    const firstTimer = peek(timerQueue);
    if (firstTimer !== null) {
      requestHostTimeout(handleTimeout, firstTimer.startTime - currentTime);
    }
    return false;  // 没有任务了
  }
}
`;

/**
 * 📊 React 渲染循环伪代码
 */

const reactWorkLoopPseudoCode = `
📊 React workLoopConcurrent 伪代码

// 📁 ReactFiberWorkLoop.new.js:1829
function workLoopConcurrent() {
  // 当有工作要做 且 不需要让出时，继续执行
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}

// shouldYield 实际上就是 Scheduler 的 shouldYieldToHost
import { shouldYield } from 'scheduler';

// performUnitOfWork: 处理单个 Fiber
function performUnitOfWork(unitOfWork) {
  const current = unitOfWork.alternate;

  // beginWork: 递阶段
  let next = beginWork(current, unitOfWork, renderLanes);

  if (next === null) {
    // 没有子节点，进入归阶段
    completeUnitOfWork(unitOfWork);
  } else {
    // 有子节点，继续处理
    workInProgress = next;
  }
}

// 关键：每处理一个 Fiber，就检查 shouldYield()
// 这就是 React 实现可中断渲染的核心
`;

// ============================================================
// Part 3: 真实案例 A - 输入框 + 重列表渲染
// ============================================================

/**
 * 📊 场景 A：用户输入时有大列表需要更新
 *
 * 这个场景展示了 startTransition 如何利用 Scheduler 实现优先级调度
 */

const caseA_InputWithHeavyList = `
📊 场景 A：输入框 + 大列表渲染

┌─────────────────────────────────────────────────────────────────────────────┐
│  用户场景                                                                   │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  搜索框: [________________]                                            │ │
│  │                                                                        │ │
│  │  搜索结果列表（10000 项）:                                              │ │
│  │  ├── Item 1                                                            │ │
│  │  ├── Item 2                                                            │ │
│  │  ├── ...                                                               │ │
│  │  └── Item 10000                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  期望行为：                                                                 │
│  - 输入框响应要即时（<50ms）                                                │
│  - 列表可以稍后更新                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
`;

/**
 * 📊 组件代码示例
 */

const caseA_Code = `
📊 组件代码

import { useState, useTransition } from 'react';

function SearchableList() {
  const [inputValue, setInputValue] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isPending, startTransition] = useTransition();

  const handleChange = (e) => {
    const value = e.target.value;

    // 1. 高优先级：输入框立即更新
    setInputValue(value);

    // 2. 低优先级：列表可以稍后更新
    startTransition(() => {
      setSearchQuery(value);
    });
  };

  return (
    <div>
      <input value={inputValue} onChange={handleChange} />
      {isPending && <span>Loading...</span>}
      <HeavyList query={searchQuery} />
    </div>
  );
}

function HeavyList({ query }) {
  // 假设这里要渲染 10000 个项
  const items = generateItems(10000, query);
  return (
    <ul>
      {items.map(item => <li key={item.id}>{item.text}</li>)}
    </ul>
  );
}
`;

/**
 * 📊 场景 A 调度时间线
 */

const caseA_Timeline = `
📊 场景 A 调度时间线

时间轴 (ms)
0         5         10        15        20        25        30
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼───→

T=0ms: 用户输入字符 'a'
       │
       ├── dispatchSetState(inputValue, 'a')
       │   └── lane = SyncLane (最高优先级)
       │
       └── startTransition(() => setSearchQuery('a'))
           └── lane = TransitionLane (低优先级)

       scheduleUpdateOnFiber() 被调用两次

T=0.1ms: ensureRootIsScheduled()
         │
         ├── 检测到 SyncLane 更新
         │   └── scheduleMicrotask(flushSyncCallbacks)
         │       (同步更新，不走 Scheduler)
         │
         └── 检测到 TransitionLane 更新
             └── scheduleCallback(NormalPriority, performConcurrentWorkOnRoot)
                 (异步更新，走 Scheduler)

T=0.2ms: 微任务执行，处理 SyncLane
         │
         └── renderRootSync() → commitRoot()
             └── 输入框立即显示 'a' ✅

T=0.3ms: 浏览器绘制输入框

T=1ms:   MessageChannel 回调，开始处理 TransitionLane
         │
         └── performConcurrentWorkOnRoot()
             └── workLoopConcurrent()
                 │
                 ├── 处理 HeavyList (10000 项)
                 │
                 └── 每处理一个 Fiber:
                     if (shouldYield()) break;

T=5ms:   时间片用完，shouldYieldToHost() = true
         │
         └── workLoopConcurrent() 跳出
             └── 返回 continuationCallback
                 └── 任务未完成，重新调度

T=5.1ms: 让出主线程，浏览器处理其他事件

T=6ms:   再次进入 workLoopConcurrent()
         └── 继续处理剩余 Fiber

... 重复直到完成 ...

T=50ms:  HeavyList 渲染完成
         └── commitRoot()
             └── 列表显示更新结果 ✅

关键点：
1. 输入框在 <1ms 内就响应了
2. 列表分多个时间片完成，不阻塞用户交互
3. startTransition 把列表更新标记为低优先级
`;

/**
 * 📊 场景 A 函数调用链
 */

const caseA_CallStack = `
📊 场景 A 关键函数调用

1. 用户输入触发
   onChange()
   └── setInputValue('a')
       └── dispatchSetState()
           📁 ReactFiberHooks.new.js:2476

2. 调度更新
   └── scheduleUpdateOnFiber(root, fiber, SyncLane)
       📁 ReactFiberWorkLoop.new.js:533
       └── markRootUpdated(root, SyncLane)
       └── ensureRootIsScheduled(root)
           📁 ReactFiberWorkLoop.new.js:696

3. 同步更新路径 (SyncLane)
   └── scheduleSyncCallback(performSyncWorkOnRoot)
       📁 ReactFiberWorkLoop.new.js:768
   └── scheduleMicrotask(flushSyncCallbacks)
       📁 ReactFiberWorkLoop.new.js:778
   └── flushSyncCallbacks()
       └── performSyncWorkOnRoot()
           └── renderRootSync()
               └── workLoopSync()
           └── commitRoot()

4. Transition 更新路径 (TransitionLane)
   └── scheduleCallback(NormalPriority, performConcurrentWorkOnRoot)
       📁 Scheduler.js:308
       └── push(taskQueue, task)
       └── requestHostCallback(flushWork)
           └── schedulePerformWorkUntilDeadline()
               (MessageChannel.postMessage)

5. Scheduler 执行
   └── performWorkUntilDeadline()
       📁 Scheduler.js:515
       └── flushWork()
           📁 Scheduler.js:147
           └── workLoop()
               📁 Scheduler.js:189
               └── callback(didTimeout)  // = performConcurrentWorkOnRoot
                   📁 ReactFiberWorkLoop.new.js:829

6. React 渲染
   └── performConcurrentWorkOnRoot()
       └── renderRootConcurrent()
           📁 ReactFiberWorkLoop.new.js:1748
           └── workLoopConcurrent()
               📁 ReactFiberWorkLoop.new.js:1829
               └── while (workInProgress && !shouldYield()) {
                       performUnitOfWork(workInProgress)
                   }

7. 被打断
   └── shouldYield() === true
       └── workLoopConcurrent() 退出
       └── performConcurrentWorkOnRoot() 返回 continuation
       └── workLoop() 保存 continuation 到 task.callback
       └── return true (hasMoreWork)
       └── schedulePerformWorkUntilDeadline() (再次调度)

8. 恢复执行
   └── 重复步骤 5-7 直到完成
`;

// ============================================================
// Part 4: 真实案例 B - 高频滚动
// ============================================================

/**
 * 📊 场景 B：高频滚动触发更新
 */

const caseB_HighFrequencyScroll = `
📊 场景 B：高频滚动

┌─────────────────────────────────────────────────────────────────────────────┐
│  用户场景                                                                   │
│                                                                             │
│  用户快速滚动虚拟列表，每次滚动都触发 setState 更新可视区域                   │
│                                                                             │
│  问题：                                                                     │
│  - 滚动事件每 16ms 触发一次（60fps）                                         │
│  - 每次都触发更新，可能导致任务堆积                                           │
│  - 旧的更新可能还没完成，新的更新就来了                                       │
│                                                                             │
│  期望行为：                                                                 │
│  - 滚动要流畅                                                               │
│  - 丢弃过时的更新                                                           │
│  - 只渲染最新状态                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
`;

const caseB_Code = `
📊 组件代码

function VirtualList() {
  const [scrollTop, setScrollTop] = useState(0);

  const handleScroll = (e) => {
    // 每次滚动都触发更新
    setScrollTop(e.target.scrollTop);
  };

  // 计算可视区域内的项
  const visibleItems = calculateVisibleItems(scrollTop);

  return (
    <div onScroll={handleScroll} style={{ height: 500, overflow: 'auto' }}>
      <div style={{ height: totalHeight }}>
        {visibleItems.map(item => (
          <div key={item.id} style={{ position: 'absolute', top: item.top }}>
            {item.content}
          </div>
        ))}
      </div>
    </div>
  );
}
`;

const caseB_Timeline = `
📊 场景 B 调度时间线

时间轴 (ms)
0    16    32    48    64    80    96
├────┼─────┼─────┼─────┼─────┼─────┼────→

T=0ms:   滚动事件 #1，scrollTop=100
         └── scheduleUpdateOnFiber(DefaultLane)
         └── ensureRootIsScheduled()
             └── scheduleCallback(NormalPriority, performConcurrentWork)
             └── root.callbackNode = task1

T=5ms:   开始渲染 task1...

T=16ms:  滚动事件 #2，scrollTop=200
         └── scheduleUpdateOnFiber(DefaultLane)
         └── ensureRootIsScheduled()
             │
             ├── 检查 existingCallbackPriority === newCallbackPriority?
             │   └── true，优先级相同
             │
             └── return; // 复用现有任务，不重新调度！⭐

         // 但是 lane 已经被标记到 root.pendingLanes

T=20ms:  task1 继续渲染...
         └── 检查 lanes 发现有新的更新
         └── 使用最新的 scrollTop=200 渲染

T=32ms:  滚动事件 #3，scrollTop=300
         └── 同上，复用现有任务

T=35ms:  task1 完成渲染 scrollTop=200
         └── commitRoot()
         │
         └── ensureRootIsScheduled()
             └── 发现还有 pendingLanes
             └── scheduleCallback(NormalPriority, performConcurrentWork)
             └── 开始渲染 scrollTop=300

... 依此类推 ...

关键优化：
1. 相同优先级的更新复用同一个 Scheduler 任务
2. 不会创建大量任务导致堆积
3. 渲染时使用最新状态，自动"跳过"中间状态
`;

/**
 * 📊 场景 B 任务复用机制
 */

const caseB_TaskReuse = `
📊 任务复用机制

📁 ReactFiberWorkLoop.new.js:696 - ensureRootIsScheduled

function ensureRootIsScheduled(root, currentTime) {
  const existingCallbackNode = root.callbackNode;

  // 计算下一个要处理的 lanes
  const nextLanes = getNextLanes(root, ...);

  if (nextLanes === NoLanes) {
    // 没有任务了
    if (existingCallbackNode !== null) {
      cancelCallback(existingCallbackNode);
    }
    return;
  }

  const newCallbackPriority = getHighestPriorityLane(nextLanes);
  const existingCallbackPriority = root.callbackPriority;

  // ⭐ 关键：优先级相同，复用任务
  if (existingCallbackPriority === newCallbackPriority) {
    // 不需要重新调度，复用现有任务
    return;
  }

  // 优先级不同，取消旧任务，创建新任务
  if (existingCallbackNode !== null) {
    cancelCallback(existingCallbackNode);
  }

  // 创建新任务...
  let newCallbackNode = scheduleCallback(priority, callback);
  root.callbackNode = newCallbackNode;
  root.callbackPriority = newCallbackPriority;
}

这就是为什么高频更新不会导致 Scheduler 任务堆积：
- 相同优先级的更新会合并到同一个任务中
- 只有优先级变化时才会重新调度
`;

// ============================================================
// Part 5: 打断与恢复机制
// ============================================================

/**
 * 📊 任务打断与恢复
 */

const interruptAndResume = `
📊 任务打断与恢复机制

┌─────────────────────────────────────────────────────────────────────────────┐
│  打断发生的条件                                                             │
│                                                                             │
│  1. 时间片用完                                                              │
│     shouldYieldToHost() === true                                            │
│     (执行时间 > 5ms)                                                        │
│                                                                             │
│  2. 有更高优先级任务                                                        │
│     用户点击触发 SyncLane 更新                                               │
│     当前正在渲染 DefaultLane                                                │
│                                                                             │
│  3. 有用户输入待处理                                                        │
│     isInputPending() === true                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  打断时保存的状态                                                           │
│                                                                             │
│  Scheduler 层面:                                                            │
│    - task.callback = continuationCallback                                   │
│    - task 保留在 taskQueue 中                                               │
│                                                                             │
│  React 层面:                                                                │
│    - workInProgress: 当前处理到的 Fiber                                     │
│    - workInProgressRoot: 当前根节点                                         │
│    - workInProgressRootRenderLanes: 当前渲染的 lanes                        │
│    - 这些变量是模块级的，不会丢失                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  恢复执行                                                                   │
│                                                                             │
│  1. Scheduler 再次调用 task.callback                                        │
│     └── performConcurrentWorkOnRoot(root, didTimeout)                       │
│                                                                             │
│  2. React 检查是否可以继续                                                   │
│     if (workInProgressRoot === root &&                                      │
│         workInProgressRootRenderLanes === lanes) {                          │
│       // 可以继续，不需要重新开始                                            │
│     } else {                                                                │
│       prepareFreshStack(root, lanes);  // 需要重新开始                       │
│     }                                                                       │
│                                                                             │
│  3. 继续 workLoopConcurrent()                                               │
│     从 workInProgress 继续处理                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

/**
 * 📊 continuationCallback 机制
 */

const continuationCallback = `
📊 continuationCallback 机制

performConcurrentWorkOnRoot 返回值的含义:

1. 返回 null
   → 任务完成，可以移出队列

2. 返回自身 (performConcurrentWorkOnRoot.bind(null, root))
   → 任务未完成，需要继续调度

📁 ReactFiberWorkLoop.new.js:829

function performConcurrentWorkOnRoot(root, didTimeout) {
  // ... 渲染逻辑 ...

  // 检查渲染结果
  if (workInProgress !== null) {
    // ⭐ 渲染未完成（被打断了）
    // 返回自身作为 continuation
    return performConcurrentWorkOnRoot.bind(null, root);
  }

  // 渲染完成
  // ... commit 逻辑 ...

  // 检查是否还有其他待处理的 lanes
  ensureRootIsScheduled(root, now());

  if (root.callbackNode === originalCallbackNode) {
    // 如果任务没变，说明还需要继续处理其他 lanes
    return performConcurrentWorkOnRoot.bind(null, root);
  }

  // 完全完成
  return null;
}

Scheduler 中的处理:
📁 Scheduler.js:189

function workLoop() {
  // ...
  const continuationCallback = callback(didUserCallbackTimeout);

  if (typeof continuationCallback === 'function') {
    // 任务未完成，保存 continuation
    currentTask.callback = continuationCallback;
    // 不从队列移除，下次继续执行
  } else {
    // 任务完成，移出队列
    pop(taskQueue);
  }
  // ...
}
`;

export {
  schedulingFlowOverview,
  workLoopPseudoCode,
  reactWorkLoopPseudoCode,
  caseA_InputWithHeavyList,
  caseA_Code,
  caseA_Timeline,
  caseA_CallStack,
  caseB_HighFrequencyScroll,
  caseB_Code,
  caseB_Timeline,
  caseB_TaskReuse,
  interruptAndResume,
  continuationCallback,
};

