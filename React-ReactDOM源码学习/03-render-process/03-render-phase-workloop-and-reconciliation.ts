/**
 * ============================================================
 * 📚 Phase 3: 渲染流程 - Part 3: Render 阶段与 WorkLoop
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 * - packages/react-reconciler/src/ReactFiberBeginWork.new.js
 * - packages/react-reconciler/src/ReactFiberCompleteWork.new.js
 *
 * ⏱️ 预计时间：2-3 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: scheduleUpdateOnFiber 入口
// ============================================================

/**
 * 📊 scheduleUpdateOnFiber - 所有更新的入口
 */

const scheduleUpdateEntry = `
📊 scheduleUpdateOnFiber - 所有更新的入口

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js (Line 533)
═══════════════════════════════════════════════════════════════════════════════

无论是初次渲染还是状态更新，最终都会调用这个函数:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   触发更新的入口                                                            │
│   ─────────────                                                             │
│                                                                             │
│   root.render(<App />)                                                      │
│         │                                                                   │
│         ├──▶ updateContainer()                                              │
│         │                                                                   │
│   setState() / setCount()                                                   │
│         │                                                                   │
│         ├──▶ dispatchSetState() / dispatchReducerAction()                   │
│         │                                                                   │
│   forceUpdate()                                                             │
│         │                                                                   │
│         └──▶ enqueueForceUpdate()                                           │
│                    │                                                        │
│                    │                                                        │
│                    ▼                                                        │
│         ┌─────────────────────────┐                                         │
│         │  scheduleUpdateOnFiber  │  ⭐ 统一入口                            │
│         └────────────┬────────────┘                                         │
│                      │                                                      │
│                      ▼                                                      │
│                    开始调度                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


函数签名与核心逻辑:
═══════════════════════════════════════════════════════════════════════════════

export function scheduleUpdateOnFiber(
  root: FiberRoot,       // 应用根节点
  fiber: Fiber,          // 产生更新的 Fiber
  lane: Lane,            // 更新的优先级
  eventTime: number,     // 事件发生时间
) {
  // 1. 检查是否有嵌套更新（防止无限循环）
  checkForNestedUpdates();

  // 2. ⭐ 标记 root 有待处理的更新
  markRootUpdated(root, lane, eventTime);

  // 3. 检查是否在渲染阶段产生的更新（特殊处理）
  if ((executionContext & RenderContext) !== NoLanes && root === workInProgressRoot) {
    // 渲染阶段的更新，特殊标记
    workInProgressRootRenderPhaseUpdatedLanes =
      mergeLanes(workInProgressRootRenderPhaseUpdatedLanes, lane);
  } else {
    // 4. ⭐ 正常路径：确保 root 被调度
    ensureRootIsScheduled(root, eventTime);

    // 5. 特殊情况：同步 lane 且当前空闲，立即执行
    if (
      lane === SyncLane &&
      executionContext === NoContext &&
      (fiber.mode & ConcurrentMode) === NoMode
    ) {
      // 立即同步刷新
      flushSyncCallbacks();
    }
  }
}
`;

// ============================================================
// Part 2: ensureRootIsScheduled - 调度策略决策
// ============================================================

/**
 * 📊 ensureRootIsScheduled - 决定如何调度
 */

const ensureRootScheduled = `
📊 ensureRootIsScheduled - 调度策略决策

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js (Line 701)
═══════════════════════════════════════════════════════════════════════════════

这个函数决定 React 如何调度这次更新:

function ensureRootIsScheduled(root: FiberRoot, currentTime: number) {
  const existingCallbackNode = root.callbackNode;

  // 1. 标记过期的 lanes 需要同步执行
  markStarvedLanesAsExpired(root, currentTime);

  // 2. ⭐ 获取下一个要处理的 lanes
  const nextLanes = getNextLanes(
    root,
    root === workInProgressRoot ? workInProgressRootRenderLanes : NoLanes,
  );

  // 3. 没有待处理的工作，清理并返回
  if (nextLanes === NoLanes) {
    if (existingCallbackNode !== null) {
      cancelCallback(existingCallbackNode);
    }
    root.callbackNode = null;
    root.callbackPriority = NoLane;
    return;
  }

  // 4. 获取最高优先级的 lane
  const newCallbackPriority = getHighestPriorityLane(nextLanes);

  // 5. 检查是否可以复用现有的回调
  const existingCallbackPriority = root.callbackPriority;
  if (existingCallbackPriority === newCallbackPriority) {
    // 优先级相同，可以复用，不需要重新调度
    return;
  }

  // 6. 有更高优先级的更新，取消现有回调
  if (existingCallbackNode != null) {
    cancelCallback(existingCallbackNode);
  }

  // 7. ⭐ 根据优先级选择调度方式
  let newCallbackNode;
  if (newCallbackPriority === SyncLane) {
    // 同步优先级：用同步队列调度
    if (root.tag === LegacyRoot) {
      scheduleLegacySyncCallback(performSyncWorkOnRoot.bind(null, root));
    } else {
      scheduleSyncCallback(performSyncWorkOnRoot.bind(null, root));
    }
    // 使用微任务来执行同步任务
    if (supportsMicrotasks) {
      scheduleMicrotask(flushSyncCallbacks);
    } else {
      scheduleCallback(ImmediateSchedulerPriority, flushSyncCallbacks);
    }
    newCallbackNode = null;
  } else {
    // 并发优先级：用 Scheduler 调度
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
    }
    // ⭐ 调用 Scheduler 调度 performConcurrentWorkOnRoot
    newCallbackNode = scheduleCallback(
      schedulerPriorityLevel,
      performConcurrentWorkOnRoot.bind(null, root),
    );
  }

  root.callbackPriority = newCallbackPriority;
  root.callbackNode = newCallbackNode;
}


调度决策流程图:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ensureRootIsScheduled(root)                                               │
│         │                                                                   │
│         ▼                                                                   │
│   ┌───────────────────────────┐                                             │
│   │ getNextLanes(root)        │  获取下一个要处理的优先级                   │
│   └─────────────┬─────────────┘                                             │
│                 │                                                           │
│                 ▼                                                           │
│         nextLanes === NoLanes?                                              │
│         /              \\                                                    │
│       Yes               No                                                  │
│        │                 │                                                  │
│        ▼                 ▼                                                  │
│   ┌─────────┐   ┌──────────────────────────┐                               │
│   │ 清理返回 │   │ getHighestPriorityLane   │                               │
│   └─────────┘   └────────────┬─────────────┘                               │
│                              │                                              │
│                              ▼                                              │
│                   newCallbackPriority === SyncLane?                         │
│                   /                           \\                             │
│                 Yes                            No                           │
│                  │                              │                           │
│                  ▼                              ▼                           │
│   ┌─────────────────────────┐    ┌─────────────────────────────────┐       │
│   │  scheduleSyncCallback   │    │  scheduleCallback               │       │
│   │  (performSyncWorkOnRoot)│    │  (performConcurrentWorkOnRoot)  │       │
│   │                         │    │                                 │       │
│   │  使用微任务调度          │    │  使用 Scheduler 调度             │       │
│   └─────────────────────────┘    └─────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 3: performSyncWorkOnRoot vs performConcurrentWorkOnRoot
// ============================================================

/**
 * 📊 同步渲染 vs 并发渲染入口
 */

const performWorkOnRoot = `
📊 performSyncWorkOnRoot vs performConcurrentWorkOnRoot

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
═══════════════════════════════════════════════════════════════════════════════


performSyncWorkOnRoot (Line 1229)
─────────────────────────────────────────────────────────────────

function performSyncWorkOnRoot(root) {
  // 1. 刷新被动副作用
  flushPassiveEffects();

  // 2. 获取要处理的 lanes
  let lanes = getNextLanes(root, NoLanes);
  if (!includesSomeLane(lanes, SyncLane)) {
    ensureRootIsScheduled(root, now());
    return null;
  }

  // 3. ⭐ 同步渲染
  let exitStatus = renderRootSync(root, lanes);

  // 4. 错误处理（重试一次）
  if (root.tag !== LegacyRoot && exitStatus === RootErrored) {
    const errorRetryLanes = getLanesToRetrySynchronouslyOnError(root);
    if (errorRetryLanes !== NoLanes) {
      lanes = errorRetryLanes;
      exitStatus = recoverFromConcurrentError(root, errorRetryLanes);
    }
  }

  // 5. ⭐ 提交更新
  const finishedWork = root.finishedWork;
  if (finishedWork !== null) {
    root.finishedWork = null;
    commitRoot(root, ...);
  }

  // 6. 安排可能的下一次更新
  ensureRootIsScheduled(root, now());
  return null;
}


performConcurrentWorkOnRoot (Line 829)
─────────────────────────────────────────────────────────────────

function performConcurrentWorkOnRoot(root, didTimeout) {
  // 1. 刷新被动副作用
  const didFlushPassiveEffects = flushPassiveEffects();
  if (didFlushPassiveEffects && root.callbackNode !== originalCallbackNode) {
    return null;  // 任务被取消
  }

  // 2. 获取要处理的 lanes
  let lanes = getNextLanes(root, ...);
  if (lanes === NoLanes) {
    return null;
  }

  // 3. ⭐ 决定是否使用时间切片
  const shouldTimeSlice =
    !includesBlockingLane(root, lanes) &&
    !includesExpiredLane(root, lanes) &&
    (disableSchedulerTimeoutInWorkLoop || !didTimeout);

  // 4. ⭐ 渲染
  let exitStatus = shouldTimeSlice
    ? renderRootConcurrent(root, lanes)  // 并发渲染（可中断）
    : renderRootSync(root, lanes);       // 同步渲染（不可中断）

  // 5. 处理渲染结果
  if (exitStatus === RootInProgress) {
    // 渲染被中断，返回函数本身让 Scheduler 继续调度
    return performConcurrentWorkOnRoot.bind(null, root);
  }

  // 6. 渲染完成，准备提交
  if (exitStatus === RootCompleted) {
    const finishedWork = root.finishedWork;
    if (finishedWork !== null) {
      // ⭐ 进入 commit 阶段
      commitRoot(root, ...);
    }
  }

  ensureRootIsScheduled(root, now());
  return null;
}


两者对比:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────┬───────────────────────┬───────────────────────────────┐
│ 特性                │ performSyncWorkOnRoot │ performConcurrentWorkOnRoot   │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ 调用方式            │ 微任务 / Scheduler    │ Scheduler                     │
│ 渲染函数            │ renderRootSync        │ renderRootSync/Concurrent     │
│ 时间切片            │ ❌                    │ ✅（视情况）                  │
│ 可中断              │ ❌                    │ ✅（视情况）                  │
│ 返回值              │ null                  │ null / 自身（被中断时）       │
│ 触发场景            │ SyncLane、flushSync   │ 默认更新、Transition          │
└─────────────────────┴───────────────────────┴───────────────────────────────┘
`;

// ============================================================
// Part 4: WorkLoop - 工作循环
// ============================================================

/**
 * 📊 WorkLoop 工作循环
 */

const workLoop = `
📊 WorkLoop - 工作循环

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
═══════════════════════════════════════════════════════════════════════════════

WorkLoop 是 Render 阶段的核心，负责遍历 Fiber 树:


workLoopSync (Line 1741)
─────────────────────────────────────────────────────────────────

function workLoopSync() {
  // 不检查是否需要让出，一直执行到完成
  while (workInProgress !== null) {
    performUnitOfWork(workInProgress);
  }
}


workLoopConcurrent (Line 1829)
─────────────────────────────────────────────────────────────────

function workLoopConcurrent() {
  // 每处理一个 Fiber，检查是否需要让出主线程
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}


performUnitOfWork (Line 1836)
─────────────────────────────────────────────────────────────────

function performUnitOfWork(unitOfWork: Fiber): void {
  // 获取 current Fiber（旧的）
  const current = unitOfWork.alternate;

  // ⭐ beginWork: 处理当前 Fiber，返回子 Fiber
  let next = beginWork(current, unitOfWork, renderLanes);

  // 更新 memoizedProps
  unitOfWork.memoizedProps = unitOfWork.pendingProps;

  if (next === null) {
    // ⭐ 没有子节点了，进入 completeWork
    completeUnitOfWork(unitOfWork);
  } else {
    // 继续处理子节点
    workInProgress = next;
  }
}


completeUnitOfWork (Line 1873)
─────────────────────────────────────────────────────────────────

function completeUnitOfWork(unitOfWork: Fiber): void {
  let completedWork = unitOfWork;

  do {
    const current = completedWork.alternate;
    const returnFiber = completedWork.return;

    // ⭐ completeWork: 完成当前 Fiber 的工作
    let next = completeWork(current, completedWork, renderLanes);

    if (next !== null) {
      // 产生了新的工作（如 Suspense fallback）
      workInProgress = next;
      return;
    }

    // 处理兄弟节点
    const siblingFiber = completedWork.sibling;
    if (siblingFiber !== null) {
      workInProgress = siblingFiber;
      return;
    }

    // 回到父节点
    completedWork = returnFiber;
    workInProgress = completedWork;

  } while (completedWork !== null);

  // 整棵树处理完成
  if (workInProgressRootExitStatus === RootInProgress) {
    workInProgressRootExitStatus = RootCompleted;
  }
}


遍历过程图示:
═══════════════════════════════════════════════════════════════════════════════

组件树:
─────────
    <App>
      <Header />
      <Main>
        <List />
      </Main>
    </App>

遍历顺序:
─────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   workInProgress: App                                                       │
│         │                                                                   │
│         │ beginWork(App) → 返回 Header                                      │
│         ▼                                                                   │
│   workInProgress: Header                                                    │
│         │                                                                   │
│         │ beginWork(Header) → 返回 null                                     │
│         │ completeWork(Header)                                              │
│         │ sibling: Main                                                     │
│         ▼                                                                   │
│   workInProgress: Main                                                      │
│         │                                                                   │
│         │ beginWork(Main) → 返回 List                                       │
│         ▼                                                                   │
│   workInProgress: List                                                      │
│         │                                                                   │
│         │ beginWork(List) → 返回 null                                       │
│         │ completeWork(List)                                                │
│         │ sibling: null, return: Main                                       │
│         ▼                                                                   │
│   completeWork(Main)                                                        │
│         │ sibling: null, return: App                                        │
│         ▼                                                                   │
│   completeWork(App)                                                         │
│         │                                                                   │
│         ▼                                                                   │
│   workInProgress: null                                                      │
│   ════════════════════                                                      │
│   Render 阶段完成！                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

规则总结:
─────────
1. beginWork: 深度优先，向下遍历
   - 处理当前 Fiber
   - 返回第一个子 Fiber

2. completeWork: 向上回溯
   - 当前节点没有子节点时执行
   - 先看 sibling，有就处理兄弟
   - 没有就回到 return（父节点）
`;

// ============================================================
// Part 5: beginWork - 向下遍历
// ============================================================

/**
 * 📊 beginWork 详解
 */

const beginWorkDetail = `
📊 beginWork - 向下遍历，处理组件

源码位置: packages/react-reconciler/src/ReactFiberBeginWork.new.js
═══════════════════════════════════════════════════════════════════════════════

beginWork 的核心职责:
1. 根据 Fiber 的 tag 调用对应的处理函数
2. 执行组件的 render / 函数调用
3. 处理 Hooks
4. Diff 子元素，创建/复用子 Fiber
5. 标记副作用（flags）


函数签名:
─────────────────────────────────────────────────────────────────

function beginWork(
  current: Fiber | null,      // 旧 Fiber（可能为 null）
  workInProgress: Fiber,      // 新 Fiber（正在处理）
  renderLanes: Lanes,         // 渲染优先级
): Fiber | null {             // 返回子 Fiber 或 null

  // ⭐ 1. 尝试 bailout（跳过优化）
  if (current !== null) {
    const oldProps = current.memoizedProps;
    const newProps = workInProgress.pendingProps;

    if (
      oldProps === newProps &&
      !hasContextChanged() &&
      !includesSomeLane(renderLanes, updateLanes)
    ) {
      // props 没变，没有更新，可以跳过
      return bailoutOnAlreadyFinishedWork(current, workInProgress, renderLanes);
    }
  }

  // ⭐ 2. 根据 tag 处理不同类型的组件
  switch (workInProgress.tag) {
    case FunctionComponent:
      return updateFunctionComponent(current, workInProgress, ...);
    case ClassComponent:
      return updateClassComponent(current, workInProgress, ...);
    case HostRoot:
      return updateHostRoot(current, workInProgress, renderLanes);
    case HostComponent:
      return updateHostComponent(current, workInProgress, renderLanes);
    case HostText:
      return updateHostText(current, workInProgress);
    // ... 更多类型
  }
}


不同 tag 的处理逻辑:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ FunctionComponent (tag = 0)                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ function updateFunctionComponent(current, workInProgress, Component, ...) { │
│   // ⭐ 调用函数组件，执行 Hooks                                            │
│   let children = renderWithHooks(                                           │
│     current,                                                                │
│     workInProgress,                                                         │
│     Component,        // 组件函数                                           │
│     nextProps,        // props                                              │
│     context,                                                                │
│     renderLanes,                                                            │
│   );                                                                        │
│                                                                             │
│   // 协调子元素（Diff）                                                     │
│   reconcileChildren(current, workInProgress, children, renderLanes);        │
│                                                                             │
│   return workInProgress.child;                                              │
│ }                                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HostComponent (tag = 5) - 如 div、span                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ function updateHostComponent(current, workInProgress, renderLanes) {        │
│   const type = workInProgress.type;      // 'div', 'span' 等                │
│   const nextProps = workInProgress.pendingProps;                            │
│   const prevProps = current !== null ? current.memoizedProps : null;        │
│                                                                             │
│   let nextChildren = nextProps.children;                                    │
│                                                                             │
│   // 检查是否是纯文本子节点                                                 │
│   const isDirectTextChild = shouldSetTextContent(type, nextProps);          │
│   if (isDirectTextChild) {                                                  │
│     nextChildren = null;  // 文本内容在 completeWork 处理                   │
│   }                                                                         │
│                                                                             │
│   // 协调子元素                                                             │
│   reconcileChildren(current, workInProgress, nextChildren, renderLanes);    │
│                                                                             │
│   return workInProgress.child;                                              │
│ }                                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ HostRoot (tag = 3) - 根节点                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ function updateHostRoot(current, workInProgress, renderLanes) {             │
│   // 处理 updateQueue                                                       │
│   const nextState = processUpdateQueue(workInProgress, ...);                │
│                                                                             │
│   // ⭐ 从 memoizedState.element 获取子元素（<App />）                       │
│   const nextChildren = nextState.element;                                   │
│                                                                             │
│   // 协调子元素                                                             │
│   reconcileChildren(current, workInProgress, nextChildren, renderLanes);    │
│                                                                             │
│   return workInProgress.child;                                              │
│ }                                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 6: completeWork - 向上回溯
// ============================================================

/**
 * 📊 completeWork 详解
 */

const completeWorkDetail = `
📊 completeWork - 向上回溯，创建 DOM

源码位置: packages/react-reconciler/src/ReactFiberCompleteWork.new.js
═══════════════════════════════════════════════════════════════════════════════

completeWork 的核心职责:
1. 创建真实 DOM 节点（初次渲染）
2. 更新 DOM 属性（更新渲染）
3. 收集副作用标记（flags）
4. 冒泡副作用到父节点（subtreeFlags）


函数签名:
─────────────────────────────────────────────────────────────────

function completeWork(
  current: Fiber | null,
  workInProgress: Fiber,
  renderLanes: Lanes,
): Fiber | null {
  const newProps = workInProgress.pendingProps;

  switch (workInProgress.tag) {
    case HostComponent: {
      const type = workInProgress.type;  // 'div', 'span' 等

      if (current !== null && workInProgress.stateNode != null) {
        // ⭐ 更新流程
        updateHostComponent(current, workInProgress, type, newProps, ...);
      } else {
        // ⭐ 初次渲染，创建 DOM
        const instance = createInstance(type, newProps, ...);
        appendAllChildren(instance, workInProgress, ...);
        workInProgress.stateNode = instance;

        // 设置 DOM 属性
        finalizeInitialChildren(instance, type, newProps, ...);
      }

      // ⭐ 冒泡副作用
      bubbleProperties(workInProgress);
      return null;
    }

    case HostText: {
      const newText = newProps;
      if (current !== null && workInProgress.stateNode != null) {
        // 更新文本
        updateHostText(current, workInProgress, oldText, newText);
      } else {
        // 创建文本节点
        workInProgress.stateNode = createTextInstance(newText, ...);
      }
      bubbleProperties(workInProgress);
      return null;
    }

    // ... 其他类型
  }
}


DOM 创建流程:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   completeWork(HostComponent)                                               │
│         │                                                                   │
│         │ 首次渲染（stateNode === null）                                    │
│         ▼                                                                   │
│   ┌─────────────────────┐                                                   │
│   │   createInstance    │  创建 DOM 元素                                    │
│   │   document.createElement('div')                                         │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  appendAllChildren  │  将子 DOM 插入到当前 DOM                          │
│   │                     │  (此时还没插入到页面，只是组装 DOM 树)            │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────┐                                           │
│   │ workInProgress.stateNode =  │  将 DOM 保存到 Fiber                      │
│   │   instance                  │                                           │
│   └──────────┬──────────────────┘                                           │
│              │                                                              │
│              ▼                                                              │
│   ┌──────────────────────────┐                                              │
│   │ finalizeInitialChildren  │  设置 DOM 属性（className, style 等）        │
│   └──────────┬───────────────┘                                              │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │   bubbleProperties  │  收集子树的副作用标记                             │
│   └─────────────────────┘                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


bubbleProperties - 副作用冒泡:
═══════════════════════════════════════════════════════════════════════════════

function bubbleProperties(completedWork: Fiber) {
  let subtreeFlags = NoFlags;
  let child = completedWork.child;

  while (child !== null) {
    // ⭐ 收集子节点的 flags 和 subtreeFlags
    subtreeFlags |= child.subtreeFlags;
    subtreeFlags |= child.flags;
    child = child.sibling;
  }

  // 保存到当前节点的 subtreeFlags
  completedWork.subtreeFlags |= subtreeFlags;
}

作用:
- 在 commit 阶段，可以通过 subtreeFlags 快速判断子树是否有副作用
- 如果 subtreeFlags === NoFlags，可以跳过整棵子树
- 这是一种性能优化
`;

// ============================================================
// Part 7: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 3 面试要点

Q1: scheduleUpdateOnFiber 的作用是什么？
A: 是所有更新的统一入口。它会：
   1. 标记 root 有待处理的更新
   2. 调用 ensureRootIsScheduled 进行调度
   3. 某些情况下立即同步执行（SyncLane + 空闲状态）

Q2: React 如何决定使用同步渲染还是并发渲染？
A: 在 ensureRootIsScheduled 中决定：
   - SyncLane → performSyncWorkOnRoot → workLoopSync
   - 其他 Lane → performConcurrentWorkOnRoot → 可能使用 workLoopConcurrent
   并发模式还要看 shouldTimeSlice 条件（不包含 Blocking/Expired Lane）

Q3: workLoopSync 和 workLoopConcurrent 的区别？
A: - workLoopSync: while (workInProgress !== null) 一直执行
   - workLoopConcurrent: while (wIP !== null && !shouldYield()) 可被打断
   区别在于是否检查 shouldYield()，决定是否让出主线程

Q4: beginWork 和 completeWork 各做什么？
A: - beginWork: 向下遍历
     • 调用组件函数/render
     • 处理 Hooks
     • Diff 子元素，创建子 Fiber
   - completeWork: 向上回溯
     • 创建/更新 DOM 节点
     • 设置 DOM 属性
     • 收集副作用（bubbleProperties）

Q5: Fiber 树是如何遍历的？
A: 深度优先遍历：
   1. beginWork 处理当前节点，返回第一个子节点
   2. 重复步骤 1 直到没有子节点
   3. completeWork 完成当前节点
   4. 如果有 sibling，对 sibling 执行 beginWork
   5. 如果没有 sibling，回到 return 执行 completeWork
   6. 重复直到回到根节点

Q6: 为什么 completeWork 要做 bubbleProperties？
A: 将子树的副作用标记（flags）冒泡到父节点的 subtreeFlags。
   这样在 commit 阶段可以快速判断子树是否需要处理，
   如果 subtreeFlags === NoFlags，可以跳过整棵子树，提升性能。

Q7: DOM 是在哪个阶段创建的？
A: 在 Render 阶段的 completeWork 中创建。但此时只是创建和组装 DOM 树，
   还没有插入到页面。真正插入页面是在 Commit 阶段的 mutation 子阶段。
`;

export {
  scheduleUpdateEntry,
  ensureRootScheduled,
  performWorkOnRoot,
  workLoop,
  beginWorkDetail,
  completeWorkDetail,
  interviewPoints,
};

