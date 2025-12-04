/**
 * ============================================================
 * 📚 Phase 3: 渲染流程深度解析
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 React 渲染的两大阶段：Render 和 Commit
 * 2. 掌握 beginWork 和 completeWork 的工作
 * 3. 理解 Commit 阶段的三个子阶段
 * 4. 理解更新触发和调度机制
 *
 * 📁 核心源码位置：
 * - packages/react-reconciler/src/ReactFiberWorkLoop.new.js    # 工作循环
 * - packages/react-reconciler/src/ReactFiberBeginWork.new.js   # beginWork
 * - packages/react-reconciler/src/ReactFiberCompleteWork.new.js # completeWork
 * - packages/react-reconciler/src/ReactFiberCommitWork.new.js   # Commit 阶段
 *
 * ⏱️ 预计时间：6-8 小时
 * 🎯 面试权重：⭐⭐⭐⭐
 */

// ============================================================
// Part 1: 渲染流程总览
// ============================================================

/**
 * 📊 React 渲染流程全景图
 */

const renderFlowOverview = `
┌─────────────────────────────────────────────────────────────────────────┐
│                     React 渲染流程全景图                                │
│                                                                         │
│  触发更新                                                               │
│  ──────                                                                │
│  • ReactDOM.createRoot().render()  // 首次渲染                          │
│  • setState() / useState()         // 状态更新                          │
│  • forceUpdate()                   // 强制更新                          │
│  • props 变化                       // 父组件传递                        │
│                                                                         │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   调度阶段 (Schedule)                           │   │
│  │                                                                 │   │
│  │   scheduleUpdateOnFiber() → ensureRootIsScheduled()             │   │
│  │         │                                                       │   │
│  │         ▼                                                       │   │
│  │   Scheduler 调度任务（根据优先级）                               │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               Render 阶段 (可中断) ⭐                           │   │
│  │                                                                 │   │
│  │   performSyncWorkOnRoot() / performConcurrentWorkOnRoot()       │   │
│  │         │                                                       │   │
│  │         ▼                                                       │   │
│  │   renderRootSync() / renderRootConcurrent()                     │   │
│  │         │                                                       │   │
│  │         ▼                                                       │   │
│  │   workLoopSync() / workLoopConcurrent()                         │   │
│  │         │                                                       │   │
│  │    ┌────┴────────────────┐                                      │   │
│  │    │                     │                                      │   │
│  │    ▼                     ▼                                      │   │
│  │ beginWork()         completeWork()                              │   │
│  │  (递阶段)             (归阶段)                                   │   │
│  │  • 创建子 Fiber       • 创建 DOM                                │   │
│  │  • 标记副作用         • 收集副作用                               │   │
│  │  • Diff 算法                                                    │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               Commit 阶段 (不可中断) ⭐                         │   │
│  │                                                                 │   │
│  │   commitRoot()                                                  │   │
│  │         │                                                       │   │
│  │    ┌────┼────────────┬────────────────┐                         │   │
│  │    │    │            │                │                         │   │
│  │    ▼    ▼            ▼                ▼                         │   │
│  │ Before  Mutation    Layout        Passive                       │   │
│  │ Mutation (DOM 操作)  (DOM 后)       (异步)                       │   │
│  │                                                                 │   │
│  │ • getSnapshot  • 插入/更新  • ref 绑定   • useEffect             │   │
│  │ • Blur 事件    • 删除 DOM   • 生命周期                           │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: 更新触发与调度
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 *
 * 所有更新最终都会调用 scheduleUpdateOnFiber
 */

const scheduleUpdateFlow = `
📊 更新触发流程

1. setState 触发更新:
   this.setState({ count: 1 })
        │
        ▼
   enqueueSetState()
        │
        ▼
   enqueueUpdate(fiber, update, lane)  // 创建更新对象，加入队列
        │
        ▼
   scheduleUpdateOnFiber(fiber, lane)  // ⭐ 统一入口


2. useState 触发更新:
   setCount(1)
        │
        ▼
   dispatchSetState()
        │
        ▼
   scheduleUpdateOnFiber(fiber, lane)  // ⭐ 统一入口


3. scheduleUpdateOnFiber 内部:
   scheduleUpdateOnFiber(fiber, lane)
        │
        ├── markUpdateLaneFromFiberToRoot()  // 向上标记 lane
        │
        └── ensureRootIsScheduled(root)      // 确保根节点被调度
             │
             ├── scheduleSyncCallback()       // 同步任务
             │
             └── scheduleCallback()           // 异步任务（Scheduler）
`;

/**
 * 📊 ensureRootIsScheduled - 核心调度逻辑
 *
 * 📁 源码位置: ReactFiberWorkLoop.new.js (约 700 行)
 */

// 简化版 ensureRootIsScheduled
function ensureRootIsScheduledSimplified(root: FiberRoot) {
  // 1. 获取下一个要处理的 lanes
  const nextLanes = getNextLanes(root, NoLanes);

  if (nextLanes === NoLanes) {
    // 没有待处理的更新
    return;
  }

  // 2. 获取最高优先级
  const newCallbackPriority = getHighestPriorityLane(nextLanes);
  const existingCallbackPriority = root.callbackPriority;

  // 3. 如果已有相同优先级任务在调度，复用
  if (existingCallbackPriority === newCallbackPriority) {
    return;
  }

  // 4. 取消低优先级任务
  if (existingCallbackPriority !== NoLane) {
    cancelCallback(root.callbackNode);
  }

  // 5. 调度新任务
  let newCallbackNode;
  if (newCallbackPriority === SyncLane) {
    // 同步更新（Legacy 模式或 flushSync）
    scheduleSyncCallback(performSyncWorkOnRoot.bind(null, root));
    newCallbackNode = null;
  } else {
    // 并发更新
    const schedulerPriority = lanesToSchedulerPriority(newCallbackPriority);
    newCallbackNode = scheduleCallback(
      schedulerPriority,
      performConcurrentWorkOnRoot.bind(null, root)
    );
  }

  root.callbackPriority = newCallbackPriority;
  root.callbackNode = newCallbackNode;
}

// ============================================================
// Part 3: Render 阶段 - 工作循环
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 *
 * 工作循环是 Render 阶段的核心
 */

/**
 * 📊 同步模式 vs 并发模式
 */

const workLoopComparison = `
📊 workLoopSync vs workLoopConcurrent

// 同步模式：一次性完成，不检查时间
function workLoopSync() {
  while (workInProgress !== null) {
    performUnitOfWork(workInProgress);
  }
}

// 并发模式：每个工作单元后检查是否需要让出
function workLoopConcurrent() {
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}

shouldYield() 检查：
- 当前时间切片是否用完（默认 5ms）
- 是否有更高优先级任务插入

如果 shouldYield() 返回 true：
- 保存当前 workInProgress
- 让出主线程
- 等待 Scheduler 下次调度继续
`;

/**
 * 📊 performUnitOfWork - 执行单个工作单元
 */

// 简化版 performUnitOfWork
function performUnitOfWorkSimplified(unitOfWork: Fiber): void {
  const current = unitOfWork.alternate;

  // 1. "递"阶段：执行 beginWork
  let next = beginWork(current, unitOfWork, renderLanes);

  // 更新 memoizedProps
  unitOfWork.memoizedProps = unitOfWork.pendingProps;

  if (next === null) {
    // 2. 没有子节点，进入"归"阶段
    completeUnitOfWork(unitOfWork);
  } else {
    // 继续处理子节点
    workInProgress = next;
  }
}

// 简化版 completeUnitOfWork
function completeUnitOfWorkSimplified(unitOfWork: Fiber): void {
  let completedWork: Fiber | null = unitOfWork;

  do {
    const current = completedWork.alternate;
    const returnFiber = completedWork.return;

    // 执行 completeWork
    completeWork(current, completedWork, renderLanes);

    // 收集副作用到父节点
    if (returnFiber !== null) {
      // 冒泡 subtreeFlags
      returnFiber.subtreeFlags |= completedWork.subtreeFlags;
      returnFiber.subtreeFlags |= completedWork.flags;
    }

    // 检查兄弟节点
    const siblingFiber = completedWork.sibling;
    if (siblingFiber !== null) {
      workInProgress = siblingFiber;
      return;
    }

    // 返回父节点
    completedWork = returnFiber;
    workInProgress = completedWork;
  } while (completedWork !== null);
}

// ============================================================
// Part 4: beginWork - 递阶段
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberBeginWork.new.js
 *
 * beginWork 根据 Fiber 类型执行不同的处理逻辑
 */

const beginWorkExplanation = `
📊 beginWork 核心逻辑

beginWork(current, workInProgress, renderLanes)
    │
    ├── 检查是否可以复用（bailout 优化）
    │   if (current !== null) {
    │     // 更新阶段
    │     const oldProps = current.memoizedProps;
    │     const newProps = workInProgress.pendingProps;
    │     
    │     if (oldProps === newProps && !hasContextChanged()) {
    │       // props 没变，尝试 bailout
    │       return bailoutOnAlreadyFinishedWork();
    │     }
    │   }
    │
    └── 根据 tag 处理不同类型
        switch (workInProgress.tag) {
          case FunctionComponent:
            return updateFunctionComponent(...);
            
          case ClassComponent:
            return updateClassComponent(...);
            
          case HostRoot:
            return updateHostRoot(...);
            
          case HostComponent:
            return updateHostComponent(...);
            
          case HostText:
            return updateHostText(...);
            
          // ... 更多类型
        }
`;

/**
 * 📊 不同类型组件的 beginWork 处理
 */

// 1. 函数组件
const updateFunctionComponentExample = `
updateFunctionComponent(current, workInProgress, Component, nextProps, renderLanes)
    │
    ├── 设置 ReactCurrentDispatcher（Hooks）
    │
    ├── renderWithHooks()
    │   │
    │   ├── 设置当前渲染的 Fiber
    │   ├── 调用函数组件：nextChildren = Component(props)
    │   └── 重置 Hooks dispatcher
    │
    └── reconcileChildren(current, workInProgress, nextChildren)
        │
        └── Diff 算法，创建子 Fiber
`;

// 2. 类组件
const updateClassComponentExample = `
updateClassComponent(current, workInProgress, Component, nextProps, renderLanes)
    │
    ├── 实例化（首次渲染）或获取实例（更新）
    │   instance = workInProgress.stateNode;
    │
    ├── 处理生命周期
    │   ├── getDerivedStateFromProps
    │   └── shouldComponentUpdate
    │
    ├── 调用 render
    │   nextChildren = instance.render();
    │
    └── reconcileChildren(current, workInProgress, nextChildren)
`;

// 3. HostComponent (原生 DOM)
const updateHostComponentExample = `
updateHostComponent(current, workInProgress)
    │
    ├── 获取 props
    │   const nextProps = workInProgress.pendingProps;
    │
    ├── 处理 children
    │   const nextChildren = nextProps.children;
    │
    ├── 标记更新（如果需要）
    │   if (current !== null && current.stateNode !== null) {
    │     // 更新阶段，标记 Update flag
    │   }
    │
    └── reconcileChildren(current, workInProgress, nextChildren)
`;

/**
 * 📊 reconcileChildren - 核心 Diff
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactChildFiber.new.js
 */

const reconcileChildrenExplanation = `
reconcileChildren(current, workInProgress, nextChildren)
    │
    ├── 首次渲染
    │   mountChildFibers(workInProgress, null, nextChildren, renderLanes)
    │   // 不标记 Placement，因为整个应用都是新的
    │
    └── 更新阶段
        reconcileChildFibers(workInProgress, current.child, nextChildren, renderLanes)
        // 标记 Placement/Deletion 等副作用
        
reconcileChildFibers 内部（Diff 算法）:
    │
    ├── 单节点 Diff
    │   reconcileSingleElement()
    │   reconcileSingleTextNode()
    │
    └── 多节点 Diff
        reconcileChildrenArray()
        // 两轮遍历：
        // 第一轮：处理更新的节点
        // 第二轮：处理新增/移动的节点
`;

// ============================================================
// Part 5: completeWork - 归阶段
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberCompleteWork.new.js
 *
 * completeWork 主要做：
 * 1. 创建/更新 DOM 节点
 * 2. 收集副作用
 */

const completeWorkExplanation = `
📊 completeWork 核心逻辑

completeWork(current, workInProgress, renderLanes)
    │
    └── switch (workInProgress.tag) {
    
        case HostComponent:  // div, span 等
            │
            ├── 首次渲染（current === null）
            │   │
            │   ├── createInstance()    // 创建 DOM 元素
            │   ├── appendAllChildren() // 添加子 DOM
            │   ├── finalizeInitialChildren() // 设置属性
            │   └── workInProgress.stateNode = instance
            │
            └── 更新阶段
                │
                ├── prepareUpdate()  // 计算需要更新的属性
                │   // 返回 updatePayload: ['className', 'new-class', 'style', {...}]
                │
                └── workInProgress.updateQueue = updatePayload
                    workInProgress.flags |= Update  // 标记需要更新
        
        case HostText:  // 文本节点
            │
            ├── 首次渲染
            │   createTextInstance(newText)
            │
            └── 更新阶段
                if (oldText !== newText) {
                  workInProgress.flags |= Update
                }
        
        case FunctionComponent:
        case ClassComponent:
            // 这些类型通常只做一些清理工作
            bubbleProperties(workInProgress)  // 冒泡副作用
    }
`;

/**
 * 📊 appendAllChildren - 构建 DOM 树
 */

const appendAllChildrenExample = `
appendAllChildren(parent, workInProgress)

作用：将子 Fiber 对应的 DOM 节点添加到父 DOM

示例 Fiber 树:
    div (workInProgress)
     │
     ├── span
     │    └── "Hello"
     │
     └── p
          └── "World"

执行过程:
1. 遍历 workInProgress 的子 Fiber
2. 如果子 Fiber 是 HostComponent/HostText，将其 stateNode 添加到 parent
3. 如果子 Fiber 是组件类型，递归找到其子树中的 DOM 节点

结果:
<div>
  <span>Hello</span>
  <p>World</p>
</div>
`;

/**
 * 📊 bubbleProperties - 副作用冒泡
 */

const bubblePropertiesExample = `
bubbleProperties(completedWork)

作用：将子树的副作用冒泡到父节点

// 子节点副作用冒泡到 subtreeFlags
let subtreeFlags = NoFlags;
let child = completedWork.child;

while (child !== null) {
  subtreeFlags |= child.subtreeFlags;
  subtreeFlags |= child.flags;
  child = child.sibling;
}

completedWork.subtreeFlags |= subtreeFlags;

优化点：
- 如果 subtreeFlags === NoFlags，Commit 阶段可以跳过这个子树
- 避免遍历没有副作用的节点
`;

// ============================================================
// Part 6: Commit 阶段
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberCommitWork.new.js
 *
 * Commit 阶段不可中断，分为三个子阶段
 */

const commitPhaseOverview = `
📊 Commit 阶段全流程

commitRoot(root)
    │
    ├── 1️⃣ Before Mutation 阶段
    │   commitBeforeMutationEffects()
    │   │
    │   ├── 处理 DOM 失焦（blur）
    │   └── 调用 getSnapshotBeforeUpdate
    │
    ├── 2️⃣ Mutation 阶段 ⭐
    │   commitMutationEffects()
    │   │
    │   ├── ChildDeletion: 删除子节点
    │   │   └── 递归调用 componentWillUnmount
    │   │   └── 移除 DOM
    │   │
    │   ├── Placement: 插入 DOM
    │   │   └── appendChild / insertBefore
    │   │
    │   └── Update: 更新 DOM
    │       └── commitUpdate (更新属性)
    │       └── commitTextUpdate (更新文本)
    │
    ├── ⭐ 切换 current 指针
    │   root.current = finishedWork
    │
    ├── 3️⃣ Layout 阶段
    │   commitLayoutEffects()
    │   │
    │   ├── 调用生命周期
    │   │   └── componentDidMount
    │   │   └── componentDidUpdate
    │   │
    │   ├── 绑定 ref
    │   │   └── commitAttachRef
    │   │
    │   └── 调用 useLayoutEffect 回调
    │
    └── 4️⃣ 调度 Passive Effects（异步）
        scheduleCallback(flushPassiveEffects)
        │
        └── 执行 useEffect 回调
            ├── 先执行上次的销毁函数
            └── 再执行本次的创建函数
`;

/**
 * 📊 Mutation 阶段详解
 */

const mutationPhaseDetail = `
📊 commitMutationEffects 详解

commitMutationEffects(root, finishedWork)
    │
    └── commitMutationEffectsOnFiber(finishedWork)
        │
        ├── 递归处理子树
        │   recursivelyTraverseMutationEffects(root, fiber)
        │
        └── 处理当前节点
            commitReconciliationEffects(fiber)
            │
            ├── Placement（插入）
            │   commitPlacement(fiber)
            │   │
            │   ├── 找到最近的 Host 祖先
            │   │   let parent = fiber.return;
            │   │   while (parent !== null) {
            │   │     if (isHostParent(parent)) break;
            │   │     parent = parent.return;
            │   │   }
            │   │
            │   ├── 找到插入位置（兄弟 DOM）
            │   │   const before = getHostSibling(fiber);
            │   │
            │   └── 插入 DOM
            │       if (before) {
            │         insertBefore(parent, node, before);
            │       } else {
            │         appendChild(parent, node);
            │       }
            │
            ├── Update（更新）
            │   commitWork(fiber)
            │   │
            │   ├── HostComponent
            │   │   const updatePayload = fiber.updateQueue;
            │   │   commitUpdate(dom, updatePayload, type, oldProps, newProps);
            │   │
            │   └── HostText
            │       commitTextUpdate(textInstance, oldText, newText);
            │
            └── ChildDeletion（删除）
                commitDeletionEffects(fiber)
                │
                ├── 递归删除子树
                ├── 调用 componentWillUnmount
                ├── 解绑 ref
                └── removeChild(parent, child)
`;

/**
 * 📊 Layout 阶段详解
 */

const layoutPhaseDetail = `
📊 commitLayoutEffects 详解

commitLayoutEffects(finishedWork, root)
    │
    └── commitLayoutEffectOnFiber(root, current, fiber)
        │
        ├── FunctionComponent
        │   commitHookEffectListMount(HookLayout | HookHasEffect, fiber)
        │   // 执行 useLayoutEffect 的创建函数
        │
        ├── ClassComponent
        │   │
        │   ├── 首次渲染
        │   │   instance.componentDidMount()
        │   │
        │   └── 更新
        │       instance.componentDidUpdate(prevProps, prevState, snapshot)
        │
        └── HostRoot
            // 处理 ReactDOM.render 的回调
            commitUpdateQueue(fiber, updateQueue, instance)

// ref 绑定
commitAttachRef(fiber)
    │
    └── if (typeof ref === 'function') {
          ref(instanceToUse);
        } else {
          ref.current = instanceToUse;
        }
`;

/**
 * 📊 Passive Effects（useEffect）
 */

const passiveEffectsDetail = `
📊 useEffect 执行时机

commitRoot()
    │
    └── scheduleCallback(NormalPriority, flushPassiveEffects)
        // 异步调度，不阻塞渲染

flushPassiveEffects()
    │
    ├── 1. 执行销毁函数（上次 useEffect 返回的函数）
    │   commitPassiveUnmountEffects(root.current)
    │   │
    │   └── effect.destroy()
    │
    └── 2. 执行创建函数
        commitPassiveMountEffects(root, finishedWork)
        │
        └── effect.create()
             │
             └── 返回值作为下次的 destroy

执行顺序示例:
// 组件 A 中
useEffect(() => {
  console.log('A mount');
  return () => console.log('A unmount');
});

// 组件 B 中
useEffect(() => {
  console.log('B mount');
  return () => console.log('B unmount');
});

// 首次渲染后输出:
// A mount
// B mount

// 更新时输出:
// A unmount  ← 先执行所有销毁
// B unmount
// A mount    ← 再执行所有创建
// B mount
`;

// ============================================================
// Part 7: 完整渲染示例
// ============================================================

const fullRenderExample = `
📊 完整渲染示例

假设有以下组件:
function App() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <span>{count}</span>
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}

首次渲染流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. createRoot(container).render(<App />)
   │
   ▼
2. scheduleUpdateOnFiber(rootFiber, SyncLane)
   │
   ▼
3. Render 阶段 - beginWork
   │
   ├── beginWork(HostRoot)
   │   └── 创建 App Fiber
   │
   ├── beginWork(FunctionComponent - App)
   │   ├── renderWithHooks() → 调用 App()
   │   ├── useState 初始化，state = 0
   │   └── 创建 div/span/button Fiber
   │
   ├── beginWork(HostComponent - div)
   │   └── 创建 span/button Fiber
   │
   ├── beginWork(HostComponent - span)
   │   └── 创建 text Fiber "0"
   │
   ├── beginWork(HostText - "0")
   │   └── 无子节点
   │
   ...继续递归...
   │
   ▼
4. Render 阶段 - completeWork
   │
   ├── completeWork(HostText - "0")
   │   └── createTextInstance("0")
   │
   ├── completeWork(HostComponent - span)
   │   ├── createInstance("span")
   │   └── appendAllChildren()
   │
   ├── completeWork(HostComponent - button)
   │   ├── createInstance("button")
   │   └── 添加事件监听
   │
   ├── completeWork(HostComponent - div)
   │   ├── createInstance("div")
   │   └── appendAllChildren()
   │
   ├── completeWork(FunctionComponent - App)
   │   └── bubbleProperties()
   │
   └── completeWork(HostRoot)
       └── bubbleProperties()
   │
   ▼
5. Commit 阶段
   │
   ├── Before Mutation
   │   └── (无)
   │
   ├── Mutation
   │   └── appendChild(container, divDOM)  // 整个 DOM 树插入
   │
   ├── root.current = finishedWork
   │
   └── Layout
       └── (无生命周期)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

点击按钮更新流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. onClick → setCount(1)
   │
   ▼
2. dispatchSetState()
   │
   ├── 创建 Update { action: 1 }
   │
   └── scheduleUpdateOnFiber(appFiber, DefaultLane)
   │
   ▼
3. Render 阶段
   │
   ├── beginWork(HostRoot)
   │   └── bailout（props 没变）
   │
   ├── beginWork(App)
   │   ├── renderWithHooks()
   │   ├── 处理 Update，计算新 state = 1
   │   └── reconcileChildren() → 对比 children
   │
   ├── beginWork(div)
   │   └── reconcileChildren()
   │
   ├── beginWork(span)
   │   └── reconcileChildren()
   │
   └── beginWork(text)
       └── 标记 Update（文本 "0" → "1"）
   │
   ▼
4. completeWork (收集副作用)
   │
   ▼
5. Commit 阶段
   │
   └── Mutation
       └── commitTextUpdate(textNode, "0", "1")
           // textNode.nodeValue = "1"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`;

// ============================================================
// Part 8: 面试题
// ============================================================

const interviewQuestions = `
💡 Q1: React 渲染分为哪些阶段？
A: 三个阶段：
   1. Schedule（调度）：确定优先级，安排任务
   2. Render（可中断）：构建 Fiber 树，计算副作用
   3. Commit（不可中断）：执行 DOM 操作

💡 Q2: Render 阶段可以中断，Commit 阶段为什么不行？
A: - Render 阶段只是"计算"，不产生用户可见的副作用
   - Commit 阶段涉及 DOM 操作，如果中断会导致 UI 不一致
   - 例如：更新了一半的 DOM 就暂停，用户会看到"撕裂"的界面

💡 Q3: beginWork 和 completeWork 分别做什么？
A: beginWork（递阶段）：
   - 根据 Fiber 类型调用对应处理函数
   - 调用组件的 render 方法获取子节点
   - Diff 算法，创建子 Fiber
   - 标记副作用 flags

   completeWork（归阶段）：
   - 创建 DOM 节点（HostComponent）
   - 收集副作用（冒泡 subtreeFlags）
   - 计算更新内容（updatePayload）

💡 Q4: Commit 阶段三个子阶段分别做什么？
A: 1. Before Mutation：
      - getSnapshotBeforeUpdate
      - DOM 失焦处理
   
   2. Mutation：
      - 执行 DOM 操作（增删改）
      - 卸载组件（componentWillUnmount）
   
   3. Layout：
      - componentDidMount / componentDidUpdate
      - useLayoutEffect
      - ref 绑定

💡 Q5: useEffect 和 useLayoutEffect 执行时机有什么区别？
A: - useLayoutEffect 在 Layout 阶段同步执行
   - useEffect 在 Commit 完成后异步执行
   
   执行顺序：
   Mutation → current 切换 → Layout(useLayoutEffect) → 渲染
   → 下一帧 → useEffect

💡 Q6: current 指针什么时候切换？
A: 在 Mutation 阶段之后、Layout 阶段之前
   root.current = finishedWork;
   
   这样设计的原因：
   - Mutation 阶段操作的是旧 DOM
   - Layout 阶段（componentDidMount）需要访问新 DOM

💡 Q7: 什么是 bailout？
A: bailout 是 React 的优化机制：
   - 当组件 props/state 没变时跳过 Render
   - 检查条件：oldProps === newProps && !hasContextChanged
   - 如果 bailout，直接复用 current 的子树

💡 Q8: subtreeFlags 有什么作用？
A: subtreeFlags 是副作用冒泡机制：
   - 子节点的 flags 会冒泡到父节点的 subtreeFlags
   - Commit 阶段检查 subtreeFlags === NoFlags 可跳过整个子树
   - 避免遍历没有副作用的节点，提升性能
`;

// ============================================================
// Part 9: 类型定义和辅助函数
// ============================================================

interface Fiber {
  tag: number;
  alternate: Fiber | null;
  return: Fiber | null;
  child: Fiber | null;
  sibling: Fiber | null;
  memoizedProps: any;
  pendingProps: any;
  memoizedState: any;
  updateQueue: any;
  stateNode: any;
  flags: number;
  subtreeFlags: number;
}

interface FiberRoot {
  current: Fiber;
  finishedWork: Fiber | null;
  callbackNode: any;
  callbackPriority: number;
}

type Lanes = number;
const NoLanes = 0;
const SyncLane = 1;
let renderLanes: Lanes = 0;
let workInProgress: Fiber | null = null;

declare function getNextLanes(root: FiberRoot, wipLanes: Lanes): Lanes;
declare function getHighestPriorityLane(lanes: Lanes): number;
declare function cancelCallback(node: any): void;
declare function scheduleSyncCallback(callback: () => void): void;
declare function scheduleCallback(priority: number, callback: () => void): any;
declare function performSyncWorkOnRoot(root: FiberRoot): void;
declare function performConcurrentWorkOnRoot(root: FiberRoot): void;
declare function lanesToSchedulerPriority(lanes: Lanes): number;
declare function beginWork(current: Fiber | null, workInProgress: Fiber, renderLanes: Lanes): Fiber | null;
declare function completeWork(current: Fiber | null, workInProgress: Fiber, renderLanes: Lanes): void;
const NoLane = 0;

// ============================================================
// 学习检查清单
// ============================================================

/**
 * ✅ Phase 3 学习检查
 *
 * 流程理解：
 * - [ ] 能说出 React 渲染的三个阶段
 * - [ ] 理解 Render 阶段可中断的原因
 * - [ ] 理解 Commit 阶段不可中断的原因
 *
 * Render 阶段：
 * - [ ] 理解 workLoop 的工作方式
 * - [ ] 理解 beginWork 的作用（递）
 * - [ ] 理解 completeWork 的作用（归）
 * - [ ] 理解 reconcileChildren 的时机
 *
 * Commit 阶段：
 * - [ ] 能说出三个子阶段的名称
 * - [ ] 理解 current 指针切换的时机
 * - [ ] 理解 useEffect 和 useLayoutEffect 的区别
 *
 * 源码位置：
 * - [ ] 能找到 scheduleUpdateOnFiber
 * - [ ] 能找到 workLoopSync / workLoopConcurrent
 * - [ ] 能找到 commitRoot
 */

export {
  renderFlowOverview,
  scheduleUpdateFlow,
  workLoopComparison,
  beginWorkExplanation,
  updateFunctionComponentExample,
  updateClassComponentExample,
  updateHostComponentExample,
  reconcileChildrenExplanation,
  completeWorkExplanation,
  appendAllChildrenExample,
  bubblePropertiesExample,
  commitPhaseOverview,
  mutationPhaseDetail,
  layoutPhaseDetail,
  passiveEffectsDetail,
  fullRenderExample,
  interviewQuestions,
  ensureRootIsScheduledSimplified,
  performUnitOfWorkSimplified,
  completeUnitOfWorkSimplified,
};

