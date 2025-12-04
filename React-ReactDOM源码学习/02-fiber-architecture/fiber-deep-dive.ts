/**
 * ============================================================
 * 📚 Phase 2: Fiber 架构深度解析
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解为什么需要 Fiber 架构
 * 2. 掌握 Fiber 节点的数据结构
 * 3. 理解 Fiber 树的双缓冲机制
 * 4. 理解 FiberRoot 和 HostRootFiber 的关系
 *
 * 📁 核心源码位置：
 * - packages/react-reconciler/src/ReactFiber.new.js        # Fiber 节点
 * - packages/react-reconciler/src/ReactFiberRoot.new.js    # FiberRoot
 * - packages/react-reconciler/src/ReactWorkTags.js         # Fiber 类型
 * - packages/react-reconciler/src/ReactFiberFlags.js       # 副作用标记
 * - packages/react-reconciler/src/ReactInternalTypes.js    # 类型定义
 *
 * ⏱️ 预计时间：6-8 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐（最高！）
 */

// ============================================================
// Part 1: 为什么需要 Fiber？
// ============================================================

/**
 * 📊 React 15 的问题：Stack Reconciler
 *
 * React 15 使用递归方式遍历虚拟 DOM 树，一旦开始就无法中断
 *
 * ```
 * 问题场景：
 *
 * 假设有一个包含 10000 个节点的树需要更新
 *
 * Stack Reconciler（React 15）:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │  开始更新 ──────────────────────────────────────────► 完成  │
 * │     │                                                       │
 * │     │◄──────────── 100ms 阻塞 ────────────────────────►│    │
 * │     │                                                       │
 * │  用户点击 ──────────── 等待 100ms 才能响应 ───────────►     │
 * │                                                             │
 * │  ⚠️ 问题：JS 单线程，长时间占用导致页面卡顿                  │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Fiber Reconciler（React 16+）:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │  工作单元1 → 暂停 → 工作单元2 → 暂停 → 工作单元3 → ...      │
 * │     5ms      │       5ms      │       5ms                   │
 * │              ▼                ▼                             │
 * │         用户点击          动画帧                            │
 * │        （立即响应）      （保持流畅）                        │
 * │                                                             │
 * │  ✅ 解决：可中断的渲染，优先响应用户操作                     │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 */

const whyFiberExplanation = `
📊 Stack vs Fiber 对比

┌─────────────────┬──────────────────┬──────────────────────┐
│     特性        │  Stack (React 15) │   Fiber (React 16+)  │
├─────────────────┼──────────────────┼──────────────────────┤
│ 渲染方式        │ 递归（同步）      │ 循环（可中断）        │
│ 中断能力        │ ❌ 不可中断       │ ✅ 可中断            │
│ 优先级调度      │ ❌ 无             │ ✅ 有               │
│ 时间切片        │ ❌ 无             │ ✅ 有               │
│ 并发模式        │ ❌ 不支持         │ ✅ 支持             │
│ 数据结构        │ 虚拟 DOM 树       │ Fiber 链表树         │
│ 复杂度          │ 简单             │ 复杂                 │
└─────────────────┴──────────────────┴──────────────────────┘

为什么递归不能中断？
- 递归调用使用函数调用栈
- 调用栈是隐式的，无法保存中间状态
- 一旦中断，调用栈信息丢失

Fiber 如何实现中断？
- 使用链表结构，每个节点是一个"工作单元"
- 链表遍历可以随时保存当前位置
- 中断后可以从保存的位置继续
`;

// ============================================================
// Part 2: Fiber 节点数据结构（核心！）
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiber.new.js
 *             packages/react-reconciler/src/ReactInternalTypes.js
 *
 * Fiber 节点就是一个普通的 JavaScript 对象
 * 但它包含了组件的所有信息
 */

// Fiber 节点完整结构
interface Fiber {
  // ==================== 实例相关 ====================

  /**
   * Fiber 类型标签
   * 📁 定义位置: ReactWorkTags.js
   */
  tag: WorkTag;

  /**
   * 唯一标识，用于 Diff 优化
   * 来源: React Element 的 key
   */
  key: string | null;

  /**
   * 元素类型
   * - 函数组件: function
   * - 类组件: class
   * - 原生标签: string ('div', 'span')
   */
  elementType: any;

  /**
   * 解析后的类型
   * 通常与 elementType 相同
   * lazy 组件解析后可能不同
   */
  type: any;

  /**
   * 对应的真实 DOM 节点或组件实例
   * - HostComponent (div): DOM 节点
   * - ClassComponent: 组件实例
   * - FunctionComponent: null
   * - HostRoot: FiberRoot
   */
  stateNode: any;

  // ==================== 树结构（链表） ====================

  /**
   * 父 Fiber
   * 命名为 return 是因为处理完当前节点后要"返回"到父节点
   */
  return: Fiber | null;

  /**
   * 第一个子 Fiber
   */
  child: Fiber | null;

  /**
   * 下一个兄弟 Fiber
   */
  sibling: Fiber | null;

  /**
   * 在兄弟节点中的索引
   */
  index: number;

  // ==================== 引用相关 ====================

  /**
   * ref 属性
   */
  ref: any;

  // ==================== 状态相关 ====================

  /**
   * 新的 props（待处理）
   */
  pendingProps: any;

  /**
   * 上次渲染使用的 props
   */
  memoizedProps: any;

  /**
   * 更新队列
   * - 类组件: UpdateQueue
   * - 函数组件: Effect 链表
   */
  updateQueue: any;

  /**
   * ⭐ 上次渲染的 state
   * - 类组件: state 对象
   * - 函数组件: Hooks 链表！
   */
  memoizedState: any;

  /**
   * Context 依赖
   */
  dependencies: Dependencies | null;

  // ==================== 模式相关 ====================

  /**
   * 渲染模式位掩码
   * - NoMode
   * - ConcurrentMode
   * - StrictMode
   * - ProfileMode
   */
  mode: TypeOfMode;

  // ==================== 副作用相关 ====================

  /**
   * ⭐ 副作用标记
   * 📁 定义位置: ReactFiberFlags.js
   * - Placement: 插入
   * - Update: 更新
   * - Deletion: 删除
   * - Ref: ref 变更
   * - Passive: useEffect
   */
  flags: Flags;

  /**
   * 子树副作用标记（冒泡）
   */
  subtreeFlags: Flags;

  /**
   * 要删除的子节点
   */
  deletions: Array<Fiber> | null;

  // ==================== 调度相关 ====================

  /**
   * 优先级（Lane 模型）
   */
  lanes: Lanes;

  /**
   * 子树优先级
   */
  childLanes: Lanes;

  // ==================== 双缓冲 ====================

  /**
   * ⭐ 指向另一棵树的对应节点
   * current.alternate = workInProgress
   * workInProgress.alternate = current
   */
  alternate: Fiber | null;
}

// ============================================================
// Part 3: WorkTag - Fiber 类型标签
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactWorkTags.js
 *
 * WorkTag 标识 Fiber 节点的类型
 */

const WorkTags = {
  FunctionComponent: 0,        // 函数组件
  ClassComponent: 1,           // 类组件
  IndeterminateComponent: 2,   // 未确定类型（首次渲染前）
  HostRoot: 3,                 // ⭐ 根节点（FiberRoot.current）
  HostPortal: 4,               // Portal
  HostComponent: 5,            // ⭐ 原生 DOM 元素 (div, span)
  HostText: 6,                 // ⭐ 文本节点
  Fragment: 7,                 // Fragment
  Mode: 8,                     // StrictMode, ConcurrentMode
  ContextConsumer: 9,          // Context.Consumer
  ContextProvider: 10,         // Context.Provider
  ForwardRef: 11,              // forwardRef 组件
  Profiler: 12,                // Profiler 组件
  SuspenseComponent: 13,       // ⭐ Suspense 组件
  MemoComponent: 14,           // memo 组件
  SimpleMemoComponent: 15,     // 简单 memo 组件
  LazyComponent: 16,           // lazy 组件
  IncompleteClassComponent: 17, // 未完成的类组件
  DehydratedFragment: 18,      // SSR 脱水 Fragment
  SuspenseListComponent: 19,   // SuspenseList
  ScopeComponent: 21,          // Scope
  OffscreenComponent: 22,      // Offscreen
  LegacyHiddenComponent: 23,   // 旧版 Hidden
  CacheComponent: 24,          // Cache
  TracingMarkerComponent: 25,  // Tracing Marker
};

/**
 * 📊 常见 WorkTag 示例
 */

const workTagExamples = `
// JSX 代码
<div className="app">
  <Header />
  <Content>
    <p>Hello</p>
    text content
  </Content>
</div>

// 对应的 Fiber 树 WorkTag:
HostRoot (3)
└── HostComponent (5)  // div
    ├── FunctionComponent (0)  // Header（假设是函数组件）
    └── ClassComponent (1)     // Content（假设是类组件）
        ├── HostComponent (5)  // p
        │   └── HostText (6)   // "Hello"
        └── HostText (6)       // "text content"
`;

// ============================================================
// Part 4: Flags - 副作用标记
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberFlags.js
 *
 * Flags 使用二进制位掩码，可以同时表示多个副作用
 */

const FiberFlags = {
  NoFlags: 0b00000000000000000000000000,       // 无副作用
  PerformedWork: 0b00000000000000000000000001, // 执行了工作
  Placement: 0b00000000000000000000000010,     // ⭐ 插入 DOM
  Update: 0b00000000000000000000000100,        // ⭐ 更新 DOM
  Deletion: 0b00000000000000000000001000,      // ⭐ 删除（已废弃，用 ChildDeletion）
  ChildDeletion: 0b00000000000000000000010000, // ⭐ 删除子节点
  ContentReset: 0b00000000000000000000100000,  // 重置文本内容
  Callback: 0b00000000000000000001000000,      // 有回调（setState 回调）
  DidCapture: 0b00000000000000000010000000,    // 捕获了错误
  Ref: 0b00000000000000001000000000,           // ⭐ ref 变更
  Snapshot: 0b00000000000000010000000000,      // getSnapshotBeforeUpdate
  Passive: 0b00000000000000100000000000,       // ⭐ useEffect
  Hydrating: 0b00000000000001000000000000,     // SSR 水合中
  Visibility: 0b00000000000010000000000000,    // 可见性变更
};

/**
 * 📊 位掩码运算示例
 */

const flagsExample = `
// 添加副作用
fiber.flags |= Placement;    // 添加 Placement 标记
fiber.flags |= Update;       // 同时添加 Update 标记

// 检查副作用
if (fiber.flags & Placement) {
  // 需要插入 DOM
}

// 移除副作用
fiber.flags &= ~Placement;   // 移除 Placement 标记

// 组合检查
const MutationMask = Placement | Update | ChildDeletion;
if (fiber.flags & MutationMask) {
  // 有 DOM 操作需要执行
}
`;

// ============================================================
// Part 5: FiberRoot 和 HostRootFiber
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberRoot.new.js
 *
 * FiberRoot 是整个应用的根节点
 * HostRootFiber 是 Fiber 树的根节点
 *
 * 它们的关系：
 * FiberRoot.current = HostRootFiber
 * HostRootFiber.stateNode = FiberRoot
 */

interface FiberRoot {
  /**
   * 根节点类型
   * - LegacyRoot: ReactDOM.render()
   * - ConcurrentRoot: createRoot()
   */
  tag: RootTag;

  /**
   * 容器 DOM 节点
   * 例如: document.getElementById('root')
   */
  containerInfo: any;

  /**
   * ⭐ 当前显示的 Fiber 树
   */
  current: Fiber;

  /**
   * ⭐ 已完成的工作（待提交）
   */
  finishedWork: Fiber | null;

  /**
   * 调度相关
   */
  callbackNode: any;
  callbackPriority: Lane;

  /**
   * 优先级相关（Lane 模型）
   */
  pendingLanes: Lanes;         // 待处理的优先级
  suspendedLanes: Lanes;       // 挂起的优先级
  pingedLanes: Lanes;          // 被 ping 的优先级
  expiredLanes: Lanes;         // 过期的优先级
  finishedLanes: Lanes;        // 已完成的优先级

  /**
   * 事件时间和过期时间
   */
  eventTimes: LaneMap<number>;
  expirationTimes: LaneMap<number>;
}

/**
 * 📊 FiberRoot 和 Fiber 树的关系图
 */

const fiberRootRelation = `
                     FiberRoot
                         │
                         │ current
                         ▼
           ┌───────────────────────────────┐
           │        HostRootFiber          │
           │    (tag: 3 = HostRoot)        │
           │    stateNode → FiberRoot      │
           └───────────────┬───────────────┘
                           │ child
                           ▼
           ┌───────────────────────────────┐
           │         App Fiber             │
           │  (tag: 0 = FunctionComponent) │
           └───────────────┬───────────────┘
                           │ child
                           ▼
           ┌───────────────────────────────┐
           │       HostComponent           │
           │    (tag: 5 = div)             │
           │    stateNode → <div>          │
           └───────────────────────────────┘

代码对应:
const root = createRoot(document.getElementById('root'));
root.render(<App />);

// FiberRoot.containerInfo = document.getElementById('root')
// FiberRoot.current = HostRootFiber
// HostRootFiber.child = App Fiber
`;

// ============================================================
// Part 6: 双缓冲机制（Double Buffering）
// ============================================================

/**
 * 📊 双缓冲概念
 *
 * React 同时维护两棵 Fiber 树：
 * - current 树：当前屏幕显示的内容
 * - workInProgress 树：正在构建的新树
 *
 * 为什么需要双缓冲？
 * 1. 构建过程可中断，current 树保持稳定显示
 * 2. 如果构建失败，可以丢弃 workInProgress
 * 3. 完成后通过指针切换，O(1) 复杂度
 */

const doubleBufferingExplanation = `
📊 双缓冲工作流程

初始状态（首次渲染后）:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  FiberRoot                                                  │
│      │                                                      │
│      │ current                                              │
│      ▼                                                      │
│  ┌───────────┐         alternate         ┌───────────┐     │
│  │  current  │ ◄─────────────────────►   │ (无)      │     │
│  │   树      │                           │           │     │
│  └───────────┘                           └───────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

更新开始:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  FiberRoot                                                  │
│      │                                                      │
│      │ current                                              │
│      ▼                                                      │
│  ┌───────────┐         alternate         ┌───────────┐     │
│  │  current  │ ◄─────────────────────►   │   WIP     │     │
│  │   树      │                           │   树      │     │
│  │ (显示中)  │                           │ (构建中)  │     │
│  └───────────┘                           └───────────┘     │
│                                                             │
│  current 树继续显示，WIP 树在后台构建                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Commit 阶段（指针切换）:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  FiberRoot                                                  │
│      │                                                      │
│      │ current（指针切换！）                                │
│      │                                                      │
│      │              alternate                               │
│      ▼         ┌───────────────────►                        │
│  ┌───────────┐ │                     ┌───────────┐         │
│  │  旧树     │◄┘                     │   新树    │         │
│  │ (备用)   │ ◄─────────────────────►│ (当前)   │         │
│  └───────────┘         alternate     └───────────┘         │
│                                          ▲                  │
│                                          │                  │
│                                     现在显示这棵             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

下次更新时，旧树变成新的 workInProgress，循环复用
`;

/**
 * 📊 alternate 连接
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiber.new.js
 *             createWorkInProgress 函数
 */

// 简化版 createWorkInProgress
function createWorkInProgressSimplified(current: Fiber, pendingProps: any): Fiber {
  let workInProgress = current.alternate;

  if (workInProgress === null) {
    // 首次渲染，创建新 Fiber
    workInProgress = createFiber(current.tag, pendingProps, current.key, current.mode);
    workInProgress.elementType = current.elementType;
    workInProgress.type = current.type;
    workInProgress.stateNode = current.stateNode;

    // ⭐ 建立双向连接
    workInProgress.alternate = current;
    current.alternate = workInProgress;
  } else {
    // 更新，复用已有 Fiber
    workInProgress.pendingProps = pendingProps;
    workInProgress.type = current.type;

    // 重置副作用
    workInProgress.flags = NoFlags;
    workInProgress.subtreeFlags = NoFlags;
    workInProgress.deletions = null;
  }

  // 复制其他属性
  workInProgress.flags = current.flags & StaticMask;
  workInProgress.childLanes = current.childLanes;
  workInProgress.lanes = current.lanes;

  workInProgress.child = current.child;
  workInProgress.memoizedProps = current.memoizedProps;
  workInProgress.memoizedState = current.memoizedState;
  workInProgress.updateQueue = current.updateQueue;

  return workInProgress;
}

// ============================================================
// Part 7: Fiber 树的遍历顺序
// ============================================================

/**
 * 📊 深度优先遍历（DFS）
 *
 * Fiber 树的遍历分为两个阶段：
 * 1. "递"阶段（beginWork）：从根向下
 * 2. "归"阶段（completeWork）：从下向上
 */

const traversalOrder = `
📊 Fiber 树遍历顺序示例

假设有以下组件结构:
<App>
  <Header />
  <Main>
    <Article />
    <Sidebar />
  </Main>
</App>

Fiber 树结构:
        App
         │
    ┌────┴────┐
 Header     Main
             │
        ┌────┴────┐
    Article    Sidebar

遍历顺序:
┌─────────────────────────────────────────────────────────────┐
│  阶段        │  节点        │  操作                         │
├─────────────────────────────────────────────────────────────┤
│  1. 递       │  App         │  beginWork(App)               │
│  2. 递       │  Header      │  beginWork(Header)            │
│  3. 归       │  Header      │  completeWork(Header)         │
│  4. 递       │  Main        │  beginWork(Main)              │
│  5. 递       │  Article     │  beginWork(Article)           │
│  6. 归       │  Article     │  completeWork(Article)        │
│  7. 递       │  Sidebar     │  beginWork(Sidebar)           │
│  8. 归       │  Sidebar     │  completeWork(Sidebar)        │
│  9. 归       │  Main        │  completeWork(Main)           │
│  10. 归      │  App         │  completeWork(App)            │
└─────────────────────────────────────────────────────────────┘

遍历规则:
1. 有 child → 进入 child（递）
2. 无 child → completeWork（归）
3. 有 sibling → 进入 sibling（递）
4. 无 sibling → 返回 parent，继续归
`;

/**
 * 📊 简化版工作循环
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 */

function workLoopSimplified(unitOfWork: Fiber | null) {
  while (unitOfWork !== null) {
    // "递"阶段：执行 beginWork
    const next = performUnitOfWork(unitOfWork);

    if (next === null) {
      // 没有子节点，进入"归"阶段
      completeUnitOfWork(unitOfWork);
    }

    unitOfWork = next;
  }
}

function performUnitOfWork(unitOfWork: Fiber): Fiber | null {
  // beginWork: 根据 Fiber 类型进行不同处理
  // 返回 child Fiber 或 null
  const next = beginWork(unitOfWork);
  return next;
}

function completeUnitOfWork(unitOfWork: Fiber) {
  let completedWork: Fiber | null = unitOfWork;

  while (completedWork !== null) {
    // completeWork: 创建 DOM 节点，收集副作用
    completeWork(completedWork);

    // 检查是否有兄弟节点
    const siblingFiber = completedWork.sibling;
    if (siblingFiber !== null) {
      // 有兄弟节点，开始处理兄弟（递）
      workLoopSimplified(siblingFiber);
      return;
    }

    // 没有兄弟，返回父节点继续归
    completedWork = completedWork.return;
  }
}

// 辅助函数声明（实际实现在其他文件）
declare function beginWork(fiber: Fiber): Fiber | null;
declare function completeWork(fiber: Fiber): void;
declare function createFiber(tag: number, pendingProps: any, key: string | null, mode: number): Fiber;
const NoFlags = 0;
const StaticMask = 0;

// ============================================================
// Part 8: 从 Element 到 Fiber
// ============================================================

/**
 * 📊 React Element 和 Fiber 的关系
 */

const elementToFiber = `
📊 Element → Fiber 转换

React Element（描述 UI）:
{
  $$typeof: Symbol(react.element),
  type: 'div',
  key: 'unique',
  props: { className: 'container', children: 'Hello' }
}

         ↓ 首次渲染时创建 Fiber

Fiber 节点（工作单元）:
{
  tag: 5 (HostComponent),
  type: 'div',
  key: 'unique',
  pendingProps: { className: 'container', children: 'Hello' },
  memoizedProps: null,  // 首次渲染前
  stateNode: null,      // completeWork 时创建 DOM
  return: parentFiber,
  child: textFiber,     // 'Hello' 对应的 TextFiber
  sibling: null,
  alternate: null,      // 首次渲染
  flags: Placement,     // 需要插入 DOM
  ...
}

         ↓ completeWork 阶段

创建真实 DOM:
const dom = document.createElement('div');
dom.className = 'container';
fiber.stateNode = dom;
`;

// ============================================================
// Part 9: 面试题
// ============================================================

const interviewQuestions = `
💡 Q1: 什么是 Fiber？为什么需要 Fiber？
A: Fiber 是 React 16 引入的新架构，每个 Fiber 是一个工作单元。
   需要 Fiber 是因为：
   1. 实现可中断渲染（递归变循环）
   2. 支持优先级调度（高优先级先处理）
   3. 支持时间切片（不阻塞主线程）
   4. 支持并发模式（Concurrent Mode）

💡 Q2: Fiber 节点有哪些重要属性？
A: 1. 结构属性：return、child、sibling（链表结构）
   2. 状态属性：memoizedState、memoizedProps、updateQueue
   3. 副作用：flags、subtreeFlags
   4. 调度：lanes、childLanes
   5. 双缓冲：alternate

💡 Q3: 什么是双缓冲？为什么需要？
A: 双缓冲是同时维护两棵 Fiber 树：
   - current 树：当前显示
   - workInProgress 树：正在构建
   
   好处：
   1. 构建失败可丢弃，不影响当前显示
   2. 构建过程可中断
   3. 切换只需改指针，O(1) 复杂度
   4. Fiber 节点可复用（通过 alternate）

💡 Q4: Fiber 树如何遍历？
A: 深度优先遍历，分两个阶段：
   1. "递"阶段（beginWork）：从根向下处理
   2. "归"阶段（completeWork）：从下向上
   
   顺序：先 child，后 sibling，再 return

💡 Q5: flags 有什么作用？
A: flags 是副作用标记，用位掩码表示，常见的：
   - Placement: 需要插入 DOM
   - Update: 需要更新 DOM 属性
   - ChildDeletion: 需要删除子节点
   - Ref: 需要处理 ref
   - Passive: 有 useEffect 需要执行

💡 Q6: stateNode 存的是什么？
A: 根据 Fiber 类型不同：
   - HostRoot: FiberRoot
   - HostComponent (div): DOM 节点
   - ClassComponent: 组件实例
   - FunctionComponent: null

💡 Q7: 为什么函数组件的 Hooks 在 memoizedState 上？
A: memoizedState 对于不同类型 Fiber 存储不同内容：
   - ClassComponent: state 对象
   - FunctionComponent: Hooks 链表
   这样设计是为了复用字段，减少内存占用。

💡 Q8: 什么是 Lane？和 Fiber 什么关系？
A: Lane 是优先级模型，用位掩码表示：
   - Fiber.lanes: 该节点的更新优先级
   - Fiber.childLanes: 子树的更新优先级
   - FiberRoot.pendingLanes: 待处理的优先级
   用于调度决定先处理哪些更新。
`;

// ============================================================
// Part 10: 实践练习
// ============================================================

/**
 * 练习 1：手写简化版 Fiber 节点创建
 */
function createFiberNode(
  tag: number,
  pendingProps: any,
  key: string | null
): Fiber {
  return {
    // 实例相关
    tag,
    key,
    elementType: null,
    type: null,
    stateNode: null,

    // 树结构
    return: null,
    child: null,
    sibling: null,
    index: 0,

    // ref
    ref: null,

    // 状态
    pendingProps,
    memoizedProps: null,
    updateQueue: null,
    memoizedState: null,
    dependencies: null,

    // 模式
    mode: 0,

    // 副作用
    flags: 0,
    subtreeFlags: 0,
    deletions: null,

    // 调度
    lanes: 0,
    childLanes: 0,

    // 双缓冲
    alternate: null,
  } as Fiber;
}

/**
 * 练习 2：模拟 Fiber 树遍历
 */
function traverseFiberTree(root: Fiber) {
  let node: Fiber | null = root;

  while (node !== null) {
    // 递阶段
    console.log('beginWork:', node.type || node.tag);

    if (node.child !== null) {
      // 有子节点，继续递
      node = node.child;
      continue;
    }

    // 归阶段
    let completedNode: Fiber | null = node;
    while (completedNode !== null) {
      console.log('completeWork:', completedNode.type || completedNode.tag);

      if (completedNode.sibling !== null) {
        // 有兄弟，处理兄弟
        node = completedNode.sibling;
        break;
      }

      // 返回父节点
      completedNode = completedNode.return;
      if (completedNode === null) {
        node = null;
      }
    }
  }
}

// 类型定义
type WorkTag = number;
type TypeOfMode = number;
type Flags = number;
type Lanes = number;
type Lane = number;
type LaneMap<T> = Array<T>;
type RootTag = number;
interface Dependencies {
  lanes: Lanes;
  firstContext: any;
}

// ============================================================
// 学习检查清单
// ============================================================

/**
 * ✅ Phase 2 学习检查
 *
 * 基础概念：
 * - [ ] 理解 Stack Reconciler 的问题
 * - [ ] 理解 Fiber 如何解决可中断渲染
 * - [ ] 能说出 Fiber 的核心优势
 *
 * 数据结构：
 * - [ ] 能画出 Fiber 节点的结构图
 * - [ ] 理解 return/child/sibling 的链表关系
 * - [ ] 理解 stateNode 在不同类型的含义
 * - [ ] 理解 memoizedState 的作用
 *
 * 双缓冲：
 * - [ ] 理解 current 和 workInProgress 的关系
 * - [ ] 理解 alternate 的作用
 * - [ ] 理解指针切换的时机
 *
 * 遍历：
 * - [ ] 理解"递"和"归"两个阶段
 * - [ ] 能说出遍历顺序
 *
 * 源码位置：
 * - [ ] 能找到 FiberNode 定义
 * - [ ] 能找到 WorkTags 定义
 * - [ ] 能找到 FiberFlags 定义
 */

export {
  WorkTags,
  FiberFlags,
  createFiberNode,
  traverseFiberTree,
  createWorkInProgressSimplified,
  workLoopSimplified,
  whyFiberExplanation,
  workTagExamples,
  flagsExample,
  fiberRootRelation,
  doubleBufferingExplanation,
  traversalOrder,
  elementToFiber,
  interviewQuestions,
};

