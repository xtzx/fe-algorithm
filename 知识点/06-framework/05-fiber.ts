/**
 * ============================================================
 * 📚 React Fiber 架构
 * ============================================================
 *
 * 面试考察重点：
 * 1. Fiber 解决什么问题？
 * 2. Fiber 的数据结构
 * 3. 双缓冲机制
 * 4. 调度机制（Scheduler）
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 为什么需要 Fiber？
 *
 * React 15 的问题（Stack Reconciler）：
 * - 递归遍历虚拟 DOM
 * - 无法中断，必须一次性完成
 * - 大组件树会导致主线程长时间阻塞
 * - 用户交互无响应（卡顿）
 *
 * Fiber 的解决方案：
 * - 将递归改为链表遍历
 * - 可以随时中断和恢复
 * - 实现时间切片
 * - 高优先级任务可以打断低优先级
 *
 * 📊 Stack Reconciler vs Fiber Reconciler
 *
 * Stack Reconciler（React 15）：
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 递归渲染，不可中断                                          │
 * │                                                             │
 * │ render ──────────────────────────────────────────────► done │
 * │ (长时间阻塞主线程)                                          │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Fiber Reconciler（React 16+）：
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 时间切片，可中断可恢复                                       │
 * │                                                             │
 * │ render ─► pause ─► render ─► pause ─► render ─► commit      │
 * │ (5ms)    (让出)    (5ms)    (让出)    (5ms)    (不可中断)   │
 * └─────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. Fiber 数据结构
// ============================================================

/**
 * 📊 Fiber 节点结构
 *
 * Fiber 是一个 JavaScript 对象，包含：
 * 1. 静态数据（对应的 React 元素信息）
 * 2. 动态数据（组件状态、副作用）
 * 3. 关系指针（形成链表结构）
 */

interface FiberNode {
  // === 静态数据 ===
  tag: number;           // 组件类型（函数组件、类组件、DOM 元素等）
  type: any;             // 对应的 React 元素类型
  key: string | null;    // key 属性

  // === 动态数据 ===
  memoizedState: any;    // Hooks 链表 / 类组件 state
  memoizedProps: any;    // 上次渲染的 props
  pendingProps: any;     // 新的 props
  updateQueue: any;      // 更新队列

  // === 副作用 ===
  flags: number;         // 副作用标记（新增、更新、删除）
  subtreeFlags: number;  // 子树副作用标记
  deletions: FiberNode[] | null; // 要删除的子 Fiber

  // === 关系指针（链表结构）===
  return: FiberNode | null;   // 父节点
  child: FiberNode | null;    // 第一个子节点
  sibling: FiberNode | null;  // 下一个兄弟节点

  // === 双缓冲 ===
  alternate: FiberNode | null; // 对应的另一个 Fiber（current/workInProgress）

  // === DOM ===
  stateNode: any;        // 对应的真实 DOM 或组件实例
}

/**
 * 📊 Fiber 树的遍历顺序
 *
 *        App
 *       / | \
 *     A   B   C
 *    / \
 *   D   E
 *
 * 遍历顺序（深度优先）：
 * App → A → D → E → B → C
 *
 * 关系：
 * - App.child = A
 * - A.sibling = B
 * - B.sibling = C
 * - A.return = App
 * - A.child = D
 * - D.sibling = E
 */

// Fiber 树遍历（简化版）
function performUnitOfWork(fiber: FiberNode): FiberNode | null {
  // 1. 处理当前 Fiber（beginWork）
  beginWork(fiber);

  // 2. 如果有子节点，返回子节点
  if (fiber.child) {
    return fiber.child;
  }

  // 3. 没有子节点，处理当前节点（completeWork）
  let current: FiberNode | null = fiber;
  while (current) {
    completeWork(current);

    // 4. 有兄弟节点，返回兄弟节点
    if (current.sibling) {
      return current.sibling;
    }

    // 5. 没有兄弟节点，返回父节点继续处理
    current = current.return;
  }

  return null;
}

function beginWork(fiber: FiberNode) {
  // 处理 Fiber：创建子 Fiber、Diff 等
  console.log('beginWork:', fiber.type);
}

function completeWork(fiber: FiberNode) {
  // 完成 Fiber：创建 DOM、收集副作用等
  console.log('completeWork:', fiber.type);
}

// ============================================================
// 3. 双缓冲机制
// ============================================================

/**
 * 📊 双缓冲（Double Buffering）
 *
 * React 维护两棵 Fiber 树：
 * 1. current：当前页面显示的树
 * 2. workInProgress：正在构建的新树
 *
 * 更新流程：
 * 1. 基于 current 创建 workInProgress
 * 2. 在 workInProgress 上进行更新
 * 3. 完成后，workInProgress 变成新的 current
 *
 * 优势：
 * - 可以复用 Fiber 节点
 * - 更新过程可中断
 * - 不影响当前显示
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │   current                workInProgress                     │
 * │      │                         │                            │
 * │      ▼                         ▼                            │
 * │   ┌─────┐     alternate     ┌─────┐                        │
 * │   │  A  │ ◄───────────────► │  A' │                        │
 * │   └──┬──┘                   └──┬──┘                        │
 * │      │                         │                            │
 * │   ┌──┴──┐                   ┌──┴──┐                        │
 * │   │  B  │ ◄───────────────► │  B' │                        │
 * │   └─────┘                   └─────┘                        │
 * │                                                             │
 * │   更新完成后，workInProgress 变成新的 current               │
 * └─────────────────────────────────────────────────────────────┘
 */

// 双缓冲创建 workInProgress
function createWorkInProgress(current: FiberNode, pendingProps: any): FiberNode {
  let workInProgress = current.alternate;

  if (workInProgress === null) {
    // 首次渲染，创建新 Fiber
    workInProgress = {
      tag: current.tag,
      type: current.type,
      key: current.key,
      stateNode: current.stateNode,

      return: null,
      child: null,
      sibling: null,

      memoizedState: current.memoizedState,
      memoizedProps: current.memoizedProps,
      pendingProps: pendingProps,
      updateQueue: current.updateQueue,

      flags: 0,
      subtreeFlags: 0,
      deletions: null,

      alternate: current,
    };
    current.alternate = workInProgress;
  } else {
    // 更新渲染，复用 Fiber
    workInProgress.pendingProps = pendingProps;
    workInProgress.flags = 0;
    workInProgress.subtreeFlags = 0;
    workInProgress.deletions = null;
  }

  return workInProgress;
}

// ============================================================
// 4. 调度机制（Scheduler）
// ============================================================

/**
 * 📊 优先级调度
 *
 * React 定义了不同的优先级：
 * 1. Immediate（同步）：用户输入、动画
 * 2. UserBlocking：点击、输入
 * 3. Normal：普通更新
 * 4. Low：数据获取
 * 5. Idle：不紧急的更新
 *
 * 调度流程：
 * 1. 创建更新，标记优先级
 * 2. 调度器选择最高优先级任务
 * 3. 在时间切片内执行
 * 4. 时间用尽，让出主线程
 * 5. 继续调度下一个任务
 */

/**
 * 📊 时间切片（Time Slicing）
 *
 * 每个时间切片约 5ms
 * 如果有更高优先级任务，会打断当前任务
 */

// 简化的调度实现
class SimpleScheduler {
  private taskQueue: Array<{ callback: () => void; priority: number }> = [];
  private isScheduled = false;

  scheduleTask(callback: () => void, priority: number) {
    this.taskQueue.push({ callback, priority });
    // 按优先级排序
    this.taskQueue.sort((a, b) => a.priority - b.priority);

    if (!this.isScheduled) {
      this.isScheduled = true;
      this.schedulePerform();
    }
  }

  private schedulePerform() {
    // 使用 MessageChannel 创建宏任务
    const channel = new MessageChannel();
    channel.port1.onmessage = () => this.performWork();
    channel.port2.postMessage(null);
  }

  private performWork() {
    const startTime = performance.now();
    const frameTime = 5; // 5ms 时间切片

    while (this.taskQueue.length > 0) {
      // 检查是否超时
      if (performance.now() - startTime >= frameTime) {
        // 让出主线程，下一帧继续
        this.schedulePerform();
        return;
      }

      const task = this.taskQueue.shift()!;
      task.callback();
    }

    this.isScheduled = false;
  }
}

// ============================================================
// 5. 渲染流程
// ============================================================

/**
 * 📊 两个阶段
 *
 * 1. Render 阶段（可中断）
 *    - 创建 Fiber 树
 *    - Diff 对比
 *    - 标记副作用
 *    - 可以被高优先级任务打断
 *
 * 2. Commit 阶段（不可中断）
 *    - 执行 DOM 操作
 *    - 执行生命周期
 *    - 必须同步完成
 *
 * Commit 阶段三个子阶段：
 * 1. Before Mutation：DOM 变更前
 *    - getSnapshotBeforeUpdate
 * 2. Mutation：执行 DOM 操作
 *    - 插入、更新、删除 DOM
 * 3. Layout：DOM 变更后
 *    - componentDidMount
 *    - componentDidUpdate
 *    - useLayoutEffect
 */

// 简化的渲染流程
function renderRoot(root: FiberNode) {
  let workInProgress: FiberNode | null = root;

  // Render 阶段：可中断
  while (workInProgress !== null) {
    workInProgress = performUnitOfWork(workInProgress);

    // 检查是否需要让出（简化版）
    if (shouldYield()) {
      // 保存进度，下次继续
      return;
    }
  }

  // Commit 阶段：不可中断
  commitRoot(root);
}

function shouldYield(): boolean {
  // 检查是否有更高优先级任务
  // 检查时间片是否用尽
  return false;
}

function commitRoot(root: FiberNode) {
  // Before Mutation
  commitBeforeMutationEffects(root);

  // Mutation：执行 DOM 操作
  commitMutationEffects(root);

  // 切换 current 指针
  // root.current = root.workInProgress;

  // Layout：执行生命周期
  commitLayoutEffects(root);
}

function commitBeforeMutationEffects(fiber: FiberNode) {
  // getSnapshotBeforeUpdate
}

function commitMutationEffects(fiber: FiberNode) {
  // DOM 操作
}

function commitLayoutEffects(fiber: FiberNode) {
  // componentDidMount, useLayoutEffect
}

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见误解
 *
 * 1. Fiber 不是虚拟 DOM
 *    - 虚拟 DOM 是 React 元素（描述结构）
 *    - Fiber 是调度单元（包含更多信息）
 *
 * 2. 时间切片不是总是开启
 *    - 同步更新不会时间切片
 *    - 只有并发模式下才有
 *
 * 3. 并发模式不是多线程
 *    - JavaScript 仍然是单线程
 *    - 只是任务可以被打断和恢复
 *
 * 4. Commit 阶段不能中断
 *    - DOM 操作必须一次完成
 *    - 避免中间状态显示给用户
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Fiber 和虚拟 DOM 的区别？
 * A:
 *    虚拟 DOM：
 *    - React 元素，描述 UI 结构
 *    - 每次渲染都会重新创建
 *
 *    Fiber：
 *    - 工作单元，包含状态、副作用、调度信息
 *    - 可以复用（双缓冲）
 *
 * Q2: 为什么 Render 阶段可以中断，Commit 阶段不能？
 * A:
 *    Render 阶段：
 *    - 只是在内存中计算
 *    - 不影响页面显示
 *    - 可以重新开始
 *
 *    Commit 阶段：
 *    - 操作真实 DOM
 *    - 中断会导致页面不一致
 *    - 必须同步完成
 *
 * Q3: React 18 的并发特性有哪些？
 * A:
 *    - useTransition：标记低优先级更新
 *    - useDeferredValue：延迟更新值
 *    - Suspense：异步渲染
 *    - 自动批量更新
 *
 * Q4: 什么时候会触发时间切片？
 * A:
 *    - 使用 createRoot（并发模式）
 *    - 使用 startTransition 标记的更新
 *    - 低优先级更新
 *
 * Q5: 如何理解 Lane 模型？
 * A:
 *    - 用位运算表示优先级
 *    - 可以合并多个优先级
 *    - 比之前的 ExpirationTime 更灵活
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：使用 useTransition 优化搜索
 */

const useTransitionExample = `
function SearchResults() {
  const [query, setQuery] = useState('');
  const [isPending, startTransition] = useTransition();

  const handleChange = (e) => {
    // 输入是高优先级，立即更新
    setQuery(e.target.value);

    // 搜索结果是低优先级，可以被打断
    startTransition(() => {
      setSearchResults(search(e.target.value));
    });
  };

  return (
    <div>
      <input value={query} onChange={handleChange} />
      {isPending ? <Spinner /> : <Results />}
    </div>
  );
}
`;

/**
 * 🏢 场景 2：使用 useDeferredValue 优化列表
 */

const useDeferredValueExample = `
function List({ query }) {
  // query 变化时，deferredQuery 会延迟更新
  const deferredQuery = useDeferredValue(query);

  // 使用 deferredQuery 渲染列表
  // 输入时 UI 不会卡顿
  const items = useMemo(
    () => filterItems(deferredQuery),
    [deferredQuery]
  );

  return (
    <ul style={{ opacity: query !== deferredQuery ? 0.5 : 1 }}>
      {items.map(item => <li key={item.id}>{item.name}</li>)}
    </ul>
  );
}
`;

/**
 * 🏢 场景 3：Suspense 数据获取
 */

const suspenseExample = `
// 使用 Suspense 包裹异步组件
function App() {
  return (
    <Suspense fallback={<Loading />}>
      <UserProfile userId={1} />
    </Suspense>
  );
}

// 配合 React Query / SWR 等库使用
function UserProfile({ userId }) {
  const { data } = useSuspenseQuery(['user', userId], fetchUser);
  return <div>{data.name}</div>;
}
`;

export {
  // Fiber 相关
  performUnitOfWork,
  beginWork,
  completeWork,
  createWorkInProgress,

  // 调度相关
  SimpleScheduler,

  // 渲染相关
  renderRoot,
  commitRoot,

  // 示例
  useTransitionExample,
  useDeferredValueExample,
  suspenseExample,
};

export type { FiberNode };

