/**
 * ============================================================
 * 📚 Phase 4: Hooks 原理（核心重点）
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 Hooks 的存储结构
 * 2. 掌握 useState 的实现原理
 * 3. 掌握 useEffect 的实现原理
 * 4. 理解 Hooks 规则的原因
 *
 * 📁 源码位置：
 * - packages/react-reconciler/src/ReactFiberHooks.js
 *
 * ⏱️ 预计时间：8 小时
 * 🔥 面试权重：⭐⭐⭐⭐⭐（必考）
 */

// ============================================================
// 1. Hooks 存储结构
// ============================================================

/**
 * 📊 Hooks 链表
 *
 * 每个函数组件的 Fiber 节点上有 memoizedState
 * memoizedState 指向一个 Hooks 链表
 *
 * ```
 * FiberNode
 *     │
 *     │ memoizedState
 *     ▼
 * ┌─────────┐     ┌─────────┐     ┌─────────┐
 * │  Hook1  │────►│  Hook2  │────►│  Hook3  │
 * │useState │     │useEffect│     │useMemo  │
 * └─────────┘     └─────────┘     └─────────┘
 *     next            next            next
 * ```
 *
 * Hook 数据结构：
 */

interface Hook {
  memoizedState: any;      // 保存的状态值
  baseState: any;          // 基础状态
  baseQueue: any;          // 基础队列
  queue: UpdateQueue | null; // 更新队列
  next: Hook | null;       // 下一个 Hook
}

interface UpdateQueue {
  pending: Update | null;   // 待处理的更新
  dispatch: any;            // dispatch 函数
  lastRenderedState: any;   // 上次渲染的状态
}

interface Update {
  action: any;              // 更新的值或函数
  next: Update | null;      // 下一个更新
}

// ============================================================
// 2. useState 实现
// ============================================================

/**
 * 📊 useState 工作流程
 *
 * 首次渲染（mount）：
 * 1. 创建 Hook 对象
 * 2. 初始化 state
 * 3. 创建 dispatch 函数
 * 4. 返回 [state, dispatch]
 *
 * 更新（update）：
 * 1. 找到对应的 Hook
 * 2. 计算新的 state
 * 3. 返回 [newState, dispatch]
 */

// 简化版 useState 实现
let currentlyRenderingFiber: any = null;  // 当前渲染的 Fiber
let workInProgressHook: Hook | null = null; // 当前处理的 Hook
let currentHook: Hook | null = null;        // current 树的 Hook

// mount 阶段的 useState
function mountState<S>(initialState: S | (() => S)): [S, (action: S | ((s: S) => S)) => void] {
  // 1. 创建 Hook
  const hook: Hook = {
    memoizedState: typeof initialState === 'function'
      ? (initialState as () => S)()
      : initialState,
    baseState: null,
    baseQueue: null,
    queue: {
      pending: null,
      dispatch: null,
      lastRenderedState: null,
    },
    next: null,
  };

  // 2. 添加到链表
  if (workInProgressHook === null) {
    currentlyRenderingFiber.memoizedState = hook;
    workInProgressHook = hook;
  } else {
    workInProgressHook.next = hook;
    workInProgressHook = hook;
  }

  // 3. 创建 dispatch
  const queue = hook.queue!;
  const dispatch = (queue.dispatch = dispatchSetState.bind(
    null,
    currentlyRenderingFiber,
    queue
  ));

  return [hook.memoizedState, dispatch];
}

// update 阶段的 useState
function updateState<S>(): [S, (action: S | ((s: S) => S)) => void] {
  // 1. 获取当前 Hook
  const hook = updateWorkInProgressHook();

  // 2. 计算新状态
  const queue = hook.queue!;
  const pending = queue.pending;

  if (pending !== null) {
    let newState = hook.memoizedState;
    let update: Update | null = pending.next;

    // 遍历更新链表
    do {
      const action = update!.action;
      newState = typeof action === 'function'
        ? action(newState)
        : action;
      update = update!.next;
    } while (update !== pending.next);

    hook.memoizedState = newState;
    queue.pending = null;
  }

  return [hook.memoizedState, queue.dispatch];
}

// dispatch 函数
function dispatchSetState<S>(
  fiber: any,
  queue: UpdateQueue,
  action: S | ((s: S) => S)
) {
  // 1. 创建 Update
  const update: Update = {
    action,
    next: null,
  };

  // 2. 加入更新队列（环形链表）
  const pending = queue.pending;
  if (pending === null) {
    update.next = update;
  } else {
    update.next = pending.next;
    pending.next = update;
  }
  queue.pending = update;

  // 3. 调度更新
  scheduleUpdateOnFiber(fiber);
}

function scheduleUpdateOnFiber(fiber: any) {
  // 简化：触发重新渲染
  console.log('Schedule update on fiber:', fiber);
}

function updateWorkInProgressHook(): Hook {
  // 从 current 树获取对应的 Hook
  const current = currentHook;
  currentHook = current!.next;

  // 复制到 workInProgress
  const newHook: Hook = {
    memoizedState: current!.memoizedState,
    baseState: current!.baseState,
    baseQueue: current!.baseQueue,
    queue: current!.queue,
    next: null,
  };

  if (workInProgressHook === null) {
    currentlyRenderingFiber.memoizedState = newHook;
    workInProgressHook = newHook;
  } else {
    workInProgressHook.next = newHook;
    workInProgressHook = newHook;
  }

  return newHook;
}

// ============================================================
// 3. useEffect 实现
// ============================================================

/**
 * 📊 useEffect 工作流程
 *
 * Effect 数据结构：
 */

interface Effect {
  tag: number;              // 类型标记
  create: () => (() => void) | void;  // 回调函数
  destroy: (() => void) | void;        // 清理函数
  deps: any[] | null;       // 依赖数组
  next: Effect | null;      // 下一个 Effect
}

/**
 * 📊 Effect 执行时机
 *
 * ```
 * Commit 阶段
 *     │
 *     ├─ Before Mutation
 *     │      调度 useEffect（不执行）
 *     │
 *     ├─ Mutation
 *     │      执行 useLayoutEffect 的 destroy
 *     │
 *     ├─ Layout
 *     │      执行 useLayoutEffect 的 create
 *     │
 *     └─ 异步（下一帧）
 *            执行 useEffect 的 destroy 和 create
 * ```
 */

// 简化版 useEffect 实现
function mountEffect(
  create: () => (() => void) | void,
  deps: any[] | null
) {
  const hook: Hook = {
    memoizedState: null,
    baseState: null,
    baseQueue: null,
    queue: null,
    next: null,
  };

  // 创建 Effect
  const effect: Effect = {
    tag: 0, // Passive
    create,
    destroy: undefined,
    deps,
    next: null,
  };

  hook.memoizedState = effect;

  // 添加到 Fiber 的 updateQueue
  pushEffect(effect);
}

function updateEffect(
  create: () => (() => void) | void,
  deps: any[] | null
) {
  const hook = updateWorkInProgressHook();
  const prevEffect = hook.memoizedState as Effect;

  // 比较依赖
  if (deps !== null && areHookInputsEqual(deps, prevEffect.deps)) {
    // 依赖没变，不需要执行
    return;
  }

  // 依赖变了，创建新 Effect
  const effect: Effect = {
    tag: 0,
    create,
    destroy: prevEffect.destroy,
    deps,
    next: null,
  };

  hook.memoizedState = effect;
  pushEffect(effect);
}

function pushEffect(effect: Effect) {
  // 添加到 Fiber 的 Effect 链表
  console.log('Push effect:', effect);
}

function areHookInputsEqual(nextDeps: any[], prevDeps: any[] | null): boolean {
  if (prevDeps === null) return false;
  for (let i = 0; i < prevDeps.length && i < nextDeps.length; i++) {
    if (Object.is(nextDeps[i], prevDeps[i])) {
      continue;
    }
    return false;
  }
  return true;
}

// ============================================================
// 4. Hooks 规则的原因
// ============================================================

/**
 * 📊 为什么 Hooks 不能放在条件语句中？
 *
 * 因为 Hooks 是链表结构，通过顺序匹配！
 *
 * 正确示例：
 * ```
 * 第一次渲染：Hook1 → Hook2 → Hook3
 * 第二次渲染：Hook1 → Hook2 → Hook3  ✅ 顺序一致
 * ```
 *
 * 错误示例（条件语句）：
 * ```
 * 第一次渲染：Hook1 → Hook2 → Hook3
 * 第二次渲染：Hook1 → Hook3          ❌ 顺序不一致
 *                     ↑
 *                 Hook2 被跳过了
 *                 但是 React 仍然按顺序取
 *                 导致 Hook3 拿到了 Hook2 的状态
 * ```
 *
 * 源码验证：
 * ```js
 * // packages/react-reconciler/src/ReactFiberHooks.js
 * function updateWorkInProgressHook() {
 *   // 直接取 next，不做任何检查
 *   currentHook = currentHook.next;
 *   // ...
 * }
 * ```
 */

// ============================================================
// 5. 💡 面试题
// ============================================================

/**
 * 💡 Q1: Hooks 的实现原理是什么？
 *
 * A: Hooks 存储在 Fiber 节点的 memoizedState 上，
 *    是一个链表结构。
 *
 *    - 每次调用 Hook 会创建一个 Hook 对象
 *    - 多个 Hook 通过 next 连接成链表
 *    - 更新时按顺序遍历链表获取状态
 *
 * 💡 Q2: 为什么 Hooks 不能放在条件语句中？
 *
 * A: 因为 Hooks 是链表，通过调用顺序匹配。
 *    如果放在条件语句中，可能导致顺序不一致，
 *    从而取到错误的状态。
 *
 * 💡 Q3: useState 的 dispatch 是同步还是异步？
 *
 * A: React 18 中，dispatch 总是异步的（批量更新）。
 *    多次调用 dispatch 会合并成一次更新。
 *
 * 💡 Q4: useEffect 和 useLayoutEffect 的区别？
 *
 * A: 执行时机不同：
 *    - useLayoutEffect：DOM 更新后同步执行，会阻塞渲染
 *    - useEffect：DOM 更新后异步执行，不阻塞渲染
 *
 *    使用场景：
 *    - useLayoutEffect：需要同步读取/修改 DOM
 *    - useEffect：大多数副作用（数据获取、订阅等）
 *
 * 💡 Q5: useEffect 的依赖数组是如何比较的？
 *
 * A: 使用 Object.is 浅比较每个依赖项。
 *    所以对象引用变了就会重新执行。
 */

// ============================================================
// 6. 🏢 实际开发应用
// ============================================================

/**
 * 🏢 应用 1：理解闭包陷阱
 *
 * 问题代码：
 * ```jsx
 * function Counter() {
 *   const [count, setCount] = useState(0);
 *
 *   useEffect(() => {
 *     const timer = setInterval(() => {
 *       console.log(count); // 始终是 0
 *     }, 1000);
 *     return () => clearInterval(timer);
 *   }, []); // 依赖为空
 * }
 * ```
 *
 * 原因：useEffect 的回调在 mount 时创建，
 *       闭包捕获了当时的 count 值（0）
 *
 * 解决方案：
 * ```jsx
 * // 方案 1：添加依赖
 * useEffect(() => { ... }, [count]);
 *
 * // 方案 2：使用 useRef
 * const countRef = useRef(count);
 * countRef.current = count;
 *
 * // 方案 3：使用函数式更新
 * setCount(c => c + 1);
 * ```
 */

/**
 * 🏢 应用 2：理解批量更新
 *
 * ```jsx
 * function handleClick() {
 *   setCount(c => c + 1);
 *   setCount(c => c + 1);
 *   setCount(c => c + 1);
 *   // React 18: 只触发一次渲染，count +3
 * }
 * ```
 *
 * 原理：多次 dispatch 会加入同一个更新队列，
 *       一次性计算所有更新
 */

/**
 * 🏢 应用 3：自定义 Hook
 *
 * 理解 Hooks 链表后，自定义 Hook 就是
 * 在链表中插入多个 Hook 节点
 */

// ============================================================
// 7. 📖 源码阅读指南
// ============================================================

/**
 * 📖 阅读顺序：
 *
 * 1. packages/react-reconciler/src/ReactFiberHooks.js
 *    - Hook 类型定义
 *    - renderWithHooks（入口）
 *    - mountState / updateState
 *    - mountEffect / updateEffect
 *    - dispatchSetState
 *
 * 2. packages/react-reconciler/src/ReactFiberFlags.js
 *    - Effect 相关的 Flags
 *
 * 3. packages/react/src/ReactHooks.js
 *    - useState / useEffect 的 API 入口
 */

// ============================================================
// 8. ✅ 学习检查
// ============================================================

/**
 * ✅ 完成以下任务：
 *
 * - [ ] 理解 Hooks 链表结构
 * - [ ] 理解 useState 的实现
 * - [ ] 理解 useEffect 的执行时机
 * - [ ] 理解 Hooks 规则的原因
 * - [ ] 能手写简化版 useState
 * - [ ] 阅读源码：ReactFiberHooks.js
 */

export {
  mountState,
  updateState,
  mountEffect,
  updateEffect,
  dispatchSetState,
};

