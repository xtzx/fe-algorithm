/**
 * ============================================================
 * 📚 Phase 4: Hooks 原理 - Part 3: useState 与基础 Hooks 内部实现
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-reconciler/src/ReactFiberHooks.new.js
 *   - mountState (Line 1505)
 *   - updateState (Line 1532)
 *   - dispatchSetState (Line 2228)
 *   - mountWorkInProgressHook (Line 636)
 *   - updateWorkInProgressHook (Line 657)
 *
 * ⏱️ 预计时间：2-3 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: mountWorkInProgressHook - 创建 Hook 节点
// ============================================================

/**
 * 📊 mountWorkInProgressHook - 首次渲染时创建 Hook
 */

const mountWorkInProgressHookFn = `
📊 mountWorkInProgressHook - 创建 Hook 节点

源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js (Line 636)
═══════════════════════════════════════════════════════════════════════════════

function mountWorkInProgressHook(): Hook {
  // 1. 创建新的 Hook 对象
  const hook: Hook = {
    memoizedState: null,
    baseState: null,
    baseQueue: null,
    queue: null,
    next: null,
  };

  // 2. 将 Hook 加入链表
  if (workInProgressHook === null) {
    // ⭐ 第一个 Hook：挂到 Fiber.memoizedState
    currentlyRenderingFiber.memoizedState = workInProgressHook = hook;
  } else {
    // ⭐ 后续 Hook：追加到链表末尾
    workInProgressHook = workInProgressHook.next = hook;
  }

  return workInProgressHook;
}


图示:
═══════════════════════════════════════════════════════════════════════════════

调用第一个 Hook (useState):
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   workInProgressHook === null ?  → Yes                                      │
│                                                                             │
│   创建 Hook1:                                                               │
│   ┌─────────────────┐                                                       │
│   │ memoizedState   │                                                       │
│   │ next: null      │                                                       │
│   └─────────────────┘                                                       │
│            ↑                                                                │
│            │                                                                │
│   Fiber.memoizedState = workInProgressHook = Hook1                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

调用第二个 Hook (useMemo):
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   workInProgressHook === null ?  → No                                       │
│                                                                             │
│   创建 Hook2，追加到末尾:                                                   │
│   ┌─────────────────┐     ┌─────────────────┐                              │
│   │ Hook1           │────▶│ Hook2           │                              │
│   │ next: Hook2     │     │ next: null      │                              │
│   └─────────────────┘     └─────────────────┘                              │
│            ↑                       ↑                                        │
│            │                       │                                        │
│   Fiber.memoizedState     workInProgressHook                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: updateWorkInProgressHook - 复用 Hook 节点
// ============================================================

/**
 * 📊 updateWorkInProgressHook - 更新渲染时复用 Hook
 */

const updateWorkInProgressHookFn = `
📊 updateWorkInProgressHook - 更新渲染时复用 Hook

源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js (Line 657)
═══════════════════════════════════════════════════════════════════════════════

function updateWorkInProgressHook(): Hook {
  // 1. 找到 current Fiber 上对应的 Hook
  let nextCurrentHook: null | Hook;
  if (currentHook === null) {
    // 第一个 Hook，从 current.memoizedState 开始
    const current = currentlyRenderingFiber.alternate;
    if (current !== null) {
      nextCurrentHook = current.memoizedState;
    } else {
      nextCurrentHook = null;
    }
  } else {
    // 后续 Hook，取 currentHook.next
    nextCurrentHook = currentHook.next;
  }

  // 2. 检查 workInProgress 上是否已有 Hook（re-render 时可能有）
  let nextWorkInProgressHook: null | Hook;
  if (workInProgressHook === null) {
    nextWorkInProgressHook = currentlyRenderingFiber.memoizedState;
  } else {
    nextWorkInProgressHook = workInProgressHook.next;
  }

  if (nextWorkInProgressHook !== null) {
    // ⭐ 有现成的 Hook（re-render 情况），直接复用
    workInProgressHook = nextWorkInProgressHook;
    currentHook = nextCurrentHook;
  } else {
    // ⭐ 没有现成的，从 current 克隆

    // 关键检查：如果 nextCurrentHook 为 null，说明 Hook 数量变了！
    if (nextCurrentHook === null) {
      throw new Error('Rendered more hooks than during the previous render.');
    }

    currentHook = nextCurrentHook;

    // 克隆 Hook
    const newHook: Hook = {
      memoizedState: currentHook.memoizedState,
      baseState: currentHook.baseState,
      baseQueue: currentHook.baseQueue,
      queue: currentHook.queue,
      next: null,
    };

    // 加入链表
    if (workInProgressHook === null) {
      currentlyRenderingFiber.memoizedState = workInProgressHook = newHook;
    } else {
      workInProgressHook = workInProgressHook.next = newHook;
    }
  }

  return workInProgressHook;
}


关键指针变化图示:
═══════════════════════════════════════════════════════════════════════════════

更新渲染时，有两棵树对应的 Hook 链表:
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   current Fiber                         workInProgress Fiber                │
│   ┌─────────────────┐                   ┌─────────────────┐                │
│   │ memoizedState   │                   │ memoizedState   │                │
│   └────────┬────────┘                   └────────┬────────┘                │
│            │                                     │                          │
│            ▼                                     ▼                          │
│   ┌─────────────────┐                   ┌─────────────────┐                │
│   │ Hook1 (旧)      │    ─ 克隆 ─▶      │ Hook1 (新)      │                │
│   └────────┬────────┘                   └────────┬────────┘                │
│            │                                     │                          │
│            ▼                                     ▼                          │
│   ┌─────────────────┐                   ┌─────────────────┐                │
│   │ Hook2 (旧)      │    ─ 克隆 ─▶      │ Hook2 (新)      │                │
│   └────────┬────────┘                   └────────┬────────┘                │
│            │                                     │                          │
│            ▼                                     ▼                          │
│   ┌─────────────────┐                   ┌─────────────────┐                │
│   │ Hook3 (旧)      │    ─ 克隆 ─▶      │ Hook3 (新)      │                │
│   └─────────────────┘                   └─────────────────┘                │
│            ↑                                     ↑                          │
│       currentHook                        workInProgressHook                 │
│       (遍历旧链表)                       (构建新链表)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

每调用一个 Hook，两个指针同步向下移动！
`;

// ============================================================
// Part 3: mountState - 首次渲染的 useState
// ============================================================

/**
 * 📊 mountState - 首次渲染
 */

const mountStateFn = `
📊 mountState - 首次渲染的 useState

源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js (Line 1505)
═══════════════════════════════════════════════════════════════════════════════

function mountState<S>(
  initialState: (() => S) | S,
): [S, Dispatch<BasicStateAction<S>>] {
  // 1. 创建 Hook 节点
  const hook = mountWorkInProgressHook();

  // 2. 处理初始值（支持函数形式）
  if (typeof initialState === 'function') {
    initialState = initialState();  // 惰性初始化
  }

  // 3. 初始化 Hook 的状态
  hook.memoizedState = hook.baseState = initialState;

  // 4. 创建 UpdateQueue
  const queue: UpdateQueue<S, BasicStateAction<S>> = {
    pending: null,                      // 待处理的更新
    lanes: NoLanes,                     // 优先级
    dispatch: null,                     // setState 函数（稍后赋值）
    lastRenderedReducer: basicStateReducer,  // 内部使用的 reducer
    lastRenderedState: (initialState: any),  // 上次渲染的 state
  };
  hook.queue = queue;

  // 5. ⭐ 创建 dispatch 函数（就是 setState）
  const dispatch: Dispatch<BasicStateAction<S>> = (
    queue.dispatch = (dispatchSetState.bind(
      null,
      currentlyRenderingFiber,  // 绑定当前 Fiber
      queue,                    // 绑定 UpdateQueue
    ): any)
  );

  // 6. 返回 [state, setState]
  return [hook.memoizedState, dispatch];
}


图示:
═══════════════════════════════════════════════════════════════════════════════

const [count, setCount] = useState(0);

执行后的数据结构:
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Hook 对象:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ memoizedState: 0                    ← 当前状态值                     │  │
│   │ baseState: 0                        ← 基础状态                       │  │
│   │ baseQueue: null                     ← 跳过的更新                     │  │
│   │ queue: {                            ← UpdateQueue                    │  │
│   │   pending: null,                    ← 待处理的 Update                │  │
│   │   dispatch: setCount,               ← ⭐ setState 函数               │  │
│   │   lastRenderedReducer: basicStateReducer,                           │  │
│   │   lastRenderedState: 0,                                             │  │
│   │ }                                                                   │  │
│   │ next: null                          ← 下一个 Hook                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   返回: [0, setCount]                                                       │
│              ↑                                                              │
│              │                                                              │
│       dispatchSetState.bind(fiber, queue)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 4: updateState - 更新渲染的 useState
// ============================================================

/**
 * 📊 updateState - 更新渲染
 */

const updateStateFn = `
📊 updateState - 更新渲染的 useState

源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js (Line 1532)
═══════════════════════════════════════════════════════════════════════════════

// useState 的更新实现其实是调用 useReducer 的更新
function updateState<S>(
  initialState: (() => S) | S,
): [S, Dispatch<BasicStateAction<S>>] {
  return updateReducer(basicStateReducer, (initialState: any));
}

// basicStateReducer：useState 内部使用的 reducer
function basicStateReducer<S>(state: S, action: BasicStateAction<S>): S {
  // 如果 action 是函数，调用它获取新 state
  // 否则 action 就是新 state
  return typeof action === 'function' ? action(state) : action;
}


updateReducer 核心逻辑（简化）:
═══════════════════════════════════════════════════════════════════════════════

function updateReducer<S, A>(
  reducer: (S, A) => S,
  initialArg: S,
): [S, Dispatch<A>] {
  // 1. 获取当前 Hook
  const hook = updateWorkInProgressHook();
  const queue = hook.queue;

  // 2. 处理更新队列
  const pending = queue.pending;

  if (pending !== null) {
    // 3. 遍历环形链表，计算新的 state
    let first = pending.next;  // 第一个 Update
    let newState = hook.baseState;
    let update = first;

    do {
      // 应用每个 Update
      const action = update.action;
      newState = reducer(newState, action);
      update = update.next;
    } while (update !== first);

    // 4. 更新 Hook 的状态
    hook.memoizedState = newState;
    hook.baseState = newState;
    queue.pending = null;
  }

  // 5. 返回新的 state 和 dispatch
  const dispatch = queue.dispatch;
  return [hook.memoizedState, dispatch];
}


图示：处理更新队列
═══════════════════════════════════════════════════════════════════════════════

假设调用了 setCount(1) 和 setCount(prev => prev + 1):
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   更新前:                                                                   │
│   ─────────                                                                 │
│   hook.memoizedState = 0 (旧 state)                                         │
│                                                                             │
│   queue.pending ──────────────────────────────────┐                        │
│                                                   │                        │
│   ┌────────────────┐     ┌────────────────┐      │                        │
│   │ Update 1       │────▶│ Update 2       │──────┘                        │
│   │ action: 1      │     │ action: fn     │  ↑                            │
│   │ (直接赋值 1)   │     │ (prev+1)       │  │                            │
│   └────────────────┘     └────────────────┘  │                            │
│         ↑                                    │                            │
│         └────────────────────────────────────┘  (环形)                    │
│                                                                             │
│   处理过程:                                                                 │
│   ──────────                                                                │
│   1. newState = basicStateReducer(0, 1)     → newState = 1                 │
│   2. newState = basicStateReducer(1, fn)    → newState = fn(1) = 2         │
│                                                                             │
│   更新后:                                                                   │
│   ─────────                                                                 │
│   hook.memoizedState = 2 (新 state)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 5: dispatchSetState - setState 的实现
// ============================================================

/**
 * 📊 dispatchSetState - setState 做了什么
 */

const dispatchSetStateFn = `
📊 dispatchSetState - setState 的内部实现

源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js (Line 2228)
═══════════════════════════════════════════════════════════════════════════════

function dispatchSetState<S, A>(
  fiber: Fiber,           // 绑定的 Fiber
  queue: UpdateQueue<S, A>,  // 绑定的 UpdateQueue
  action: A,              // 传入的新值或更新函数
) {
  // 1. 获取更新优先级
  const lane = requestUpdateLane(fiber);

  // 2. 创建 Update 对象
  const update: Update<S, A> = {
    lane,
    action,
    hasEagerState: false,
    eagerState: null,
    next: (null: any),
  };

  // 3. 检查是否是渲染阶段的更新（特殊情况）
  if (isRenderPhaseUpdate(fiber)) {
    enqueueRenderPhaseUpdate(queue, update);
  } else {
    // 4. ⭐ 提前计算优化（Eager State）
    const alternate = fiber.alternate;
    if (
      fiber.lanes === NoLanes &&
      (alternate === null || alternate.lanes === NoLanes)
    ) {
      // 当前没有待处理的更新，可以提前计算
      const lastRenderedReducer = queue.lastRenderedReducer;
      if (lastRenderedReducer !== null) {
        try {
          const currentState: S = queue.lastRenderedState;
          const eagerState = lastRenderedReducer(currentState, action);

          // 保存提前计算的结果
          update.hasEagerState = true;
          update.eagerState = eagerState;

          // ⭐ 如果新旧 state 相同，跳过更新！
          if (is(eagerState, currentState)) {
            // 使用 Object.is 比较
            enqueueConcurrentHookUpdateAndEagerlyBailout(fiber, queue, update);
            return;  // 提前返回，不触发渲染
          }
        } catch (error) {
          // 忽略错误，render 阶段会重新抛出
        }
      }
    }

    // 5. 将 Update 加入队列
    const root = enqueueConcurrentHookUpdate(fiber, queue, update, lane);

    // 6. 调度更新
    if (root !== null) {
      scheduleUpdateOnFiber(root, fiber, lane, eventTime);
    }
  }
}


关键优化：Eager State（提前计算）
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   为什么需要 Eager State？                                                  │
│   ────────────────────────                                                  │
│                                                                             │
│   场景：setCount(count)  // count 没变                                      │
│                                                                             │
│   没有优化时:                                                               │
│   ────────────                                                              │
│   1. 创建 Update                                                            │
│   2. 调度更新                                                               │
│   3. 进入 render 阶段                                                       │
│   4. 处理 Update，发现 state 没变                                           │
│   5. bailout（跳过渲染）                                                    │
│   → 浪费了一次调度！                                                        │
│                                                                             │
│   有 Eager State 优化:                                                      │
│   ─────────────────────                                                     │
│   1. 创建 Update                                                            │
│   2. ⭐ 提前计算新 state                                                    │
│   3. ⭐ 发现新旧 state 相同                                                 │
│   4. ⭐ 直接返回，不调度更新                                                │
│   → 完全跳过！                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

代码示例:
─────────────────────────────────────────────────────────────────

const [count, setCount] = useState(0);

// 情况 1：会触发更新
setCount(1);          // 1 !== 0，需要更新

// 情况 2：不会触发更新（Eager State 优化）
setCount(count);      // count === count，跳过

// 情况 3：不会触发更新（函数形式也能优化）
setCount(prev => prev); // prev === prev，跳过
`;

// ============================================================
// Part 6: 其他基础 Hooks
// ============================================================

/**
 * 📊 其他基础 Hooks 的内部实现
 */

const otherBasicHooks = `
📊 其他基础 Hooks 的内部实现

useRef
═══════════════════════════════════════════════════════════════════════════════

📁 ReactFiberHooks.new.js (Line 1589, 1658)

function mountRef<T>(initialValue: T): {| current: T |} {
  const hook = mountWorkInProgressHook();
  const ref = { current: initialValue };
  hook.memoizedState = ref;  // 存储 ref 对象
  return ref;
}

function updateRef<T>(initialValue: T): {| current: T |} {
  const hook = updateWorkInProgressHook();
  return hook.memoizedState;  // 直接返回同一个 ref 对象
}

特点：
─────────────────────────────────────────────────────────────────
• memoizedState 存储 { current: value } 对象
• 更新时直接返回同一个对象引用（不变）
• 修改 ref.current 不会触发重新渲染


useMemo
═══════════════════════════════════════════════════════════════════════════════

📁 ReactFiberHooks.new.js (Line 1899, 1910)

function mountMemo<T>(
  nextCreate: () => T,
  deps: Array<mixed> | void | null,
): T {
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const nextValue = nextCreate();  // 执行计算函数
  hook.memoizedState = [nextValue, nextDeps];  // 存储 [值, 依赖]
  return nextValue;
}

function updateMemo<T>(
  nextCreate: () => T,
  deps: Array<mixed> | void | null,
): T {
  const hook = updateWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const prevState = hook.memoizedState;

  if (prevState !== null) {
    if (nextDeps !== null) {
      const prevDeps = prevState[1];
      // ⭐ 比较依赖是否变化
      if (areHookInputsEqual(nextDeps, prevDeps)) {
        return prevState[0];  // 依赖没变，返回缓存值
      }
    }
  }

  // 依赖变了，重新计算
  const nextValue = nextCreate();
  hook.memoizedState = [nextValue, nextDeps];
  return nextValue;
}

特点：
─────────────────────────────────────────────────────────────────
• memoizedState 存储 [计算结果, 依赖数组]
• 更新时比较依赖，没变则返回缓存值
• 使用 Object.is 逐项比较依赖


useCallback
═══════════════════════════════════════════════════════════════════════════════

function mountCallback<T>(callback: T, deps: Array<mixed> | void | null): T {
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  hook.memoizedState = [callback, nextDeps];  // 存储 [函数, 依赖]
  return callback;
}

function updateCallback<T>(callback: T, deps: Array<mixed> | void | null): T {
  const hook = updateWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const prevState = hook.memoizedState;

  if (prevState !== null) {
    if (nextDeps !== null) {
      const prevDeps = prevState[1];
      if (areHookInputsEqual(nextDeps, prevDeps)) {
        return prevState[0];  // 依赖没变，返回缓存的函数
      }
    }
  }

  hook.memoizedState = [callback, nextDeps];
  return callback;
}

特点：
─────────────────────────────────────────────────────────────────
• 与 useMemo 结构几乎相同
• 区别：useMemo 缓存计算结果，useCallback 缓存函数本身
• useCallback(fn, deps) 等价于 useMemo(() => fn, deps)


依赖比较函数:
═══════════════════════════════════════════════════════════════════════════════

function areHookInputsEqual(
  nextDeps: Array<mixed>,
  prevDeps: Array<mixed> | null,
): boolean {
  if (prevDeps === null) {
    return false;
  }

  for (let i = 0; i < prevDeps.length && i < nextDeps.length; i++) {
    // 使用 Object.is 比较每一项
    if (is(nextDeps[i], prevDeps[i])) {
      continue;
    }
    return false;
  }
  return true;
}
`;

// ============================================================
// Part 7: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 3 面试要点

Q1: mountWorkInProgressHook 和 updateWorkInProgressHook 有什么区别？
A: - mount：创建新的 Hook 节点，追加到链表末尾
   - update：从 current Fiber 克隆 Hook，同时移动 currentHook 和 workInProgressHook 指针

Q2: useState 内部是如何实现的？
A: useState 内部使用 useReducer，reducer 是 basicStateReducer。
   mountState 创建 Hook 和 UpdateQueue；
   updateState 处理更新队列，计算新 state。

Q3: 什么是 Eager State 优化？
A: setState 时提前计算新 state，如果和旧 state 相同（Object.is），
   直接跳过更新调度，避免不必要的 render。
   只有当前 Fiber 没有待处理更新时才能使用此优化。

Q4: setCount(1) 和 setCount(prev => prev + 1) 在内部有什么区别？
A: 创建的 Update.action 不同：
   - setCount(1): action = 1（直接值）
   - setCount(prev => prev + 1): action = fn（函数）
   处理时用 basicStateReducer：
   - 直接值：直接使用
   - 函数：调用 fn(prevState) 获取新值

Q5: useMemo 和 useCallback 的内部实现有什么区别？
A: 几乎相同，都是存储 [值, deps]：
   - useMemo：存储计算结果，需要执行 nextCreate()
   - useCallback：存储函数本身，直接存 callback
   useCallback(fn, deps) === useMemo(() => fn, deps)

Q6: useRef 为什么不会触发重新渲染？
A: useRef 返回的是同一个 { current } 对象引用。
   修改 ref.current 不会创建 Update，不会调用 scheduleUpdateOnFiber，
   所以不会触发渲染。

Q7: 为什么 "Rendered more hooks than during the previous render" 错误？
A: updateWorkInProgressHook 在遍历 current 链表时，
   如果 nextCurrentHook === null 说明 Hook 数量比上次多了。
   这意味着 Hook 调用顺序不一致（可能在条件中调用了 Hook）。
`;

export {
  mountWorkInProgressHookFn,
  updateWorkInProgressHookFn,
  mountStateFn,
  updateStateFn,
  dispatchSetStateFn,
  otherBasicHooks,
  interviewPoints,
};

