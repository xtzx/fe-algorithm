/**
 * ============================================================
 * 📚 Phase 4: Hooks 原理深度解析
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 Hooks 的数据结构（链表）
 * 2. 掌握 useState/useReducer 的实现原理
 * 3. 掌握 useEffect/useLayoutEffect 的实现原理
 * 4. 理解为什么 Hooks 不能在条件语句中调用
 * 5. 理解闭包陷阱的原因和解决方案
 *
 * 📁 核心源码位置：
 * - packages/react-reconciler/src/ReactFiberHooks.new.js  # Hooks 实现
 * - packages/react/src/ReactHooks.js                      # Hooks API
 * - packages/react/src/ReactCurrentDispatcher.js          # Dispatcher
 *
 * ⏱️ 预计时间：8-10 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐（最高！）
 */

// ============================================================
// Part 1: Hooks 架构概览
// ============================================================

/**
 * 📊 Hooks 调用流程
 */

const hooksCallFlow = `
┌─────────────────────────────────────────────────────────────────────────┐
│                        Hooks 调用流程                                   │
│                                                                         │
│   1. 用户调用 useState(0)                                               │
│         │                                                               │
│         ▼                                                               │
│   2. packages/react/src/ReactHooks.js                                   │
│      export function useState(initialState) {                           │
│        const dispatcher = resolveDispatcher();                          │
│        return dispatcher.useState(initialState);                        │
│      }                                                                  │
│         │                                                               │
│         ▼                                                               │
│   3. ReactCurrentDispatcher.current                                     │
│      ┌─────────────────────────────────────────────────────┐           │
│      │   根据当前阶段指向不同的 dispatcher                   │           │
│      │                                                     │           │
│      │   • mount 阶段: HooksDispatcherOnMount              │           │
│      │   • update 阶段: HooksDispatcherOnUpdate            │           │
│      │   • rerender 阶段: HooksDispatcherOnRerender        │           │
│      └─────────────────────────────────────────────────────┘           │
│         │                                                               │
│         ▼                                                               │
│   4. packages/react-reconciler/src/ReactFiberHooks.new.js              │
│      • mountState() / updateState() 实际执行                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
`;

/**
 * 📊 Dispatcher 切换机制
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 */

const dispatcherMechanism = `
📊 renderWithHooks 中的 Dispatcher 切换

function renderWithHooks(current, workInProgress, Component, props, ...) {
  // 1. 设置当前渲染的 Fiber
  currentlyRenderingFiber = workInProgress;

  // 2. 重置 Hooks 状态
  workInProgress.memoizedState = null;  // Hooks 链表将重新构建
  workInProgress.updateQueue = null;    // Effect 链表

  // 3. ⭐ 根据是否有 current 选择 dispatcher
  ReactCurrentDispatcher.current =
    current === null || current.memoizedState === null
      ? HooksDispatcherOnMount    // 首次渲染
      : HooksDispatcherOnUpdate;  // 更新渲染

  // 4. 调用组件函数
  let children = Component(props, secondArg);

  // 5. 处理 render phase update
  if (didScheduleRenderPhaseUpdateDuringThisPass) {
    // 重新渲染...
  }

  // 6. 重置 dispatcher 为无效状态（防止在组件外调用）
  ReactCurrentDispatcher.current = ContextOnlyDispatcher;

  return children;
}
`;

// ============================================================
// Part 2: Hooks 数据结构
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 *
 * Hook 数据结构（第 148-154 行）
 */

// Hook 节点结构
interface Hook {
  /**
   * 存储的状态/值
   * - useState: state 值
   * - useReducer: state 值
   * - useEffect: Effect 对象
   * - useRef: { current: value }
   * - useMemo: [memoizedValue, deps]
   * - useCallback: [callback, deps]
   */
  memoizedState: any;

  /**
   * 基础状态（用于并发更新计算）
   */
  baseState: any;

  /**
   * 基础更新队列（跳过的低优先级更新）
   */
  baseQueue: Update<any, any> | null;

  /**
   * 更新队列
   * - useState/useReducer: UpdateQueue
   * - useEffect: Effect 环形链表
   */
  queue: any;

  /**
   * ⭐ 指向下一个 Hook（链表结构）
   */
  next: Hook | null;
}

// Update 结构（用于 useState/useReducer）
interface Update<S, A> {
  lane: Lane;              // 优先级
  action: A;               // 更新动作（值或函数）
  hasEagerState: boolean;  // 是否有急切计算的状态
  eagerState: S | null;    // 急切计算的状态值
  next: Update<S, A>;      // 下一个 Update（环形链表）
}

// UpdateQueue 结构
interface UpdateQueue<S, A> {
  pending: Update<S, A> | null;         // 待处理的更新（环形链表）
  lanes: Lanes;                         // 更新优先级
  dispatch: ((A) => void) | null;       // dispatch 函数
  lastRenderedReducer: ((S, A) => S) | null;  // 上次使用的 reducer
  lastRenderedState: S | null;          // 上次渲染的 state
}

// Effect 结构（用于 useEffect/useLayoutEffect）
interface Effect {
  tag: HookFlags;                       // 标记（Passive/Layout/Insertion）
  create: () => (() => void) | void;    // 创建函数
  destroy: (() => void) | void;         // 销毁函数
  deps: Array<any> | null;              // 依赖数组
  next: Effect;                         // 下一个 Effect（环形链表）
}

type Lane = number;
type Lanes = number;
type HookFlags = number;

/**
 * 📊 Hooks 链表存储位置
 */

const hooksStorageLocation = `
📊 Hooks 存储在 Fiber.memoizedState 上

Fiber {
  memoizedState: Hook1 → Hook2 → Hook3 → null
                  │        │        │
                  │        │        └── useEffect
                  │        └── useRef
                  └── useState
}

示例组件:
function Counter() {
  const [count, setCount] = useState(0);    // Hook1
  const ref = useRef(null);                 // Hook2
  useEffect(() => { ... }, [count]);        // Hook3
  return <div ref={ref}>{count}</div>;
}

对应的 Hooks 链表:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Hook1 (useState)│ ──► │ Hook2 (useRef)  │ ──► │ Hook3 (useEffect)│
│                 │     │                 │     │                 │
│ memoizedState:0 │     │ memoizedState:  │     │ memoizedState:  │
│ queue: {...}    │     │ {current:null}  │     │ Effect {...}    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
`;

// ============================================================
// Part 3: useState 实现原理
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 *             mountState (第 1505-1529 行)
 *             updateState (第 1532-1535 行)
 */

/**
 * 📊 mountState - 首次渲染时调用
 */

// 简化版 mountState
function mountStateSimplified<S>(initialState: (() => S) | S): [S, (action: S | ((prevState: S) => S)) => void] {
  // 1. 创建 Hook 节点并添加到链表
  const hook = mountWorkInProgressHook();

  // 2. 处理初始值（支持函数形式）
  if (typeof initialState === 'function') {
    initialState = (initialState as () => S)();
  }

  // 3. 存储初始状态
  hook.memoizedState = hook.baseState = initialState;

  // 4. 创建更新队列
  const queue: UpdateQueue<S, any> = {
    pending: null,
    lanes: 0,
    dispatch: null,
    lastRenderedReducer: basicStateReducer,  // (state, action) => typeof action === 'function' ? action(state) : action
    lastRenderedState: initialState,
  };
  hook.queue = queue;

  // 5. ⭐ 绑定 dispatch 函数
  const dispatch = dispatchSetState.bind(null, currentlyRenderingFiber, queue);
  queue.dispatch = dispatch;

  // 6. 返回 [state, setState]
  return [hook.memoizedState, dispatch];
}

// basicStateReducer - useState 使用的 reducer
function basicStateReducer<S>(state: S, action: S | ((prevState: S) => S)): S {
  return typeof action === 'function'
    ? (action as (prevState: S) => S)(state)
    : action;
}

/**
 * 📊 mountWorkInProgressHook - 创建并链接 Hook 节点
 */

let currentlyRenderingFiber: Fiber = null as any;
let workInProgressHook: Hook | null = null;

function mountWorkInProgressHook(): Hook {
  const hook: Hook = {
    memoizedState: null,
    baseState: null,
    baseQueue: null,
    queue: null,
    next: null,
  };

  if (workInProgressHook === null) {
    // 第一个 Hook
    currentlyRenderingFiber.memoizedState = workInProgressHook = hook;
  } else {
    // 添加到链表尾部
    workInProgressHook = workInProgressHook.next = hook;
  }

  return workInProgressHook;
}

/**
 * 📊 updateState - 更新渲染时调用
 */

// updateState 实际上调用 updateReducer
function updateStateSimplified<S>(initialState: S): [S, (action: S | ((prevState: S) => S)) => void] {
  return updateReducerSimplified(basicStateReducer, initialState);
}

let currentHook: Hook | null = null;

// 简化版 updateReducer
function updateReducerSimplified<S, A>(
  reducer: (state: S, action: A) => S,
  initialArg: S
): [S, (action: A) => void] {
  // 1. 获取当前 Hook（从 current Fiber 复制）
  const hook = updateWorkInProgressHook();
  const queue = hook.queue;

  // 2. 处理更新队列
  const pending = queue.pending;
  let baseState = hook.baseState;

  if (pending !== null) {
    // 遍历环形链表，计算新状态
    let first = pending.next;
    let update = first;
    let newState = baseState;

    do {
      newState = reducer(newState, update.action);
      update = update.next;
    } while (update !== first);

    hook.memoizedState = newState;
    hook.baseState = newState;
    queue.pending = null;
  }

  const dispatch = queue.dispatch;
  return [hook.memoizedState, dispatch];
}

/**
 * 📊 updateWorkInProgressHook - 更新时获取对应的 Hook
 */

function updateWorkInProgressHook(): Hook {
  // 从 current 树获取对应的 Hook
  let nextCurrentHook: Hook | null;

  if (currentHook === null) {
    // 第一个 Hook
    const current = currentlyRenderingFiber.alternate;
    nextCurrentHook = current !== null ? current.memoizedState : null;
  } else {
    // 后续 Hook
    nextCurrentHook = currentHook.next;
  }

  // ⭐ 这就是为什么 Hooks 不能在条件语句中！
  // 如果 nextCurrentHook 为 null，说明 Hook 数量不匹配
  if (nextCurrentHook === null) {
    throw new Error('Rendered more hooks than during the previous render.');
  }

  currentHook = nextCurrentHook;

  // 复制到 workInProgress
  const newHook: Hook = {
    memoizedState: currentHook.memoizedState,
    baseState: currentHook.baseState,
    baseQueue: currentHook.baseQueue,
    queue: currentHook.queue,
    next: null,
  };

  if (workInProgressHook === null) {
    currentlyRenderingFiber.memoizedState = workInProgressHook = newHook;
  } else {
    workInProgressHook = workInProgressHook.next = newHook;
  }

  return workInProgressHook;
}

// ============================================================
// Part 4: dispatchSetState - setState 触发更新
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 *             dispatchSetState (第 2228-2300 行)
 */

/**
 * 📊 dispatchSetState 流程图
 */

const dispatchSetStateFlow = `
📊 setCount(1) 触发的流程

dispatchSetState(fiber, queue, action)
    │
    ├── 1. 获取更新优先级
    │      const lane = requestUpdateLane(fiber);
    │
    ├── 2. 创建 Update 对象
    │      const update = {
    │        lane,
    │        action: 1,  // 或 (prev) => prev + 1
    │        hasEagerState: false,
    │        eagerState: null,
    │        next: null
    │      };
    │
    ├── 3. ⭐ 急切计算（eagerState 优化）
    │      │
    │      │  条件：当前队列为空
    │      │  fiber.lanes === NoLanes
    │      │
    │      ├── 提前计算新状态
    │      │   const eagerState = reducer(currentState, action);
    │      │   update.hasEagerState = true;
    │      │   update.eagerState = eagerState;
    │      │
    │      └── 如果新状态 === 旧状态
    │          Object.is(eagerState, currentState)
    │          → 直接返回，不触发更新！（bailout 优化）
    │
    ├── 4. 将 Update 加入队列（环形链表）
    │      if (queue.pending === null) {
    │        update.next = update;  // 自己指向自己
    │      } else {
    │        update.next = queue.pending.next;
    │        queue.pending.next = update;
    │      }
    │      queue.pending = update;
    │
    └── 5. 调度更新
           scheduleUpdateOnFiber(fiber, lane)
`;

// 简化版 dispatchSetState
function dispatchSetStateSimplified<S, A>(
  fiber: Fiber,
  queue: UpdateQueue<S, A>,
  action: A
): void {
  // 1. 获取优先级
  const lane = requestUpdateLane(fiber);

  // 2. 创建 Update
  const update: Update<S, A> = {
    lane,
    action,
    hasEagerState: false,
    eagerState: null as any,
    next: null as any,
  };

  // 3. ⭐ eagerState 优化
  if (fiber.lanes === NoLanes) {
    const currentState = queue.lastRenderedState;
    const eagerState = queue.lastRenderedReducer!(currentState!, action);
    update.hasEagerState = true;
    update.eagerState = eagerState;

    // 如果状态没变，直接返回
    if (Object.is(eagerState, currentState)) {
      // Bailout - 不触发重新渲染
      return;
    }
  }

  // 4. 加入更新队列
  enqueueUpdate(queue, update);

  // 5. 调度更新
  scheduleUpdateOnFiber(fiber, lane);
}

// ============================================================
// Part 5: useEffect 实现原理
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 *             mountEffect (第 1702-1725 行)
 *             updateEffect (第 1727-1739 行)
 */

/**
 * 📊 Effect 数据结构
 */

const effectDataStructure = `
📊 useEffect 的 Effect 结构

useEffect(() => {
  console.log('effect');
  return () => console.log('cleanup');
}, [count]);

创建的 Effect 对象:
{
  tag: HookPassive | HookHasEffect,  // 标记类型和是否需要执行
  create: () => { console.log('effect'); return cleanup; },
  destroy: cleanup,                   // 上次的清理函数
  deps: [count],                      // 依赖数组
  next: nextEffect                    // 环形链表
}

存储位置:
- Hook.memoizedState = Effect 对象
- Fiber.updateQueue.lastEffect = Effect 环形链表（所有 Effect）
`;

/**
 * 📊 mountEffect - 首次渲染
 */

// HookFlags
const HookHasEffect = 0b0001;  // 本次渲染需要执行
const HookPassive = 0b0100;    // useEffect
const HookLayout = 0b0010;     // useLayoutEffect
const HookInsertion = 0b1000;  // useInsertionEffect

// FiberFlags
const PassiveEffect = 0b00000000000000100000000000;
const UpdateEffect = 0b00000000000000000000000100;

// 简化版 mountEffect
function mountEffectSimplified(
  create: () => (() => void) | void,
  deps: Array<any> | void | null
): void {
  // 1. 创建 Hook
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;

  // 2. 标记 Fiber 有 Passive Effect
  currentlyRenderingFiber.flags |= PassiveEffect;

  // 3. 创建 Effect 并存储
  hook.memoizedState = pushEffect(
    HookHasEffect | HookPassive,  // 首次渲染一定执行
    create,
    undefined,  // 首次没有 destroy
    nextDeps
  );
}

// pushEffect - 创建 Effect 并加入环形链表
function pushEffect(
  tag: HookFlags,
  create: () => (() => void) | void,
  destroy: (() => void) | void,
  deps: Array<any> | null
): Effect {
  const effect: Effect = {
    tag,
    create,
    destroy,
    deps,
    next: null as any,
  };

  // 获取或创建 updateQueue
  let componentUpdateQueue = currentlyRenderingFiber.updateQueue;
  if (componentUpdateQueue === null) {
    componentUpdateQueue = { lastEffect: null, stores: null };
    currentlyRenderingFiber.updateQueue = componentUpdateQueue;
    componentUpdateQueue.lastEffect = effect.next = effect;  // 环形链表
  } else {
    const lastEffect = componentUpdateQueue.lastEffect;
    if (lastEffect === null) {
      componentUpdateQueue.lastEffect = effect.next = effect;
    } else {
      // 插入到环形链表
      const firstEffect = lastEffect.next;
      lastEffect.next = effect;
      effect.next = firstEffect;
      componentUpdateQueue.lastEffect = effect;
    }
  }

  return effect;
}

/**
 * 📊 updateEffect - 更新渲染
 */

// 简化版 updateEffect
function updateEffectSimplified(
  create: () => (() => void) | void,
  deps: Array<any> | void | null
): void {
  const hook = updateWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  let destroy: (() => void) | void = undefined;

  if (currentHook !== null) {
    const prevEffect = currentHook.memoizedState;
    destroy = prevEffect.destroy;

    if (nextDeps !== null) {
      const prevDeps = prevEffect.deps;

      // ⭐ 比较依赖数组
      if (areHookInputsEqual(nextDeps, prevDeps)) {
        // 依赖没变，不需要执行
        // 但仍然要创建 Effect（为了保持链表结构）
        hook.memoizedState = pushEffect(
          HookPassive,  // 没有 HookHasEffect
          create,
          destroy,
          nextDeps
        );
        return;
      }
    }
  }

  // 依赖变了，需要执行
  currentlyRenderingFiber.flags |= PassiveEffect;
  hook.memoizedState = pushEffect(
    HookHasEffect | HookPassive,
    create,
    destroy,
    nextDeps
  );
}

// 比较依赖数组
function areHookInputsEqual(
  nextDeps: Array<any>,
  prevDeps: Array<any> | null
): boolean {
  if (prevDeps === null) {
    return false;
  }

  for (let i = 0; i < prevDeps.length && i < nextDeps.length; i++) {
    if (Object.is(nextDeps[i], prevDeps[i])) {
      continue;
    }
    return false;
  }
  return true;
}

/**
 * 📊 useEffect 执行时机
 */

const useEffectExecutionTiming = `
📊 useEffect vs useLayoutEffect 执行时机

┌─────────────────────────────────────────────────────────────────────────┐
│                         Commit 阶段                                     │
│                                                                         │
│  ┌──────────────────┐                                                  │
│  │ Before Mutation  │                                                  │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │    Mutation      │  DOM 操作                                        │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           │ ← root.current = finishedWork（切换 Fiber 树）              │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │     Layout       │  ⭐ useLayoutEffect 执行（同步！）               │
│  │                  │     componentDidMount/Update                     │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  浏览器渲染 ──────────────────────────────────────────────              │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ Passive Effects  │  ⭐ useEffect 执行（异步！）                      │
│  │                  │     通过 Scheduler 调度                          │
│  └──────────────────┘                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

执行顺序:
1. useInsertionEffect（CSS-in-JS 用）
2. DOM 操作
3. useLayoutEffect
4. 浏览器绘制
5. useEffect

useEffect 执行:
flushPassiveEffects()
├── commitPassiveUnmountEffects()  // 先执行所有 destroy
│   └── effect.destroy()
│
└── commitPassiveMountEffects()    // 再执行所有 create
    └── effect.create()
`;

// ============================================================
// Part 6: 其他 Hooks 实现
// ============================================================

/**
 * 📊 useRef 实现
 */

function mountRefSimplified<T>(initialValue: T): { current: T } {
  const hook = mountWorkInProgressHook();
  const ref = { current: initialValue };
  hook.memoizedState = ref;
  return ref;
}

function updateRefSimplified<T>(initialValue: T): { current: T } {
  const hook = updateWorkInProgressHook();
  return hook.memoizedState;  // 直接返回，不做任何处理
}

/**
 * 📊 useMemo 实现
 */

function mountMemoSimplified<T>(
  nextCreate: () => T,
  deps: Array<any> | void | null
): T {
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const nextValue = nextCreate();  // 执行计算
  hook.memoizedState = [nextValue, nextDeps];  // 存储值和依赖
  return nextValue;
}

function updateMemoSimplified<T>(
  nextCreate: () => T,
  deps: Array<any> | void | null
): T {
  const hook = updateWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const prevState = hook.memoizedState;

  if (prevState !== null && nextDeps !== null) {
    const prevDeps = prevState[1];
    if (areHookInputsEqual(nextDeps, prevDeps)) {
      // 依赖没变，返回缓存值
      return prevState[0];
    }
  }

  // 依赖变了，重新计算
  const nextValue = nextCreate();
  hook.memoizedState = [nextValue, nextDeps];
  return nextValue;
}

/**
 * 📊 useCallback 实现
 */

function mountCallbackSimplified<T extends Function>(
  callback: T,
  deps: Array<any> | void | null
): T {
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  hook.memoizedState = [callback, nextDeps];
  return callback;
}

function updateCallbackSimplified<T extends Function>(
  callback: T,
  deps: Array<any> | void | null
): T {
  const hook = updateWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const prevState = hook.memoizedState;

  if (prevState !== null && nextDeps !== null) {
    const prevDeps = prevState[1];
    if (areHookInputsEqual(nextDeps, prevDeps)) {
      return prevState[0];  // 返回缓存的函数
    }
  }

  hook.memoizedState = [callback, nextDeps];
  return callback;
}

// ============================================================
// Part 7: 为什么 Hooks 不能条件调用
// ============================================================

/**
 * 📊 条件调用导致的问题
 */

const whyNoConditionalHooks = `
📊 为什么 Hooks 不能在条件语句中调用

问题场景:
function Component({ showExtra }) {
  const [count, setCount] = useState(0);   // Hook1

  if (showExtra) {
    const [extra, setExtra] = useState(''); // Hook2（条件调用）
  }

  useEffect(() => { ... }, [count]);        // Hook3
}

第一次渲染 (showExtra = true):
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Hook1  │ ──► │  Hook2  │ ──► │  Hook3  │
│ useState│     │ useState│     │useEffect│
│ count=0 │     │ extra=''│     │         │
└─────────┘     └─────────┘     └─────────┘

第二次渲染 (showExtra = false):
调用顺序: useState(0) → useEffect()

但链表期望: Hook1 → Hook2 → Hook3
实际调用:   Hook1 → Hook3

updateWorkInProgressHook() 内部:
- 第一个 useState 匹配 Hook1 ✓
- useEffect 应该匹配 Hook3
- 但按顺序取的是 Hook2!

结果: useEffect 取到了 useState 的状态!
     → 类型不匹配 → 崩溃或奇怪行为

⚠️ 这就是为什么 React 要求 Hooks 必须在顶层调用！
`;

// ============================================================
// Part 8: 闭包陷阱
// ============================================================

/**
 * 📊 闭包陷阱详解
 */

const closureTrapExplanation = `
📊 闭包陷阱

问题代码:
function Counter() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      console.log(count);  // 永远是 0！
      setCount(count + 1); // 永远设置为 1！
    }, 1000);
    return () => clearInterval(timer);
  }, []);  // 空依赖数组
}

问题分析:
第一次渲染 (count = 0):
  - useEffect 创建，捕获 count = 0
  - 依赖 [] 不变，effect 不会重新创建
  - setInterval 里的 count 永远是 0

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   渲染1: count = 0                                          │
│          │                                                  │
│          └──► useEffect 创建                                │
│               │                                            │
│               └──► setInterval 闭包捕获 count = 0           │
│                                                             │
│   渲染2: count = 1（但 setInterval 里的 count 还是 0）       │
│                                                             │
│   渲染3: count 应该是 2，但因为 setCount(0+1)，还是 1        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

解决方案:

方案1: 使用函数式更新
useEffect(() => {
  const timer = setInterval(() => {
    setCount(prev => prev + 1);  // ✅ 使用函数获取最新值
  }, 1000);
  return () => clearInterval(timer);
}, []);

方案2: 添加依赖
useEffect(() => {
  const timer = setInterval(() => {
    console.log(count);
    setCount(count + 1);
  }, 1000);
  return () => clearInterval(timer);
}, [count]);  // ✅ 但会不断重建 timer

方案3: 使用 useRef
const countRef = useRef(count);
countRef.current = count;  // 每次渲染更新

useEffect(() => {
  const timer = setInterval(() => {
    console.log(countRef.current);  // ✅ 总是最新值
    setCount(prev => prev + 1);
  }, 1000);
  return () => clearInterval(timer);
}, []);
`;

// ============================================================
// Part 9: 面试题
// ============================================================

const interviewQuestions = `
💡 Q1: Hooks 的数据结构是什么？存储在哪里？
A: Hooks 是一个单向链表，存储在 Fiber.memoizedState 上。
   每个 Hook 节点包含：memoizedState、baseState、queue、next。
   每次渲染时按顺序遍历链表获取对应的 Hook。

💡 Q2: 为什么 Hooks 不能在条件语句中调用？
A: 因为 Hooks 是链表结构，按调用顺序存储和读取。
   条件调用会导致链表顺序不一致：
   - 第一次渲染：Hook1 → Hook2 → Hook3
   - 第二次渲染（跳过 Hook2）：Hook1 → Hook3
   - 但读取时按顺序取，Hook3 会取到 Hook2 的状态！

💡 Q3: useState 和 useReducer 有什么关系？
A: useState 是 useReducer 的语法糖。
   useState 内部调用 useReducer，使用 basicStateReducer：
   const basicStateReducer = (state, action) =>
     typeof action === 'function' ? action(state) : action;

💡 Q4: 什么是 eagerState 优化？
A: 当调用 setState 时，如果更新队列为空：
   1. 立即计算新状态（不等到渲染阶段）
   2. 如果新状态 === 旧状态（Object.is 比较）
   3. 直接返回，不触发重新渲染
   这样可以避免不必要的渲染。

💡 Q5: useEffect 和 useLayoutEffect 有什么区别？
A: 执行时机不同：
   - useLayoutEffect：DOM 更新后、浏览器绘制前（同步）
   - useEffect：浏览器绘制后（异步，通过 Scheduler）

   使用场景：
   - useLayoutEffect：需要同步读取/修改 DOM
   - useEffect：大多数副作用（数据请求、订阅等）

💡 Q6: 什么是闭包陷阱？如何解决？
A: 闭包陷阱：effect 捕获了旧的 state/props 值。
   解决方案：
   1. 函数式更新：setState(prev => prev + 1)
   2. 添加依赖：[count]
   3. 使用 useRef：ref.current 总是最新值

💡 Q7: 为什么 useRef 的值变化不会触发重新渲染？
A: useRef 返回的是一个普通对象 { current: value }。
   修改 ref.current 只是修改对象属性，不会触发任何更新。
   React 不会追踪 ref.current 的变化。

💡 Q8: useMemo 和 useCallback 的区别？
A: 都是用于缓存，但缓存的内容不同：
   - useMemo：缓存计算结果，返回值
   - useCallback：缓存函数引用，返回函数

   useCallback(fn, deps) 等价于 useMemo(() => fn, deps)

💡 Q9: 空依赖 [] 和不传依赖有什么区别？
A: - 空依赖 []：只在 mount 时执行一次
   - 不传依赖：每次渲染都执行

   因为 updateEffect 中：
   - deps === null 时不比较，直接标记需要执行
   - deps === [] 时比较结果为 true，不执行

💡 Q10: React 如何区分 mount 和 update？
A: 通过 Dispatcher 机制：
   - mount 阶段：ReactCurrentDispatcher.current = HooksDispatcherOnMount
   - update 阶段：ReactCurrentDispatcher.current = HooksDispatcherOnUpdate

   区分条件：current === null || current.memoizedState === null
`;

// ============================================================
// Part 10: 实践练习
// ============================================================

/**
 * 练习 1：实现简化版 useState
 */
function useStateSimple<S>(initialState: S): [S, (action: S | ((prev: S) => S)) => void] {
  // 获取或创建 Hook
  // 返回 [state, setState]
  // setState 调用后触发重新渲染
  return [initialState, () => {}]; // 实现...
}

/**
 * 练习 2：实现简化版 useEffect
 */
function useEffectSimple(
  create: () => (() => void) | void,
  deps?: Array<any>
): void {
  // 比较依赖
  // 如果依赖变化，标记需要执行
  // 存储 Effect 对象
}

/**
 * 练习 3：理解闭包陷阱
 */
function useCounterWithTrap() {
  // 修复下面代码的闭包陷阱
  // const [count, setCount] = useState(0);
  // useEffect(() => {
  //   const timer = setInterval(() => {
  //     setCount(count + 1);  // 闭包陷阱！
  //   }, 1000);
  //   return () => clearInterval(timer);
  // }, []);
}

// 类型定义
interface Fiber {
  memoizedState: any;
  updateQueue: any;
  flags: number;
  alternate: Fiber | null;
  lanes: number;
}

const NoLanes = 0;

declare function requestUpdateLane(fiber: Fiber): number;
declare function scheduleUpdateOnFiber(fiber: Fiber, lane: number): void;
declare function enqueueUpdate<S, A>(queue: UpdateQueue<S, A>, update: Update<S, A>): void;

// ============================================================
// 学习检查清单
// ============================================================

/**
 * ✅ Phase 4 学习检查
 *
 * 数据结构：
 * - [ ] 能画出 Hooks 链表结构
 * - [ ] 理解 Hook、Update、Effect 的数据结构
 * - [ ] 理解 Hooks 存储在 Fiber.memoizedState
 *
 * useState：
 * - [ ] 理解 mountState 和 updateState 的区别
 * - [ ] 理解 dispatchSetState 的流程
 * - [ ] 理解 eagerState 优化
 *
 * useEffect：
 * - [ ] 理解 Effect 环形链表
 * - [ ] 理解依赖比较机制
 * - [ ] 理解执行时机（与 useLayoutEffect 区别）
 *
 * 陷阱理解：
 * - [ ] 能解释为什么不能条件调用 Hooks
 * - [ ] 能解释闭包陷阱并给出解决方案
 *
 * 源码位置：
 * - [ ] 能找到 ReactFiberHooks.new.js
 * - [ ] 能找到 HooksDispatcherOnMount/Update
 */

export {
  hooksCallFlow,
  dispatcherMechanism,
  hooksStorageLocation,
  dispatchSetStateFlow,
  effectDataStructure,
  useEffectExecutionTiming,
  whyNoConditionalHooks,
  closureTrapExplanation,
  interviewQuestions,
  mountStateSimplified,
  updateStateSimplified,
  dispatchSetStateSimplified,
  mountEffectSimplified,
  updateEffectSimplified,
  mountRefSimplified,
  updateRefSimplified,
  mountMemoSimplified,
  updateMemoSimplified,
  mountCallbackSimplified,
  updateCallbackSimplified,
  mountWorkInProgressHook,
  updateWorkInProgressHook,
  pushEffect,
  areHookInputsEqual,
  basicStateReducer,
};

