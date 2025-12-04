/**
 * ============================================================
 * 📚 React Hooks 原理
 * ============================================================
 *
 * 面试考察重点：
 * 1. Hooks 的设计思想
 * 2. useState、useEffect 原理
 * 3. Hooks 规则及原因
 * 4. 自定义 Hooks
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 为什么需要 Hooks？
 *
 * Class 组件的问题：
 * 1. 逻辑复用困难（HOC、render props 嵌套地狱）
 * 2. 生命周期拆分相关逻辑
 * 3. this 指向问题
 * 4. 难以理解和测试
 *
 * Hooks 的优势：
 * 1. 逻辑复用简单（自定义 Hooks）
 * 2. 相关逻辑放在一起
 * 3. 没有 this 问题
 * 4. 函数式，更简洁
 *
 * 📊 Hooks 的本质
 *
 * Hooks 是一个链表结构，每个 Hook 是链表的一个节点。
 * 组件每次渲染时，按顺序遍历链表，读取/更新对应的状态。
 */

// ============================================================
// 2. Hooks 底层实现
// ============================================================

/**
 * 📊 Hooks 链表结构
 *
 * Component
 *     │
 *     └── memoizedState (第一个 Hook)
 *            │
 *            └── next (第二个 Hook)
 *                  │
 *                  └── next (第三个 Hook)
 *                        │
 *                        └── null
 *
 * 每个 Hook 节点结构：
 * {
 *   memoizedState: 状态值 / effect 对象,
 *   queue: 更新队列,
 *   next: 下一个 Hook
 * }
 */

// 模拟 Hooks 实现
interface Hook {
  memoizedState: any;
  queue: any[];
  next: Hook | null;
}

interface Fiber {
  memoizedState: Hook | null;
  stateNode: any;
}

let currentFiber: Fiber | null = null;
let workInProgressHook: Hook | null = null;

// 设置当前 Fiber（模拟 React 内部行为）
function setCurrentFiber(fiber: Fiber) {
  currentFiber = fiber;
  workInProgressHook = null;
}

// 获取当前 Hook
function getCurrentHook(): Hook {
  if (!currentFiber) {
    throw new Error('Hooks must be called inside a component');
  }

  let hook: Hook;

  if (workInProgressHook === null) {
    // 第一个 Hook
    if (currentFiber.memoizedState === null) {
      // 首次渲染，创建新 Hook
      hook = {
        memoizedState: null,
        queue: [],
        next: null,
      };
      currentFiber.memoizedState = hook;
    } else {
      // 更新渲染，复用 Hook
      hook = currentFiber.memoizedState;
    }
  } else {
    // 后续 Hook
    if (workInProgressHook.next === null) {
      // 首次渲染，创建新 Hook
      hook = {
        memoizedState: null,
        queue: [],
        next: null,
      };
      workInProgressHook.next = hook;
    } else {
      // 更新渲染，复用 Hook
      hook = workInProgressHook.next;
    }
  }

  workInProgressHook = hook;
  return hook;
}

// ============================================================
// 3. useState 实现
// ============================================================

/**
 * 📊 useState 工作原理
 *
 * 1. 首次渲染：创建 Hook 节点，存储初始值
 * 2. setState：将更新加入队列，触发重渲染
 * 3. 后续渲染：从 Hook 节点读取状态，处理更新队列
 */

function useState<T>(initialState: T | (() => T)): [T, (action: T | ((prev: T) => T)) => void] {
  const hook = getCurrentHook();

  // 首次渲染
  if (hook.memoizedState === undefined) {
    hook.memoizedState = typeof initialState === 'function'
      ? (initialState as () => T)()
      : initialState;
  }

  // 处理更新队列
  hook.queue.forEach(action => {
    hook.memoizedState = typeof action === 'function'
      ? action(hook.memoizedState)
      : action;
  });
  hook.queue = [];

  // setState 函数
  const setState = (action: T | ((prev: T) => T)) => {
    hook.queue.push(action);
    // 触发重渲染（简化版）
    scheduleUpdate();
  };

  return [hook.memoizedState, setState];
}

// 模拟调度更新
function scheduleUpdate() {
  // 实际 React 会调度 Fiber 更新
  console.log('Schedule update');
}

// ============================================================
// 4. useEffect 实现
// ============================================================

/**
 * 📊 useEffect 工作原理
 *
 * 1. 首次渲染：创建 effect 对象，渲染后执行
 * 2. 后续渲染：对比依赖数组
 *    - 依赖变化：清理上一次 effect，执行新 effect
 *    - 依赖不变：跳过
 * 3. 卸载时：执行清理函数
 */

interface Effect {
  create: () => (() => void) | void;
  destroy: (() => void) | undefined;
  deps: any[] | undefined;
}

function useEffect(create: () => (() => void) | void, deps?: any[]) {
  const hook = getCurrentHook();

  const prevEffect = hook.memoizedState as Effect | null;
  
  // 判断依赖是否变化
  let hasChanged = true;
  if (prevEffect && deps !== undefined) {
    hasChanged = deps.some((dep, i) => !Object.is(dep, prevEffect.deps?.[i]));
  }

  if (hasChanged) {
    // 清理上一次的 effect
    if (prevEffect?.destroy) {
      prevEffect.destroy();
    }

    // 创建新的 effect
    const effect: Effect = {
      create,
      destroy: undefined,
      deps,
    };
    hook.memoizedState = effect;

    // 渲染后执行（简化版，实际是异步的）
    setTimeout(() => {
      effect.destroy = effect.create() || undefined;
    }, 0);
  }
}

// ============================================================
// 5. useRef 实现
// ============================================================

/**
 * 📊 useRef 特点
 *
 * - 返回一个可变的 ref 对象
 * - .current 属性可以存储任意值
 * - 修改不会触发重渲染
 * - 整个生命周期保持不变
 */

function useRef<T>(initialValue: T): { current: T } {
  const hook = getCurrentHook();

  if (hook.memoizedState === undefined) {
    hook.memoizedState = { current: initialValue };
  }

  return hook.memoizedState;
}

// ============================================================
// 6. useMemo / useCallback 实现
// ============================================================

/**
 * 📊 useMemo
 *
 * 缓存计算结果，依赖不变时返回缓存值
 */

function useMemo<T>(factory: () => T, deps: any[]): T {
  const hook = getCurrentHook();

  const prevDeps = hook.memoizedState?.[1];
  
  // 依赖不变，返回缓存值
  if (prevDeps && deps.every((dep, i) => Object.is(dep, prevDeps[i]))) {
    return hook.memoizedState[0];
  }

  // 依赖变化，重新计算
  const value = factory();
  hook.memoizedState = [value, deps];
  return value;
}

/**
 * 📊 useCallback
 *
 * 缓存函数引用，本质是 useMemo 的语法糖
 */

function useCallback<T extends Function>(callback: T, deps: any[]): T {
  return useMemo(() => callback, deps);
}

// ============================================================
// 7. ⚠️ Hooks 规则（重要！）
// ============================================================

/**
 * ⚠️ 两条规则
 *
 * 1. 只在最顶层使用 Hooks
 *    - ❌ 不能在条件语句中使用
 *    - ❌ 不能在循环中使用
 *    - ❌ 不能在嵌套函数中使用
 *
 * 2. 只在 React 函数中调用 Hooks
 *    - ✅ 函数组件
 *    - ✅ 自定义 Hooks
 *    - ❌ 普通函数
 *
 * 💡 为什么有这些规则？
 *
 * 因为 Hooks 是链表结构，按调用顺序存储。
 * 如果条件/循环导致顺序变化，Hook 就会对应到错误的状态。
 */

const hooksRulesExample = `
// ❌ 错误：条件调用
function Component({ condition }) {
  if (condition) {
    const [state, setState] = useState(0); // 可能跳过
  }
  const [name, setName] = useState(''); // 位置不稳定
}

// ❌ 错误：循环调用
function Component({ items }) {
  items.forEach(item => {
    const [value, setValue] = useState(item); // 数量不确定
  });
}

// ✅ 正确：顶层调用
function Component({ condition }) {
  const [state, setState] = useState(0);
  const [name, setName] = useState('');
  
  // 条件逻辑放在 Hook 之后
  if (condition) {
    // 使用 state
  }
}
`;

// ============================================================
// 8. 常见 Hooks 陷阱
// ============================================================

/**
 * ⚠️ 闭包陷阱
 *
 * Hooks 回调函数形成闭包，可能捕获旧的 state 值。
 */

const closureTrapExample = `
function Counter() {
  const [count, setCount] = useState(0);
  
  // ❌ 闭包陷阱：setTimeout 捕获的是旧的 count
  const handleClick = () => {
    setTimeout(() => {
      console.log(count); // 永远是点击时的值
    }, 1000);
  };
  
  // ✅ 解决 1：使用函数式更新
  const handleClick2 = () => {
    setTimeout(() => {
      setCount(prev => prev + 1); // 使用最新的 state
    }, 1000);
  };
  
  // ✅ 解决 2：使用 ref
  const countRef = useRef(count);
  countRef.current = count;
  
  const handleClick3 = () => {
    setTimeout(() => {
      console.log(countRef.current); // 最新值
    }, 1000);
  };
}
`;

/**
 * ⚠️ 依赖数组问题
 */

const depsArrayExample = `
// ❌ 依赖缺失
useEffect(() => {
  fetch('/api?id=' + id);
}, []); // 缺少 id 依赖

// ❌ 依赖过多
useEffect(() => {
  handleClick(); // handleClick 每次渲染都变
}, [handleClick]); // 导致 effect 每次都执行

// ✅ 解决：useCallback 包裹
const handleClick = useCallback(() => {
  // ...
}, [/* 真正的依赖 */]);

useEffect(() => {
  handleClick();
}, [handleClick]);

// ✅ 解决：把函数移到 effect 内部
useEffect(() => {
  function handleClick() {
    // ...
  }
  handleClick();
}, [id]); // 只依赖真正需要的
`;

// ============================================================
// 9. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 为什么 Hooks 不能在条件语句中使用？
 * A:
 *    - Hooks 按顺序存储在链表中
 *    - 条件语句会导致顺序不确定
 *    - 顺序变化会导致 Hook 对应错误的状态
 *
 * Q2: useEffect 和 useLayoutEffect 的区别？
 * A:
 *    useEffect：
 *    - 异步执行
 *    - 在浏览器绑定后执行
 *    - 不会阻塞渲染
 *
 *    useLayoutEffect：
 *    - 同步执行
 *    - 在 DOM 更新后、浏览器绑定前执行
 *    - 可能阻塞渲染
 *    - 适合读取 DOM 布局信息
 *
 * Q3: useState 的更新是同步还是异步？
 * A:
 *    - React 18 之前：事件处理中异步，setTimeout 中同步
 *    - React 18 之后：默认都是批量异步
 *    - 可以用 flushSync 强制同步
 *
 * Q4: useCallback 和 useMemo 的区别？
 * A:
 *    - useMemo：缓存值
 *    - useCallback：缓存函数
 *    - useCallback(fn, deps) 等价于 useMemo(() => fn, deps)
 *
 * Q5: 什么时候需要 useCallback？
 * A:
 *    - 传递给子组件的回调（配合 memo）
 *    - 作为 useEffect 的依赖
 *    - 不是所有函数都需要，过度优化反而有开销
 */

// ============================================================
// 10. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：自定义 Hook - useFetch
 */

const useFetchExample = `
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    
    setLoading(true);
    fetch(url)
      .then(res => res.json())
      .then(data => {
        if (!cancelled) {
          setData(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  return { data, loading, error };
}

// 使用
function UserProfile({ userId }) {
  const { data, loading, error } = useFetch(\`/api/users/\${userId}\`);
  // ...
}
`;

/**
 * 🏢 场景 2：自定义 Hook - useDebounce
 */

const useDebounceExample = `
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

// 使用：搜索框防抖
function SearchInput() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);
  
  useEffect(() => {
    if (debouncedQuery) {
      // 发起搜索请求
    }
  }, [debouncedQuery]);
}
`;

export {
  // Hooks 实现
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
  
  // 辅助函数
  setCurrentFiber,
  getCurrentHook,
  
  // 示例
  hooksRulesExample,
  closureTrapExample,
  depsArrayExample,
  useFetchExample,
  useDebounceExample,
};

export type { Hook, Fiber, Effect };

