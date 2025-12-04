/**
 * ============================================================
 * 📚 Phase 7: 并发特性（React 18 重点）
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解并发模式的概念
 * 2. 掌握 useTransition 原理
 * 3. 掌握 useDeferredValue 原理
 * 4. 理解自动批处理
 *
 * 📁 源码位置：
 * - packages/react-reconciler/src/ReactFiberWorkLoop.js
 * - packages/react/src/ReactHooks.js
 *
 * ⏱️ 预计时间：4 小时
 * 🔥 面试权重：⭐⭐⭐⭐（React 18 重点）
 */

// ============================================================
// 1. 并发模式概述
// ============================================================

/**
 * 📊 什么是并发模式？
 *
 * 并发不等于并行！
 *
 * ```
 * 并行（Parallel）：多个任务同时执行
 * ┌──────────┐     ┌──────────┐
 * │  任务 A   │     │  任务 B   │
 * └──────────┘     └──────────┘
 *     线程 1           线程 2
 *
 * 并发（Concurrent）：多个任务交替执行
 * ┌────┐ ┌────┐ ┌────┐ ┌────┐
 * │ A  │ │ B  │ │ A  │ │ B  │
 * └────┘ └────┘ └────┘ └────┘
 *            一个线程
 * ```
 *
 * React 并发模式：
 * - 渲染可以被中断
 * - 高优先级更新可以插队
 * - 低优先级更新可以延迟
 */

/**
 * 📊 并发特性总览
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    React 18 并发特性                            │
 * │                                                                 │
 * │  ┌─────────────────────┐  ┌─────────────────────┐              │
 * │  │   useTransition     │  │  useDeferredValue   │              │
 * │  │                     │  │                     │              │
 * │  │  将更新标记为非紧急   │  │  延迟使用某个值      │              │
 * │  │  用于大列表、导航等   │  │  用于搜索、过滤等    │              │
 * │  └─────────────────────┘  └─────────────────────┘              │
 * │                                                                 │
 * │  ┌─────────────────────┐  ┌─────────────────────┐              │
 * │  │   Suspense          │  │  自动批处理          │              │
 * │  │                     │  │                     │              │
 * │  │  等待异步数据        │  │  多次 setState 合并  │              │
 * │  │  配合 lazy 使用      │  │  减少重渲染次数      │              │
 * │  └─────────────────────┘  └─────────────────────┘              │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 * ```
 */

// ============================================================
// 2. useTransition
// ============================================================

/**
 * 📊 useTransition 原理
 *
 * useTransition 将更新标记为 Transition 优先级（较低）
 * 可以被用户输入等高优先级更新打断
 *
 * 使用场景：
 * - 大列表渲染
 * - 页面导航
 * - Tab 切换
 */

// 简化版 useTransition 实现
function useTransition(): [boolean, (callback: () => void) => void] {
  // isPending 状态
  const [isPending, setIsPending] = useState(false);

  // startTransition 函数
  const startTransition = (callback: () => void) => {
    // 1. 设置 isPending 为 true（高优先级，立即更新）
    setIsPending(true);

    // 2. 在 Transition 优先级下执行回调
    // 源码使用 ReactCurrentBatchConfig.transition 标记
    runWithTransition(() => {
      callback();
      // 3. 设置 isPending 为 false（随 Transition 一起）
      setIsPending(false);
    });
  };

  return [isPending, startTransition];
}

// 模拟 useState
function useState<T>(initial: T): [T, (v: T) => void] {
  let state = initial;
  const setState = (v: T) => { state = v; };
  return [state, setState];
}

// 模拟 Transition 执行
function runWithTransition(callback: () => void) {
  // 在源码中，会设置 ReactCurrentBatchConfig.transition
  // 使得 callback 中的 setState 都是 Transition 优先级
  console.log('Run with transition');
  callback();
}

/**
 * 📊 useTransition 使用示例
 *
 * ```jsx
 * function TabContainer() {
 *   const [isPending, startTransition] = useTransition();
 *   const [tab, setTab] = useState('home');
 *
 *   function selectTab(nextTab) {
 *     startTransition(() => {
 *       setTab(nextTab);  // 低优先级更新
 *     });
 *   }
 *
 *   return (
 *     <>
 *       <TabButton onClick={() => selectTab('home')}>Home</TabButton>
 *       <TabButton onClick={() => selectTab('posts')}>Posts</TabButton>
 *       {isPending && <Spinner />}
 *       <TabPanel tab={tab} />
 *     </>
 *   );
 * }
 * ```
 */

// ============================================================
// 3. useDeferredValue
// ============================================================

/**
 * 📊 useDeferredValue 原理
 *
 * 返回一个延迟版本的值
 * 在紧急更新完成后才更新
 *
 * 使用场景：
 * - 搜索过滤
 * - 输入防抖替代方案
 */

// 简化版 useDeferredValue 实现
function useDeferredValue<T>(value: T): T {
  // 保存上一次的值
  const [deferredValue, setDeferredValue] = useState(value);

  // 在 Transition 优先级下更新
  useEffect(() => {
    runWithTransition(() => {
      setDeferredValue(value);
    });
  }, [value]);

  return deferredValue;
}

// 模拟 useEffect
function useEffect(callback: () => void, deps: any[]) {
  callback();
}

/**
 * 📊 useDeferredValue 使用示例
 *
 * ```jsx
 * function SearchResults({ query }) {
 *   // query 是用户输入（紧急）
 *   // deferredQuery 会延迟更新（不阻塞输入）
 *   const deferredQuery = useDeferredValue(query);
 *
 *   // 结果列表使用延迟的值
 *   // 在用户快速输入时不会每次都重新渲染
 *   return <Results query={deferredQuery} />;
 * }
 * ```
 */

// ============================================================
// 4. 自动批处理
// ============================================================

/**
 * 📊 React 18 自动批处理
 *
 * React 17 及之前：
 * - 只有 React 事件处理中的 setState 才会批处理
 * - setTimeout/Promise 中的不会
 *
 * React 18：
 * - 所有更新都会自动批处理
 *
 * ```jsx
 * // React 17
 * setTimeout(() => {
 *   setCount(c => c + 1);  // 触发重渲染
 *   setFlag(f => !f);      // 触发重渲染（共 2 次）
 * }, 1000);
 *
 * // React 18
 * setTimeout(() => {
 *   setCount(c => c + 1);  // 不立即渲染
 *   setFlag(f => !f);      // 不立即渲染
 *   // 批量处理，只渲染 1 次
 * }, 1000);
 * ```
 */

/**
 * 📊 如何退出批处理？
 *
 * 使用 flushSync 强制同步更新
 *
 * ```jsx
 * import { flushSync } from 'react-dom';
 *
 * function handleClick() {
 *   flushSync(() => {
 *     setCount(c => c + 1);  // 立即渲染
 *   });
 *   // DOM 已更新
 *   console.log(document.body.textContent);
 *
 *   setFlag(f => !f);  // 另一次渲染
 * }
 * ```
 */

// ============================================================
// 5. Suspense 与 lazy
// ============================================================

/**
 * 📊 Suspense 原理
 *
 * Suspense 可以捕获子组件抛出的 Promise
 * 在 Promise resolve 前显示 fallback
 *
 * ```jsx
 * const LazyComponent = React.lazy(() => import('./Component'));
 *
 * function App() {
 *   return (
 *     <Suspense fallback={<Loading />}>
 *       <LazyComponent />
 *     </Suspense>
 *   );
 * }
 * ```
 *
 * 工作原理：
 * 1. lazy 组件第一次渲染时抛出 Promise
 * 2. Suspense 捕获 Promise，显示 fallback
 * 3. Promise resolve 后，重新渲染
 * 4. 这次 lazy 组件正常渲染
 */

// 简化版 lazy 实现
function lazy<T>(
  factory: () => Promise<{ default: T }>
): React.LazyExoticComponent<any> {
  let Component: T | null = null;
  let promise: Promise<void> | null = null;

  return function LazyComponent(props: any) {
    if (Component !== null) {
      // 已加载，正常渲染
      return (Component as any)(props);
    }

    if (promise === null) {
      // 首次渲染，发起加载
      promise = factory().then(module => {
        Component = module.default;
      });
    }

    // 抛出 Promise，让 Suspense 捕获
    throw promise;
  } as any;
}

// ============================================================
// 6. 💡 面试题
// ============================================================

/**
 * 💡 Q1: 什么是 React 的并发模式？
 *
 * A: 并发模式是 React 18 的核心特性：
 *    - 渲染可以被中断
 *    - 支持任务优先级
 *    - 高优先级更新可以插队
 *    - 不会阻塞用户交互
 *
 * 💡 Q2: useTransition 和 useDeferredValue 的区别？
 *
 * A:
 *    useTransition：
 *    - 返回 [isPending, startTransition]
 *    - 用于包装导致更新的操作
 *    - 适合：导航、Tab 切换
 *
 *    useDeferredValue：
 *    - 返回延迟版本的值
 *    - 用于延迟使用某个值
 *    - 适合：搜索过滤（类似防抖）
 *
 * 💡 Q3: React 18 的自动批处理是什么？
 *
 * A: React 18 中，所有更新都会自动批处理：
 *    - 包括 setTimeout、Promise 中的更新
 *    - 多次 setState 合并为一次渲染
 *    - 可用 flushSync 退出批处理
 *
 * 💡 Q4: Suspense 是如何工作的？
 *
 * A: Suspense 捕获子组件抛出的 Promise：
 *    1. lazy 组件抛出加载 Promise
 *    2. Suspense 捕获，显示 fallback
 *    3. Promise resolve 后重新渲染
 */

// ============================================================
// 7. 🏢 实际开发应用
// ============================================================

/**
 * 🏢 应用 1：优化大列表渲染
 *
 * ```jsx
 * function FilteredList({ filter }) {
 *   const [isPending, startTransition] = useTransition();
 *   const [filterValue, setFilterValue] = useState(filter);
 *
 *   function handleChange(e) {
 *     // 输入框立即更新（高优先级）
 *     setInputValue(e.target.value);
 *
 *     // 列表过滤延迟更新（低优先级）
 *     startTransition(() => {
 *       setFilterValue(e.target.value);
 *     });
 *   }
 *
 *   return (
 *     <>
 *       <input onChange={handleChange} />
 *       {isPending && <Spinner />}
 *       <List filter={filterValue} />
 *     </>
 *   );
 * }
 * ```
 */

/**
 * 🏢 应用 2：路由切换优化
 *
 * ```jsx
 * function Router() {
 *   const [isPending, startTransition] = useTransition();
 *
 *   function navigate(url) {
 *     startTransition(() => {
 *       // 路由切换是低优先级
 *       setCurrentUrl(url);
 *     });
 *   }
 *
 *   return (
 *     <>
 *       <Nav navigate={navigate} />
 *       {isPending ? <Skeleton /> : <Page url={currentUrl} />}
 *     </>
 *   );
 * }
 * ```
 */

// ============================================================
// 8. 📖 源码阅读指南
// ============================================================

/**
 * 📖 阅读顺序：
 *
 * 1. packages/react/src/ReactHooks.js
 *    - useTransition
 *    - useDeferredValue
 *
 * 2. packages/react-reconciler/src/ReactFiberHooks.js
 *    - mountTransition / updateTransition
 *    - mountDeferredValue / updateDeferredValue
 *
 * 3. packages/react-reconciler/src/ReactFiberWorkLoop.js
 *    - 查找 Transition 相关逻辑
 *
 * 4. packages/react-reconciler/src/ReactFiberLane.js
 *    - TransitionLanes 定义
 */

// ============================================================
// 9. ✅ 学习检查
// ============================================================

/**
 * ✅ 完成以下任务：
 *
 * - [ ] 理解并发模式的概念
 * - [ ] 理解 useTransition 的原理和使用场景
 * - [ ] 理解 useDeferredValue 的原理和使用场景
 * - [ ] 理解自动批处理
 * - [ ] 理解 Suspense 原理
 */

export {
  useTransition,
  useDeferredValue,
  lazy,
};

