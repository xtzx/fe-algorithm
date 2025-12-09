/**
 * ============================================================
 * 📚 Phase 7: 并发特性 - Part 2: 核心 API 与使用
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react/src/ReactStartTransition.js
 * - packages/react/src/ReactHooks.js
 * - packages/react-reconciler/src/ReactFiberHooks.new.js
 * - packages/react-dom/src/client/ReactDOMRoot.js
 */

// ============================================================
// Part 1: createRoot vs render
// ============================================================

/**
 * 📊 createRoot：启用并发模式的入口
 *
 * 📁 源码位置: packages/react-dom/src/client/ReactDOMRoot.js
 */

const createRootVsRender = `
📊 createRoot vs render

┌─────────────────────────────────────────────────────────────────────────────┐
│                     ReactDOM.render (Legacy)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  import ReactDOM from 'react-dom';                                          │
│                                                                             │
│  // React 17 及之前的方式                                                   │
│  ReactDOM.render(<App />, document.getElementById('root'));                 │
│                                                                             │
│  特点：                                                                     │
│  - 同步渲染模式                                                             │
│  - 所有更新同等优先级                                                       │
│  - 不支持并发特性                                                           │
│  - React 18 中仍可用但会有警告                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     ReactDOM.createRoot (Concurrent)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  import { createRoot } from 'react-dom/client';                             │
│                                                                             │
│  // React 18 推荐方式                                                       │
│  const root = createRoot(document.getElementById('root'));                  │
│  root.render(<App />);                                                      │
│                                                                             │
│  特点：                                                                     │
│  - 并发渲染模式                                                             │
│  - 支持优先级调度                                                           │
│  - 支持 Suspense、Transitions 等                                            │
│  - 自动批处理（Automatic Batching）                                         │
│                                                                             │
│  额外方法：                                                                 │
│  root.unmount();  // 卸载整个 React 树                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

源码核心:

// 📁 packages/react-dom/src/client/ReactDOMRoot.js
export function createRoot(container, options) {
  // 创建 FiberRoot，标记为 ConcurrentRoot
  const root = createContainer(
    container,
    ConcurrentRoot,  // ⭐ 关键：使用并发模式
    null,
    isStrictMode,
    concurrentUpdatesByDefaultOverride,
    identifierPrefix,
    onRecoverableError,
    transitionCallbacks,
  );

  // 返回 ReactDOMRoot 实例
  return new ReactDOMRoot(root);
}
`;

// ============================================================
// Part 2: startTransition
// ============================================================

/**
 * 📊 startTransition：标记低优先级更新
 *
 * 📁 源码位置: packages/react/src/ReactStartTransition.js
 */

const startTransitionAPI = `
📊 startTransition

使用方式：
─────────────────────────────────────

import { startTransition } from 'react';

function handleSearch(query) {
  // 高优先级：输入框立即更新
  setInputValue(query);

  // 低优先级：搜索结果可以稍后更新
  startTransition(() => {
    setSearchResults(filterData(query));
  });
}

行为特征：
─────────────────────────────────────

1. 标记为低优先级
   - 内部 setState 会被标记为 TransitionLane
   - 可以被高优先级更新打断

2. 不阻塞用户交互
   - 即使过渡更新执行中，用户输入仍能立即响应

3. 可中断
   - 如果有新的高优先级更新，过渡更新会被暂停
   - 如果有新的同类过渡更新，旧的可能被丢弃

与 Scheduler 的关系：
─────────────────────────────────────

startTransition 内的更新：
  Lane: TransitionLane (0b1000000 ~ 0b1000000000000000000)
  Scheduler Priority: NormalPriority (timeout: 5000ms)
  可中断: ✅

普通 setState：
  Lane: DefaultLane (0b10000) 或更高
  可能使用更高优先级
`;

/**
 * 📊 startTransition 源码解析
 */

const startTransitionSource = `
📊 startTransition 源码解析

📁 packages/react/src/ReactStartTransition.js

export function startTransition(scope, options) {
  // 1. 保存当前 transition 状态
  const prevTransition = ReactCurrentBatchConfig.transition;

  // 2. 设置新的 transition 标记 ⭐
  ReactCurrentBatchConfig.transition = {};

  try {
    // 3. 执行回调（内部的 setState 会读取 transition 标记）
    scope();
  } finally {
    // 4. 恢复之前的状态
    ReactCurrentBatchConfig.transition = prevTransition;
  }
}

关键点：
- ReactCurrentBatchConfig.transition 是一个全局标记
- 当 transition 不为 null 时，setState 会分配 TransitionLane
- 这就是"标记低优先级"的机制

调用链：
scope() 内的 setState
    ↓
dispatchSetState
    ↓
requestUpdateLane(fiber)
    ↓
检查 ReactCurrentBatchConfig.transition !== null ?
    ↓
是 → claimNextTransitionLane()  // 返回 TransitionLane
否 → 其他优先级逻辑
`;

// ============================================================
// Part 3: useTransition
// ============================================================

/**
 * 📊 useTransition：带 pending 状态的 startTransition
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 */

const useTransitionAPI = `
📊 useTransition

使用方式：
─────────────────────────────────────

import { useTransition } from 'react';

function SearchComponent() {
  const [isPending, startTransition] = useTransition();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  function handleChange(e) {
    const value = e.target.value;
    setQuery(value);  // 高优先级

    startTransition(() => {
      setResults(filterData(value));  // 低优先级
    });
  }

  return (
    <div>
      <input value={query} onChange={handleChange} />
      {isPending && <Spinner />}  {/* ⭐ 显示加载状态 */}
      <ResultList data={results} />
    </div>
  );
}

与 startTransition 的区别：
─────────────────────────────────────

startTransition（从 react 导入）：
  - 不提供 pending 状态
  - 可以在任何地方使用（包括组件外）

useTransition（Hook）：
  - 返回 [isPending, startTransition]
  - isPending 在过渡期间为 true
  - 只能在组件内使用

isPending 的作用：
  - 显示加载指示器
  - 禁用某些交互
  - 提供视觉反馈
`;

/**
 * 📊 useTransition 源码解析
 */

const useTransitionSource = `
📊 useTransition 源码解析

📁 packages/react-reconciler/src/ReactFiberHooks.new.js 第 2049-2069 行

// mount 阶段
function mountTransition() {
  // 1. 使用 useState 管理 isPending 状态
  const [isPending, setPending] = mountState(false);

  // 2. 创建 start 函数，绑定 setPending
  const start = startTransition.bind(null, setPending);

  // 3. 保存到 Hook 中
  const hook = mountWorkInProgressHook();
  hook.memoizedState = start;

  return [isPending, start];
}

// startTransition 实现（第 2002-2047 行）
function startTransition(setPending, callback, options) {
  // 1. 降低优先级
  const previousPriority = getCurrentUpdatePriority();
  setCurrentUpdatePriority(
    higherEventPriority(previousPriority, ContinuousEventPriority),
  );

  // 2. 设置 pending = true（高优先级，立即显示）
  setPending(true);

  // 3. 设置 transition 标记
  const prevTransition = ReactCurrentBatchConfig.transition;
  ReactCurrentBatchConfig.transition = {};

  try {
    // 4. 设置 pending = false（低优先级，过渡完成后生效）
    setPending(false);
    // 5. 执行回调
    callback();
  } finally {
    setCurrentUpdatePriority(previousPriority);
    ReactCurrentBatchConfig.transition = prevTransition;
  }
}

执行顺序：
1. setPending(true)  → 高优先级，立即渲染显示 loading
2. setPending(false) → 低优先级，和 callback 内的更新一起
3. callback()        → 低优先级，实际业务更新

结果：
- 用户立即看到 isPending = true
- 过渡完成后看到 isPending = false + 新数据
`;

// ============================================================
// Part 4: useDeferredValue
// ============================================================

/**
 * 📊 useDeferredValue：延迟值更新
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberHooks.new.js
 */

const useDeferredValueAPI = `
📊 useDeferredValue

使用方式：
─────────────────────────────────────

import { useDeferredValue, useState, useMemo } from 'react';

function SearchResults({ query }) {
  // query 是"最新值"，deferredQuery 是"延迟值"
  const deferredQuery = useDeferredValue(query);

  // 使用延迟值进行昂贵计算
  const results = useMemo(
    () => filterLargeDataset(deferredQuery),
    [deferredQuery]
  );

  // 检查是否"过时"
  const isStale = query !== deferredQuery;

  return (
    <div style={{ opacity: isStale ? 0.5 : 1 }}>
      {results.map(item => <Item key={item.id} data={item} />)}
    </div>
  );
}

行为特征：
─────────────────────────────────────

1. 返回延迟版本的值
   - 紧急更新时：返回旧值
   - 非紧急更新时：返回新值

2. 自动触发低优先级渲染
   - 当值变化时，会调度一个 TransitionLane 的更新

3. 用途
   - 延迟昂贵的重渲染
   - 保持 UI 响应

与 useTransition 的区别：
─────────────────────────────────────

useTransition：
  - 主动包裹 setState
  - 控制"什么时候更新"

useDeferredValue：
  - 传入一个值
  - 控制"使用哪个版本的值"
  - 适合无法修改 setState 的场景（如第三方库传入的 props）
`;

/**
 * 📊 useDeferredValue 源码解析
 */

const useDeferredValueSource = `
📊 useDeferredValue 源码解析

📁 packages/react-reconciler/src/ReactFiberHooks.new.js 第 1931-1992 行

// mount 阶段：直接返回原值
function mountDeferredValue(value) {
  const hook = mountWorkInProgressHook();
  hook.memoizedState = value;
  return value;
}

// update 阶段：核心逻辑
function updateDeferredValue(value) {
  const hook = updateWorkInProgressHook();
  const prevValue = currentHook.memoizedState;
  return updateDeferredValueImpl(hook, prevValue, value);
}

function updateDeferredValueImpl(hook, prevValue, value) {
  // 1. 判断当前是否是"紧急更新"
  const shouldDeferValue = !includesOnlyNonUrgentLanes(renderLanes);

  if (shouldDeferValue) {
    // ⭐ 紧急更新：返回旧值，同时调度延迟更新

    if (!is(value, prevValue)) {
      // 值变了，需要调度延迟更新
      const deferredLane = claimNextTransitionLane();  // 获取 TransitionLane
      currentlyRenderingFiber.lanes = mergeLanes(
        currentlyRenderingFiber.lanes,
        deferredLane,
      );
      markSkippedUpdateLanes(deferredLane);

      // 标记为"不一致状态"
      hook.baseState = true;
    }

    // 返回旧值
    return prevValue;

  } else {
    // 非紧急更新（如 Transition）：使用新值

    if (hook.baseState) {
      // 清除"不一致"标记
      hook.baseState = false;
      markWorkInProgressReceivedUpdate();
    }

    hook.memoizedState = value;
    return value;
  }
}

工作流程：
1. 用户输入触发高优先级更新 → useDeferredValue 返回旧值
2. 同时调度一个 TransitionLane 的更新
3. 高优先级渲染完成，用户看到输入框更新
4. 低优先级渲染开始，useDeferredValue 返回新值
5. 列表重新渲染
`;

// ============================================================
// Part 5: Suspense
// ============================================================

/**
 * 📊 Suspense：异步数据加载
 */

const suspenseAPI = `
📊 Suspense

使用方式：
─────────────────────────────────────

import { Suspense } from 'react';

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <AsyncComponent />
    </Suspense>
  );
}

// 使用 React.lazy 的场景
const LazyComponent = React.lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <LazyComponent />
    </Suspense>
  );
}

// 使用数据获取库的场景（如 React Query、SWR、Relay）
function UserProfile({ userId }) {
  const user = useSuspenseQuery(['user', userId], fetchUser);
  return <div>{user.name}</div>;
}

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <UserProfile userId={1} />
    </Suspense>
  );
}

行为特征：
─────────────────────────────────────

1. 捕获子组件的"挂起"状态
   - 子组件抛出 Promise 时，显示 fallback

2. Promise resolve 后自动重试
   - React 会重新渲染子组件

3. 支持嵌套
   - 多层 Suspense 可以独立显示 fallback

4. 与并发特性配合
   - Transition 中的 Suspense 会延迟显示 fallback
   - 避免闪烁

Suspense 的挂起机制：
─────────────────────────────────────

// 组件内部抛出 Promise
function AsyncComponent() {
  const data = cache.read();  // 如果未就绪，抛出 Promise
  return <div>{data}</div>;
}

// cache.read 的实现（简化）
function read() {
  if (status === 'resolved') return value;
  if (status === 'pending') throw promise;  // ⭐ 抛出 Promise

  // 首次调用，发起请求
  status = 'pending';
  promise = fetch(url).then(data => {
    status = 'resolved';
    value = data;
  });
  throw promise;
}
`;

/**
 * 📊 Suspense 与 Transition 的配合
 */

const suspenseWithTransition = `
📊 Suspense + Transition

场景：页面切换时保持旧内容
─────────────────────────────────────

function Tabs() {
  const [tab, setTab] = useState('home');
  const [isPending, startTransition] = useTransition();

  function selectTab(nextTab) {
    startTransition(() => {
      setTab(nextTab);  // 低优先级
    });
  }

  return (
    <div>
      <TabButtons
        selectedTab={tab}
        onSelect={selectTab}
        isPending={isPending}
      />
      <Suspense fallback={<Loading />}>
        <TabContent tab={tab} />
      </Suspense>
    </div>
  );
}

行为对比：
─────────────────────────────────────

没有 Transition：
  点击 Tab → 立即显示 Loading → 数据就绪 → 显示内容
  问题：闪烁！

有 Transition：
  点击 Tab → 保持旧内容（isPending=true）→ 数据就绪 → 切换到新内容
  优势：平滑过渡，无闪烁

原理：
  - Transition 中触发的 Suspense 不会立即显示 fallback
  - React 会"等待"新内容就绪
  - 期间保持显示旧内容
`;

// ============================================================
// Part 6: API 总结表
// ============================================================

const apiSummary = `
📊 并发特性 API 总结

┌─────────────────────────────────────────────────────────────────────────────┐
│ API                  │ 用途                     │ 与 Scheduler 关系         │
├──────────────────────┼──────────────────────────┼───────────────────────────┤
│                      │                          │                           │
│ createRoot           │ 启用并发模式             │ 创建 ConcurrentRoot       │
│                      │                          │ 启用并发渲染能力          │
│                      │                          │                           │
│ startTransition      │ 标记低优先级更新         │ 分配 TransitionLane       │
│                      │ 可被打断                 │ NormalPriority (5s)       │
│                      │                          │                           │
│ useTransition        │ 同上 + isPending 状态    │ 同上                      │
│                      │ 显示加载指示             │                           │
│                      │                          │                           │
│ useDeferredValue     │ 延迟值更新               │ 返回旧值时触发            │
│                      │ 保持 UI 响应             │ TransitionLane 更新       │
│                      │                          │                           │
│ Suspense             │ 异步数据/代码加载        │ 挂起时暂停渲染            │
│                      │ 显示 fallback            │ 可与 Transition 配合      │
│                      │                          │                           │
└─────────────────────────────────────────────────────────────────────────────┘

使用场景选择：
─────────────────────────────────────

1. 搜索框 + 列表过滤
   → useTransition（需要显示 loading）
   → 或 startTransition（不需要 loading）

2. 列表虚拟化/大数据渲染
   → useDeferredValue（延迟列表渲染）

3. 页面切换
   → useTransition + Suspense（避免闪烁）

4. 懒加载组件
   → React.lazy + Suspense

5. 数据获取
   → 数据库 + Suspense（如 React Query suspense 模式）
`;

export {
  createRootVsRender,
  startTransitionAPI,
  startTransitionSource,
  useTransitionAPI,
  useTransitionSource,
  useDeferredValueAPI,
  useDeferredValueSource,
  suspenseAPI,
  suspenseWithTransition,
  apiSummary,
};

