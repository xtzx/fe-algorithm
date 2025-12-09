# 04. useSyncExternalStore 原理与 Tearing 问题

React 18 引入了 `useSyncExternalStore`，这是一个专门为第三方状态管理库（如 Redux、Zustand、MobX）设计的 Hook。它的目的是解决并发渲染（Concurrent Rendering）下的 "Tearing"（撕裂）问题。

## 1. 什么是 Tearing？

在 React 18 的并发模式下，渲染是可以被中断的。

**场景描述：**
1.  **开始渲染**：React 开始渲染组件树。
2.  **读取状态**：组件 A 读取外部 Store 的值为 `1`。
3.  **中断**：React 暂停渲染，让出主线程处理高优先级任务（如用户点击）。
4.  **外部更新**：在暂停期间，外部 Store 发生变化，值变为 `2`。
5.  **恢复渲染**：React 继续渲染组件树。
6.  **读取状态**：组件 B（与 A 依赖同一 Store）读取 Store 的值为 `2`。
7.  **提交**：界面上同时显示了 `1` 和 `2`。这就是 **Tearing**。

`useSyncExternalStore` 通过强制在某些情况下同步更新或进行一致性检查来避免这个问题。

## 2. 核心 API 与实现

```javascript
const state = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot?);
```

### 2.1 Mount 阶段

在 `ReactFiberHooks.new.js` 中：

```javascript
function mountSyncExternalStore<T>(
  subscribe: (() => void) => () => void,
  getSnapshot: () => T,
  getServerSnapshot?: () => T,
): T {
  const hook = mountWorkInProgressHook();

  // 1. 获取当前快照
  let nextSnapshot = getSnapshot();

  // 2. 存入 Hook
  hook.memoizedState = nextSnapshot;
  const inst = {
    value: nextSnapshot,
    getSnapshot,
  };
  hook.queue = inst;

  // 3. 注册订阅 Effect (Layout Effect 或 Passive Effect，稍后细说)
  mountEffect(subscribeToStore.bind(null, fiber, inst, subscribe), [subscribe]);

  // 4. 并发一致性检查 (重要!)
  // 如果当前不在阻塞模式（即处于并发模式），需要注册一个检查
  if (!includesBlockingLane(root, renderLanes)) {
    pushStoreConsistencyCheck(fiber, getSnapshot, nextSnapshot);
  }

  return nextSnapshot;
}
```

### 2.2 一致性检查机制

React 在并发渲染结束准备 Commit 之前，会执行 `pushStoreConsistencyCheck` 注册的检查逻辑。

1.  再次调用 `getSnapshot()` 获取最新值。
2.  对比渲染开始时获取的 `nextSnapshot`。
3.  如果值不一致，说明在并发渲染过程中 Store 变了。
4.  **强制重渲染**：React 丢弃本次渲染结果，强制进行一次同步阻塞渲染，确保所有组件读到的一致性状态。

### 2.3 订阅机制

`subscribeToStore` 是一个 Effect，它负责订阅外部 Store 的变化：

```javascript
function subscribeToStore(fiber, inst, subscribe) {
  const handleStoreChange = () => {
    // 当 Store 变化时
    if (checkIfSnapshotChanged(inst)) {
      // 强制更新 Fiber
      forceStoreRerender(fiber);
    }
  };
  // 调用用户传入的 subscribe
  return subscribe(handleStoreChange);
}
```

## 3. updateSyncExternalStore

更新阶段逻辑类似：

```javascript
function updateSyncExternalStore(subscribe, getSnapshot) {
  const hook = updateWorkInProgressHook();

  // 每次 Render 都重新获取 Snapshot
  const nextSnapshot = getSnapshot();
  const prevSnapshot = hook.memoizedState;

  // 如果 Snapshot 变了，说明发生了 tearing 或数据更新
  if (!is(prevSnapshot, nextSnapshot)) {
    hook.memoizedState = nextSnapshot;
    markWorkInProgressReceivedUpdate();
  }

  // ... 同样的 Effect 和一致性检查逻辑

  return nextSnapshot;
}
```

## 4. 解决 Tearing 的策略总结

`useSyncExternalStore` 采用了双重策略：

1.  **Render 期间的检测**：在并发渲染过程中，通过对比 Snapshot 发现变化，一旦发现不一致，立即放弃当前并发渲染，转为同步渲染。同步渲染是不可中断的，因此不会有 Tearing。
2.  **Commit 后的订阅**：通过标准的订阅机制，监听后续的变化来触发更新。

## 5. 对比手动实现

如果手动用 `useEffect` + `useState` 实现：

```javascript
// ❌ 可能导致 Tearing
function useStore(store) {
  const [state, setState] = useState(store.get());
  useEffect(() => {
    return store.subscribe(setState);
  }, []);
  return state;
}
```

*   `useState` 的初始值是在 Render 开始时读取的。
*   `useEffect` 的订阅是在 Commit 之后才生效的。
*   **空窗期**：从 Render 到 Commit 这段时间内，如果 Store 变了，组件感知不到，且无法保证一致性。

`useSyncExternalStore` 填补了这个空窗期，并与 React 的并发调度深度集成。

