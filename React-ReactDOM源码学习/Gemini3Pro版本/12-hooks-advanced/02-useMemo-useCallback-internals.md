# 02. useMemo 与 useCallback 内部原理

`useMemo` 和 `useCallback` 是 React 提供的性能优化 Hooks，用于缓存计算结果和函数引用。它们的实现非常相似，区别仅在于缓存的是"值"还是"函数本身"。

## 1. 存储结构

在 Fiber 的 Hook 链表中，这两个 Hooks 的 `memoizedState` 存储结构如下：

```javascript
// [缓存的值, 依赖数组]
hook.memoizedState = [value, deps];
```

这与 `useState` 存储单个 state 值不同，它们显式地存储了依赖数组，以便在更新时进行比较。

## 2. Mount 阶段

在组件初次渲染时，React 调用 `mountMemo` 或 `mountCallback`。

### 2.1 useMemo

```javascript
// packages/react-reconciler/src/ReactFiberHooks.new.js

function mountMemo<T>(
  nextCreate: () => T,
  deps: Array<mixed> | void | null,
): T {
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  // 1. 立即执行 create 函数获取值
  const nextValue = nextCreate();
  // 2. 将 [值, 依赖] 存入 memoizedState
  hook.memoizedState = [nextValue, nextDeps];
  // 3. 返回值
  return nextValue;
}
```

### 2.2 useCallback

```javascript
function mountCallback<T>(callback: T, deps: Array<mixed> | void | null): T {
  const hook = mountWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  // 1. 直接缓存 callback 函数本身
  hook.memoizedState = [callback, nextDeps];
  // 2. 返回 callback
  return callback;
}
```

**区别**：`useMemo` 执行函数取结果，`useCallback` 直接存函数。这就解释了为什么 `useCallback(fn, deps)` 等价于 `useMemo(() => fn, deps)`。

## 3. Update 阶段

在组件更新时，React 决定是复用缓存还是重新计算。

### 3.1 依赖比较逻辑

React 使用 `areHookInputsEqual` 函数来比较依赖数组：

```javascript
function areHookInputsEqual(
  nextDeps: Array<mixed>,
  prevDeps: Array<mixed> | null,
) {
  // 1. 如果新旧依赖有其一为 null (未传 deps)，则认为不相等 -> 重新计算
  if (prevDeps === null) {
    return false;
  }

  // 2. 逐个元素比较
  for (let i = 0; i < prevDeps.length && i < nextDeps.length; i++) {
    // 使用 Object.is 进行浅比较
    if (is(nextDeps[i], prevDeps[i])) {
      continue;
    }
    return false;
  }
  return true;
}
```

> **注意**：
> *   如果不传依赖数组（`undefined`），`nextDeps` 会被处理为 `null`，导致每次都重新计算。
> *   传空数组 `[]`，则 `prevDeps` 和 `nextDeps` 长度都为 0，循环不执行，直接返回 `true`，永久缓存。

### 3.2 updateMemo

```javascript
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
      // 1. 比较依赖
      if (areHookInputsEqual(nextDeps, prevDeps)) {
        // 2. 依赖未变：返回缓存值
        return prevState[0];
      }
    }
  }

  // 3. 依赖变化：重新计算
  const nextValue = nextCreate();
  hook.memoizedState = [nextValue, nextDeps];
  return nextValue;
}
```

### 3.3 updateCallback

```javascript
function updateCallback<T>(callback: T, deps: Array<mixed> | void | null): T {
  const hook = updateWorkInProgressHook();
  const nextDeps = deps === undefined ? null : deps;
  const prevState = hook.memoizedState;

  if (prevState !== null) {
    if (nextDeps !== null) {
      const prevDeps = prevState[1];
      // 1. 比较依赖
      if (areHookInputsEqual(nextDeps, prevDeps)) {
        // 2. 依赖未变：返回旧的 callback
        return prevState[0];
      }
    }
  }

  // 3. 依赖变化：缓存新的 callback
  hook.memoizedState = [callback, nextDeps];
  return callback;
}
```

## 4. 缓存失效与最佳实践

### 4.1 语义上的 "缓存建议"

官方文档提到 React 未来可能会选择"丢弃"缓存以释放内存。在源码实现中，目前主要是基于依赖数组的确定性缓存。但在 Concurrent 模式下，如果组件被挂起（Suspense）或丢弃（Offscreen），缓存策略可能会更复杂。

### 4.2 常见误区

**场景：在 Render 中创建大对象**

```javascript
function App() {
  // 错误：expensiveObject 每次 render 都会被创建，占用内存
  const expensiveObject = { ...largeData };

  // 这里的 useMemo 只是避免了 value 变量引用的变化
  // 但没有避免 expensiveObject 的创建开销！
  const value = useMemo(() => expensiveObject, [deps]);
}
```

**正确做法**：

```javascript
function App() {
  // 正确：只有 deps 变化时，才会执行函数创建对象
  const value = useMemo(() => {
    return { ...largeData };
  }, [deps]);
}
```

## 总结

1.  **存储**：`useMemo` 和 `useCallback` 都在 `memoizedState` 中存储 `[value, deps]` 元组。
2.  **比较**：使用 `Object.is` 对依赖数组进行浅比较。
3.  **差异**：`useMemo` 缓存函数执行结果，`useCallback` 缓存函数本身。
4.  **复用**：如果依赖未变，React 会直接返回之前存储的引用，这对于配合 `React.memo` 避免子组件不必要的重渲染至关重要。

