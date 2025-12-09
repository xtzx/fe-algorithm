# 05. useId 与 useInsertionEffect

本节探讨两个较为特殊的 Hook：用于生成唯一标识的 `useId` 和用于 CSS-in-JS 库的 `useInsertionEffect`。

## 1. useId：前后端一致的 ID 生成

在 SSR（服务端渲染）场景下，客户端 Hydration（注水）时要求生成的 HTML 结构与服务端完全一致。如果使用 `Math.random()` 生成 ID，会导致 mismatch 错误。`useId` 解决了这个问题。

### 1.1 原理：基于树结构的层级 ID

`useId` 不依赖随机数，而是依赖组件在 Fiber 树中的**位置（层级和顺序）**。

在 `ReactFiberHooks.new.js` 中：

```javascript
function mountId(): string {
  const hook = mountWorkInProgressHook();
  const root = getWorkInProgressRoot();

  // 1. 获取 ID 前缀 (通常是 configured identifierPrefix)
  const identifierPrefix = root.identifierPrefix;

  let id;
  if (getIsHydrating()) {
    // 2. Server 模式 / Hydration 模式
    const treeId = getTreeId(); // 获取当前层级的 Tree ID
    id = ':' + identifierPrefix + 'R' + treeId;
  } else {
    // 3. Client 模式
    const globalClientId = globalClientIdCounter++;
    id = ':' + identifierPrefix + 'r' + globalClientId.toString(32) + ':';
  }

  // 4. 处理同一个组件内多次调用 useId
  const localId = localIdCounter++;
  if (localId > 0) {
    id += 'H' + localId.toString(32);
  }

  id += ':';
  hook.memoizedState = id;
  return id;
}
```

### 1.2 Tree ID 的生成

React 内部维护了一个 `TreeContext`，在遍历 Fiber 树时动态生成 ID。
例如：`R` (Root) -> `R1` (第一个子节点) -> `R1.2` (R1 的第二个子节点)。

因为 React 的渲染顺序是确定的（深度优先遍历），所以只要组件结构一致，服务端和客户端生成的 ID 序列就是一致的。

## 2. useInsertionEffect：样式注入专用

`useInsertionEffect` 是 React 18 新增的 Hook，专门为 CSS-in-JS 库（如 styled-components, Emotion）设计，用于动态注入 `<style>` 标签。

### 2.1 为什么需要这个 Hook？

在 React 18 之前，CSS-in-JS 库通常在 `useLayoutEffect` 中注入样式。但这样做有一个性能问题：

1.  Render 结束。
2.  开始执行 Layout Effects。
3.  **CSS-in-JS 注入样式**：浏览器重新计算样式（Recalculate Style）。
4.  **组件读取布局**：用户代码中紧接着的 `useLayoutEffect` 读取 `ref.current.offsetWidth`。
5.  **强制重排（Layout Thrashing）**：因为样式刚变，浏览器必须立即重排才能返回正确的宽度。

### 2.2 Insertion Effect 的执行时机

`useInsertionEffect` 的执行时机在 **DOM 变更之前**（或紧随其后，但在所有 Layout Effects 之前）。

```javascript
// Hook Tag
HookInsertion

// 时机
function commitMutationEffects(...) {
  // 1. 先执行 Insertion Effects
  commitInsertionEffects(root, ...);

  // 2. 再执行 DOM 变更 (Mutation)
  // ...

  // 3. 最后执行 Layout Effects
  commitLayoutEffects(root, ...);
}
```

*(注：具体执行顺序在源码中可能有细微调整，但核心保证是 **早于** 所有的 `useLayoutEffect`)*。

通过在 Layout Effect 运行之前注入样式，React 确保了当用户的 `useLayoutEffect` 执行时，样式已经就位，从而避免了重复的样式计算和回流。

### 2.3 源码实现

它本质上还是一个 Effect，只是 Tag 不同：

```javascript
function mountInsertionEffect(create, deps) {
  return mountEffectImpl(
    UpdateEffect,    // 复用了 UpdateEffect 标记
    HookInsertion,   // 专门的 HookInsertion Tag
    create,
    deps,
  );
}
```

在 Commit 阶段，React 会专门遍历处理 `HookInsertion` 类型的 Effects。

## 总结与 Phase 关联

至此，我们完成了 Hooks 深入部分的学习。

*   **关联 Phase 3 (渲染流程)**：我们看到了 `mountWorkInProgressHook` 如何将 Hooks 挂载到 Fiber 的 `memoizedState` 链表上。
*   **关联 Phase 6/7 (并发)**：`useSyncExternalStore` 和 `useId` 都是为了适应并发渲染和 SSR/Hydration 的复杂性而诞生的。普通 Hooks (`useEffect`) 在并发模式下可能会面临 Tearing 问题。
*   **关联 Phase 8 (事件)**：虽然本章未深究事件，但 `useRef` 能够保持引用的特性，常用于在 Event Handler 中获取最新的 State（避免闭包过时问题），这是 Hooks 编程的一个重要模式。

接下来，可以继续探索 React 的更多高级特性。

