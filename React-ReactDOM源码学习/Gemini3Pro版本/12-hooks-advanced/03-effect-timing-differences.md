# 03. useEffect 与 useLayoutEffect 执行时机

在 React Hooks 中，`useEffect` 和 `useLayoutEffect` 的主要区别在于它们的执行时机。深入源码可以让我们更清晰地理解这一差异。

## 1. Effect 的标记与分类

在 Fiber 架构中，Effects 通过 `flags` (副作用标记) 和 `tag` (Hook 类型) 来区分。

| Hook | Fiber Flags (副作用标记) | Hook Tag |
| :--- | :--- | :--- |
| `useEffect` | `PassiveEffect` | `HookPassive` |
| `useLayoutEffect` | `UpdateEffect` (即 Layout) | `HookLayout` |
| `useInsertionEffect` | `UpdateEffect` | `HookInsertion` |

### 源码实现对比

在 `ReactFiberHooks.new.js` 中：

```javascript
// useEffect
function mountEffect(...) {
  return mountEffectImpl(
    PassiveEffect | PassiveStaticEffect, // Fiber Flags
    HookPassive,                         // Hook Tag
    create,
    deps,
  );
}

// useLayoutEffect
function mountLayoutEffect(...) {
  return mountEffectImpl(
    UpdateEffect,    // Fiber Flags: UpdateEffect 即 Layout 阶段
    HookLayout,      // Hook Tag
    create,
    deps,
  );
}
```

这两个函数最终都调用 `pushEffect` 将 Effect 对象推入 Fiber 的 `updateQueue` 中。区别仅在于传入的标记不同。

## 2. Commit 阶段的执行流

当 Render 阶段结束，进入 Commit 阶段时，React 会处理这些 Effects。Commit 阶段分为三个子阶段：

1.  **Before Mutation**: 操作 DOM 之前。
2.  **Mutation**: 操作 DOM (增删改)。
3.  **Layout**: 操作 DOM 之后。

### 2.1 useLayoutEffect 的执行 (同步)

`useLayoutEffect` 也就是 Layout Effects，是在 **Layout 阶段** 同步执行的。

```javascript
// packages/react-reconciler/src/ReactFiberCommitWork.new.js

function commitLayoutEffects(finishedWork, root, ...) {
  // 递归遍历 Fiber 树
  commitLayoutEffects_begin(finishedWork, ...);
}

function commitLayoutMountEffects_complete(...) {
  // ...
  if ((fiber.flags & LayoutMask) !== NoFlags) {
    // 执行 Layout Effect
    commitLayoutEffectOnFiber(...);
  }
}
```

**关键点**：
*   **时机**：DOM 已经更新，但在浏览器绘制（Paint）之前。
*   **阻塞**：它的执行会阻塞浏览器绘制。如果在这里执行耗时操作，用户会感觉到卡顿。
*   **用途**：适合读取 DOM 布局信息（`getBoundingClientRect`）并同步修改 DOM（防止闪烁）。

### 2.2 useEffect 的执行 (异步/调度)

`useEffect` 也就是 Passive Effects，通常不会在 Commit 阶段同步执行，而是被**调度**到 Commit 阶段之后异步执行。

在 Commit 阶段的入口 `commitRoot` 中：

```javascript
// packages/react-reconciler/src/ReactFiberWorkLoop.new.js

function commitRootImpl(...) {
  // ...

  // 如果有 Passive Effects，调度它们的执行
  if (
    (finishedWork.subtreeFlags & PassiveMask) !== NoFlags ||
    (finishedWork.flags & PassiveMask) !== NoFlags
  ) {
    if (!rootDoesHavePassiveEffects) {
      rootDoesHavePassiveEffects = true;
      // 调度 Passive Effects 的刷新
      scheduleCallback(NormalPriority, () => {
        flushPassiveEffects();
        return null;
      });
    }
  }

  // ... 执行 Mutation 和 Layout ...
}
```

**关键点**：
*   **时机**：在浏览器绘制（Paint）**之后**。它不会阻塞 UI 更新。
*   **调度**：通过 `Scheduler` 调度，通常在宏任务或微任务中执行。
*   **用途**：数据获取、订阅事件、埋点上报等不影响布局的操作。

## 3. 调用栈差异示意

假设我们有一个组件：

```javascript
function App() {
  useLayoutEffect(() => console.log('Layout'), []);
  useEffect(() => console.log('Passive'), []);
  return <div>Hello</div>;
}
```

**执行顺序：**

1.  **Render Phase**: 计算 Fiber 树。
2.  **Commit Phase (Mutation)**: 更新 DOM (`div` 插入页面)。
3.  **Commit Phase (Layout)**:
    *   **同步执行** `useLayoutEffect` 回调 -> 打印 "Layout"。
    *   (此时浏览器尚未绘制，用户看不见更新)
4.  **Browser Paint**: 浏览器将 DOM 绘制到屏幕。
5.  **Passive Effects Phase**:
    *   **异步执行** `useEffect` 回调 -> 打印 "Passive"。

## 4. 特殊情况：useEffect 可能同步执行吗？

虽然设计上是异步的，但在某些特定场景下，`useEffect` 可能会被提前刷新（同步执行），例如：

1.  在 `useEffect` 尚未执行前，又触发了一次新的同步更新（如 `flushSync` 或离散事件）。React 必须先清空上一轮的 Effects 才能开始下一轮 Render，这时 `flushPassiveEffects` 会被同步调用。

## 总结

| 特性 | useEffect | useLayoutEffect |
| :--- | :--- | :--- |
| **执行时机** | 浏览器绘制 **后** (异步) | DOM 更新后，绘制 **前** (同步) |
| **对视觉影响** | 无阻塞，用户先看到旧 UI 或中间态 | 阻塞绘制，用户直接看到最终态 |
| **源码 Flags** | `PassiveEffect` | `UpdateEffect` |
| **适用场景** | 数据请求、日志、非布局相关的订阅 | DOM 测量、DOM 变更后立即修正布局、防止闪烁 |

