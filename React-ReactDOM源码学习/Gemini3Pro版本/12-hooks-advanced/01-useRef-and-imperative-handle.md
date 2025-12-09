# 01. useRef 与 useImperativeHandle

在 React 中，Ref 是"逃脱 Hatch"（Escape Hatch），允许我们直接访问 DOM 节点或组件实例。本节我们将深入 `useRef`、`forwardRef` 和 `useImperativeHandle` 的源码实现，理解它们是如何协同工作的。

## 1. useRef 的内部实现

`useRef` 是最简单的 Hook 之一。它的核心作用是在 Fiber 节点上存储一个"在渲染之间保持引用不变"的对象。

### 1.1 数据结构

在 `ReactFiberHooks.new.js` 中，`useRef` 的实现非常直观：

```javascript
// packages/react-reconciler/src/ReactFiberHooks.new.js

function mountRef<T>(initialValue: T): {|current: T|} {
  const hook = mountWorkInProgressHook();
  // 创建 ref 对象
  const ref = {current: initialValue};
  // 存入 Hook 链表
  hook.memoizedState = ref;
  return ref;
}

function updateRef<T>(initialValue: T): {|current: T|} {
  const hook = updateWorkInProgressHook();
  // 直接返回之前存储的 ref 对象
  return hook.memoizedState;
}
```

**关键点：**
1.  **单例对象**：`mountRef` 创建了一个普通对象 `{current: initialValue}`。
2.  **持久化**：这个对象被保存在 `hook.memoizedState` 中。
3.  **引用不变**：`updateRef` 只是简单地把这个对象取出来返回。React 不会去更新这个对象引用，因此 `ref.current` 的变化不会触发组件重新渲染。

> **注意**：在开发模式（`__DEV__`）下，React 会额外封装 ref 对象以拦截 `current` 属性的读写，用于警告在渲染阶段（Render Phase）不安全的读写操作（例如 `if (ref.current) ...`）。

## 2. Ref 的挂载与更新（Commit 阶段）

虽然 `useRef` 创建了 ref 对象，但将 DOM 节点或组件实例赋值给 `ref.current` 的动作发生在 **Commit 阶段**。

### 2.1 Commit 流程中的处理

在 `ReactFiberCommitWork.new.js` 中，`commitAttachRef` 函数负责处理 Ref 的连接：

```javascript
// packages/react-reconciler/src/ReactFiberCommitWork.new.js

function commitAttachRef(finishedWork: Fiber) {
  const ref = finishedWork.ref;
  if (ref !== null) {
    const instance = finishedWork.stateNode;
    let instanceToUse;
    // ...根据组件类型获取实例 (DOM 节点或组件实例)

    // 1. 处理函数类型 Ref (callback ref)
    if (typeof ref === 'function') {
      ref(instanceToUse);
    }
    // 2. 处理对象类型 Ref (object ref)
    else {
      ref.current = instanceToUse;
    }
  }
}
```

同理，当组件卸载或 Ref 变化时，会有 `commitDetachRef` 将 `ref.current` 置为 `null`。

**时机**：Ref 的 attach 发生在 Layout 阶段（`commitLayoutEffects`），这意味着此时 DOM 已经更新完毕，可以安全地获取最新的 DOM 节点。

## 3. forwardRef 的实现

`forwardRef` 本身不是 Hook，而是一个高阶组件（HOC）工厂函数。

```javascript
// packages/react/src/ReactForwardRef.js

export function forwardRef(render) {
  // ...DEV 检查

  // 返回一个特殊类型的 React Element Type
  return {
    $$typeof: REACT_FORWARD_REF_TYPE,
    render,
  };
}
```

它仅仅是创建了一个带有特殊 `$$typeof` 的对象。真正的魔法发生在 `ReactFiberBeginWork.new.js` 中：

1.  React 遇到 `REACT_FORWARD_REF_TYPE` 的 Fiber。
2.  调用 `render` 函数时，将 `props` 和 `ref` 作为两个参数传入：`render(props, ref)`。
3.  这就解释了为什么普通函数组件没有 `ref` 参数，而 `forwardRef` 包裹的组件有。

## 4. useImperativeHandle 的工作原理

`useImperativeHandle` 允许我们自定义暴露给父组件的实例值（而不是直接暴露 DOM 节点）。

### 4.1 源码实现

它的底层实际上是调用了 `useLayoutEffect`（源码中复用了 layout effect 的机制）。

```javascript
// packages/react-reconciler/src/ReactFiberHooks.new.js

function mountImperativeHandle<T>(
  ref: {|current: T | null|} | ((inst: T | null) => mixed) | null | void,
  create: () => T,
  deps: Array<mixed> | void | null,
): void {
  // ...
  return mountEffectImpl(
    UpdateEffect | LayoutStaticEffect, // Layout Effect 标记
    HookLayout,
    imperativeHandleEffect.bind(null, create, ref),
    effectDeps,
  );
}

function imperativeHandleEffect<T>(
  create: () => T,
  ref: {|current: T | null|} | ((inst: T | null) => mixed) | null | void,
) {
  if (typeof ref === 'function') {
    const refCallback = ref;
    const inst = create(); // 调用用户提供的 create 函数
    refCallback(inst); // 调用 callback ref
    return () => {
      refCallback(null); // 清理
    };
  } else if (ref !== null && ref !== undefined) {
    const refObject = ref;
    const inst = create();
    refObject.current = inst; // 赋值 object ref
    return () => {
      refObject.current = null; // 清理
    };
  }
}
```

### 4.2 流程解析

1.  **Mount/Update**：`useImperativeHandle` 注册一个 **Layout Effect**。
2.  **Commit 阶段**：
    *   在 DOM 更新后，Layout Effects 执行。
    *   执行 `imperativeHandleEffect`。
    *   调用用户的 `create` 函数（例如返回 `{ focus: () => ... }`）。
    *   将返回值赋值给传入的 `ref`（无论是对象 ref 还是函数 ref）。

### 4.3 示例场景：父组件调用子组件方法

```javascript
const Child = forwardRef((props, ref) => {
  const inputRef = useRef();

  useImperativeHandle(ref, () => ({
    focus: () => {
      inputRef.current.focus();
    }
  }));

  return <input ref={inputRef} />;
});

const Parent = () => {
  const childRef = useRef();
  // childRef.current 将会是 { focus: ... } 而不是 Child 的 Fiber 或 DOM

  return <Child ref={childRef} />;
};
```

**底层执行流：**
1.  **Render**: `Parent` 渲染，创建 `childRef`。
2.  **Render**: `Child` 渲染，执行 `useImperativeHandle`，产生一个 Layout Effect 描述。
3.  **Commit**: DOM 节点创建并挂载。
4.  **Commit (Layout)**: `Child` 的 Layout Effect 执行。
    *   调用 `create()` 得到 `{ focus: ... }`。
    *   执行 `childRef.current = { focus: ... }`。
5.  **Commit (Layout)**: 此时在 `Parent` 的 `useLayoutEffect` 中已经可以安全访问 `childRef.current.focus()`。

## 总结

1.  **Ref 存储**：`useRef` 只是在 Fiber 的 memoizedState 上存了一个普通对象。
2.  **Ref 赋值**：React 在 Commit 阶段的 Layout 子阶段，通过 `commitAttachRef` 自动处理 DOM/Class 组件的 Ref 绑定。
3.  **自定义暴露**：`useImperativeHandle` 本质上是一个 `useLayoutEffect`，它拦截了 Ref 的赋值过程，用自定义对象替换了默认的实例。

