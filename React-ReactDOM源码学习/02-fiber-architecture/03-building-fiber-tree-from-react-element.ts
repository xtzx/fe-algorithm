/**
 * ============================================================
 * 📚 Phase 2: Fiber 架构 - Part 3: 从 ReactElement 构建 Fiber 树
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-reconciler/src/ReactFiber.new.js
 * - packages/react-reconciler/src/ReactChildFiber.new.js
 * - packages/react-reconciler/src/ReactFiberBeginWork.new.js
 *
 * ⏱️ 预计时间：3-4 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: ReactElement → FiberNode 的关键函数
// ============================================================

/**
 * 📊 创建 Fiber 的核心函数
 */

const createFiberFunctions = `
📊 创建 Fiber 的核心函数

📁 源码位置: packages/react-reconciler/src/ReactFiber.new.js

┌─────────────────────────────────────────────────────────────────────────────┐
│                       函数调用层次                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   createFiberFromElement(element, mode, lanes)                              │
│           │                                                                 │
│           │  提取 element 的 type、key、props                               │
│           ▼                                                                 │
│   createFiberFromTypeAndProps(type, key, props, owner, mode, lanes)         │
│           │                                                                 │
│           │  根据 type 判断 tag（FunctionComponent、HostComponent 等）       │
│           ▼                                                                 │
│   createFiber(tag, pendingProps, key, mode)                                 │
│           │                                                                 │
│           │  调用 FiberNode 构造函数                                        │
│           ▼                                                                 │
│   new FiberNode(tag, pendingProps, key, mode)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


createFiberFromElement 源码（简化）:
═══════════════════════════════════════════════════════════════════════════════

// 📁 packages/react-reconciler/src/ReactFiber.new.js (Line 604-628)

export function createFiberFromElement(
  element: ReactElement,
  mode: TypeOfMode,
  lanes: Lanes,
): Fiber {
  let owner = null;
  if (__DEV__) {
    owner = element._owner;  // 来自 ReactElement._owner
  }

  const type = element.type;       // 从 ReactElement 提取 type
  const key = element.key;         // 从 ReactElement 提取 key
  const pendingProps = element.props;  // 从 ReactElement 提取 props

  // ⭐ 调用下一级函数
  const fiber = createFiberFromTypeAndProps(
    type,
    key,
    pendingProps,
    owner,
    mode,
    lanes,
  );

  if (__DEV__) {
    fiber._debugSource = element._source;
    fiber._debugOwner = element._owner;
  }

  return fiber;
}


createFiberFromTypeAndProps 源码（简化）:
═══════════════════════════════════════════════════════════════════════════════

// 📁 packages/react-reconciler/src/ReactFiber.new.js (Line 468-600)

export function createFiberFromTypeAndProps(
  type: any,
  key: null | string,
  pendingProps: any,
  owner: null | Fiber,
  mode: TypeOfMode,
  lanes: Lanes,
): Fiber {
  // ⭐ 默认是 IndeterminateComponent（还不知道是函数还是类）
  let fiberTag = IndeterminateComponent;
  let resolvedType = type;

  // ========== 判断 type 类型，决定 tag ==========

  if (typeof type === 'function') {
    // 函数或类
    if (shouldConstruct(type)) {
      // 有 prototype.isReactComponent → 类组件
      fiberTag = ClassComponent;
    }
    // 否则是函数组件（或 IndeterminateComponent，首次渲染时确定）
  } else if (typeof type === 'string') {
    // 字符串 → 原生 DOM 元素
    fiberTag = HostComponent;
  } else {
    // 其他特殊类型（Fragment、Suspense 等）
    switch (type) {
      case REACT_FRAGMENT_TYPE:
        return createFiberFromFragment(pendingProps.children, mode, lanes, key);
      case REACT_SUSPENSE_TYPE:
        return createFiberFromSuspense(pendingProps, mode, lanes, key);
      // ... 其他特殊类型
    }

    // 检查 $$typeof（Provider、Consumer、ForwardRef、Memo 等）
    if (typeof type === 'object' && type !== null) {
      switch (type.$$typeof) {
        case REACT_PROVIDER_TYPE:
          fiberTag = ContextProvider;
          break;
        case REACT_CONTEXT_TYPE:
          fiberTag = ContextConsumer;
          break;
        case REACT_FORWARD_REF_TYPE:
          fiberTag = ForwardRef;
          break;
        case REACT_MEMO_TYPE:
          fiberTag = MemoComponent;
          break;
        // ...
      }
    }
  }

  // ========== 创建 Fiber 并设置属性 ==========
  const fiber = createFiber(fiberTag, pendingProps, key, mode);
  fiber.elementType = type;
  fiber.type = resolvedType;
  fiber.lanes = lanes;

  return fiber;
}
`;

// ============================================================
// Part 2: 初次渲染的 Fiber 树构建
// ============================================================

/**
 * 📊 初次渲染（Mount）流程
 */

const mountProcess = `
📊 初次渲染的 Fiber 树构建

┌─────────────────────────────────────────────────────────────────────────────┐
│                       初次渲染入口                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ReactDOM.createRoot(container).render(<App />)                            │
│                                                                             │
│   这个调用会：                                                              │
│   1. 创建 FiberRootNode（整个应用的根）                                     │
│   2. 创建 HostRoot Fiber（Fiber 树的根）                                    │
│   3. 调度首次渲染                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

初次渲染的 Fiber 树构建过程:
═══════════════════════════════════════════════════════════════════════════════

以这个组件为例:

function Child() { return <span>child</span>; }
function App() {
  return (
    <div>
      <h1>title</h1>
      <Child />
    </div>
  );
}

ReactDOM.createRoot(root).render(<App />);


Step 1: 创建根节点
─────────────────────────

┌─────────────────┐
│  FiberRootNode  │  ← createRoot 时创建
│  containerInfo: │     - 这不是 Fiber！
│    #root DOM    │     - 是整个应用的管理容器
└────────┬────────┘
         │ current
         ▼
┌─────────────────┐
│  HostRoot Fiber │  ← createHostRootFiber 创建
│  tag: 3         │     - Fiber 树的根节点
│  stateNode: ────┼────▶ FiberRootNode（互相引用）
└─────────────────┘


Step 2: 开始渲染 - 处理 HostRoot
─────────────────────────

// 创建 workInProgress 树
workInProgress = createWorkInProgress(HostRoot Fiber, pendingProps);

// beginWork(HostRoot) 调用 reconcileChildren
// 传入的 children 是 <App />，即 App 的 ReactElement
reconcileChildren(HostRoot Fiber, null, <App />, lanes);


Step 3: reconcileChildren 创建 App Fiber
─────────────────────────

// ReactElement: { type: App, props: {}, key: null }
// ⭐ 调用 createFiberFromElement
const appFiber = createFiberFromElement(<App />, mode, lanes);
// 结果:
// appFiber.tag = IndeterminateComponent (首次渲染，还不知道是函数还是类)
// appFiber.type = App (函数引用)
// appFiber.pendingProps = {}

appFiber.return = HostRoot Fiber;
HostRoot Fiber.child = appFiber;

         ┌─────────────────┐
         │  HostRoot       │
         └────────┬────────┘
                  │ child
                  ▼
         ┌─────────────────┐
         │  App Fiber      │  (workInProgress)
         │  tag: 2         │  IndeterminateComponent
         │  type: App      │
         └─────────────────┘


Step 4: beginWork(App Fiber) - 执行函数组件
─────────────────────────

// 执行 App()，返回:
// <div>
//   <h1>title</h1>
//   <Child />
// </div>

// 这个 JSX 被编译为 ReactElement:
{
  type: 'div',
  props: {
    children: [
      { type: 'h1', props: { children: 'title' } },
      { type: Child, props: {} }
    ]
  }
}

// ⭐ 此时确定 App 是函数组件
appFiber.tag = FunctionComponent;  // 从 2 改为 0

// reconcileChildren 创建 div Fiber
const divFiber = createFiberFromElement(<div>...</div>, mode, lanes);
divFiber.return = appFiber;
appFiber.child = divFiber;


Step 5: 继续向下 - 处理 div 的 children
─────────────────────────

// beginWork(div Fiber)
// reconcileChildren 处理 [<h1>title</h1>, <Child />]

// 创建 h1 Fiber
const h1Fiber = createFiberFromElement(<h1>title</h1>, mode, lanes);
h1Fiber.return = divFiber;
h1Fiber.index = 0;

// 创建 Child Fiber
const childFiber = createFiberFromElement(<Child />, mode, lanes);
childFiber.return = divFiber;
childFiber.index = 1;

// 建立兄弟关系
h1Fiber.sibling = childFiber;

// 设置 div 的第一个子节点
divFiber.child = h1Fiber;


Step 6: 处理 h1 - 创建 Text Fiber
─────────────────────────

// beginWork(h1 Fiber)
// children 是 'title'（字符串）

const textFiber = createFiberFromText('title', mode, lanes);
textFiber.return = h1Fiber;
textFiber.tag = HostText;  // 6

h1Fiber.child = textFiber;


Step 7: 处理 Child 组件
─────────────────────────

// beginWork(Child Fiber)
// 执行 Child()，返回 <span>child</span>

// 创建 span Fiber
const spanFiber = createFiberFromElement(<span>child</span>, mode, lanes);
spanFiber.return = childFiber;

// 继续创建 Text Fiber
const childTextFiber = createFiberFromText('child', mode, lanes);
childTextFiber.return = spanFiber;


最终的 Fiber 树:
─────────────────────────

         ┌─────────────────┐
         │  FiberRoot      │
         └────────┬────────┘
                  │ current
                  ▼
         ┌─────────────────┐
         │  HostRoot (3)   │
         └────────┬────────┘
                  │ child
                  ▼
         ┌─────────────────┐
         │  App (0)        │  FunctionComponent
         │  type: App      │
         └────────┬────────┘
                  │ child
                  ▼
         ┌─────────────────┐
         │  div (5)        │  HostComponent
         │  type: 'div'    │
         └────────┬────────┘
                  │ child
                  ▼
  ┌─────────────────┐       ┌─────────────────┐
  │  h1 (5)         │─sibling─▶│  Child (0)      │
  │  type: 'h1'     │       │  type: Child    │
  └────────┬────────┘       └────────┬────────┘
           │ child                    │ child
           ▼                          ▼
  ┌─────────────────┐       ┌─────────────────┐
  │  Text (6)       │       │  span (5)       │
  │  "title"        │       │  type: 'span'   │
  └─────────────────┘       └────────┬────────┘
                                     │ child
                                     ▼
                            ┌─────────────────┐
                            │  Text (6)       │
                            │  "child"        │
                            └─────────────────┘
`;

// ============================================================
// Part 3: 遍历顺序
// ============================================================

/**
 * 📊 Fiber 树的遍历顺序
 */

const traversalOrder = `
📊 Fiber 树的遍历顺序（深度优先）

┌─────────────────────────────────────────────────────────────────────────────┐
│                       遍历规则                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   beginWork 阶段（向下）:                                                   │
│   1. 处理当前节点                                                           │
│   2. 如果有 child，进入 child                                               │
│   3. 继续步骤 1                                                             │
│                                                                             │
│   completeWork 阶段（向上）:                                                │
│   1. 当没有 child 或 child 处理完了，完成当前节点                           │
│   2. 如果有 sibling，beginWork(sibling)                                     │
│   3. 如果没有 sibling，completeWork(return)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

遍历顺序示意:
═══════════════════════════════════════════════════════════════════════════════

         ┌─────────────────┐
         │  HostRoot       │  ① beginWork
         └────────┬────────┘  ⑯ completeWork
                  │
                  ▼
         ┌─────────────────┐
         │  App            │  ② beginWork
         └────────┬────────┘  ⑮ completeWork
                  │
                  ▼
         ┌─────────────────┐
         │  div            │  ③ beginWork
         └────────┬────────┘  ⑭ completeWork
                  │
                  ▼
  ┌─────────────────┐       ┌─────────────────┐
  │  h1             │──────▶│  Child          │
  │  ④ begin       │       │  ⑧ begin        │
  │  ⑦ complete    │       │  ⑬ complete     │
  └────────┬────────┘       └────────┬────────┘
           │                          │
           ▼                          ▼
  ┌─────────────────┐       ┌─────────────────┐
  │  Text "title"   │       │  span           │
  │  ⑤ begin       │       │  ⑨ begin       │
  │  ⑥ complete    │       │  ⑫ complete     │
  └─────────────────┘       └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Text "child"   │
                            │  ⑩ begin       │
                            │  ⑪ complete     │
                            └─────────────────┘


工作循环伪代码:
─────────────────────────

function workLoop() {
  while (workInProgress !== null) {
    performUnitOfWork(workInProgress);
  }
}

function performUnitOfWork(fiber) {
  // ========== beginWork 阶段 ==========
  const next = beginWork(fiber);  // 返回 child 或 null

  if (next !== null) {
    // 有 child，继续向下
    workInProgress = next;
  } else {
    // 没有 child，开始 completeWork
    completeUnitOfWork(fiber);
  }
}

function completeUnitOfWork(fiber) {
  let completedWork = fiber;

  while (completedWork !== null) {
    // ========== completeWork 阶段 ==========
    completeWork(completedWork);

    const sibling = completedWork.sibling;
    if (sibling !== null) {
      // 有 sibling，beginWork(sibling)
      workInProgress = sibling;
      return;
    }

    // 没有 sibling，继续向上完成 parent
    completedWork = completedWork.return;
  }

  // 到达根节点，整棵树处理完成
  workInProgress = null;
}
`;

// ============================================================
// Part 4: 真实案例演示
// ============================================================

/**
 * 📊 完整的构建过程演示
 */

const fullBuildDemo = `
📊 完整的 Fiber 树构建过程演示

组件代码:
─────────────────────────

function Child() {
  return <span>child</span>;
}

function App() {
  return (
    <div>
      <h1>title</h1>
      <Child />
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);


详细执行过程:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step │ 当前 Fiber  │ 操作          │ 说明                                   │
├──────┼─────────────┼───────────────┼────────────────────────────────────────┤
│  1   │ HostRoot    │ beginWork     │ 创建 App Fiber，设为 child             │
│  2   │ App         │ beginWork     │ 执行 App()，创建 div Fiber             │
│  3   │ div         │ beginWork     │ 创建 h1 + Child Fiber                  │
│  4   │ h1          │ beginWork     │ 创建 Text("title") Fiber               │
│  5   │ Text        │ beginWork     │ 叶子节点，无 child                     │
│  6   │ Text        │ completeWork  │ 创建真实 Text 节点                     │
│  7   │ h1          │ completeWork  │ 创建 <h1> DOM，插入 Text 节点          │
│      │             │               │ 有 sibling → 转到 Child                │
│  8   │ Child       │ beginWork     │ 执行 Child()，创建 span Fiber          │
│  9   │ span        │ beginWork     │ 创建 Text("child") Fiber               │
│ 10   │ Text        │ beginWork     │ 叶子节点，无 child                     │
│ 11   │ Text        │ completeWork  │ 创建真实 Text 节点                     │
│ 12   │ span        │ completeWork  │ 创建 <span> DOM，插入 Text 节点        │
│ 13   │ Child       │ completeWork  │ 函数组件无 DOM，向上                   │
│ 14   │ div         │ completeWork  │ 创建 <div> DOM，插入 h1 + span         │
│ 15   │ App         │ completeWork  │ 函数组件无 DOM，向上                   │
│ 16   │ HostRoot    │ completeWork  │ 根节点完成，准备 Commit                │
└──────┴─────────────┴───────────────┴────────────────────────────────────────┘


关键函数调用链:
─────────────────────────

beginWork(HostRoot)
  └─→ reconcileChildren(null, <App />)
      └─→ reconcileSingleElement(<App />)
          └─→ createFiberFromElement(<App />)
              └─→ createFiberFromTypeAndProps(App, null, {}, ...)
                  └─→ createFiber(IndeterminateComponent, {}, null, mode)

beginWork(App)
  └─→ renderWithHooks(App, {})  // 执行 App()
  └─→ reconcileChildren(null, <div>...</div>)
      └─→ createFiberFromElement(<div>...</div>)

beginWork(div)
  └─→ reconcileChildren(null, [<h1>...</h1>, <Child />])
      └─→ reconcileChildrenArray(...)
          └─→ createFiberFromElement(<h1>...</h1>)  // 创建 h1
          └─→ createFiberFromElement(<Child />)     // 创建 Child
          └─→ h1Fiber.sibling = childFiber          // 建立兄弟关系

// ... 继续向下
`;

// ============================================================
// Part 5: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 3 面试要点

Q1: ReactElement 是如何转换成 Fiber 的？
A: 通过 createFiberFromElement 函数：
   1. 提取 element 的 type、key、props
   2. 调用 createFiberFromTypeAndProps
   3. 根据 type 类型决定 Fiber 的 tag
   4. 调用 createFiber 创建 FiberNode

Q2: 如何判断组件是函数组件还是类组件？
A: 在 createFiberFromTypeAndProps 中：
   - 检查 type.prototype.isReactComponent
   - 如果存在，是类组件（ClassComponent）
   - 否则是函数组件（或 IndeterminateComponent）

Q3: Fiber 树是如何构建的？
A: 深度优先遍历：
   1. beginWork 处理当前节点，创建子 Fiber
   2. 如果有 child，进入 child 继续 beginWork
   3. 没有 child，执行 completeWork
   4. 如果有 sibling，转到 sibling 的 beginWork
   5. 没有 sibling，向上执行 parent 的 completeWork

Q4: beginWork 和 completeWork 分别做什么？
A: - beginWork（向下）：
     - 执行组件函数/类的 render
     - 调用 reconcileChildren 创建子 Fiber
     - 返回下一个要处理的 Fiber（child 或 null）
   - completeWork（向上）：
     - 为 HostComponent 创建真实 DOM
     - 收集副作用（flags）
     - 准备 Commit 阶段

Q5: Fiber 树的遍历为什么是深度优先？
A: 因为 React 需要：
   1. 先处理子组件，才能确定父组件的 children
   2. 子组件的 DOM 需要先创建，才能插入到父 DOM
   3. 从叶子节点向上收集副作用
`;

export {
  createFiberFunctions,
  mountProcess,
  traversalOrder,
  fullBuildDemo,
  interviewPoints,
};

