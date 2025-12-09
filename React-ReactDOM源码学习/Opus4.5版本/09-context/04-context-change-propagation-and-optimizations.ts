/**
 * ============================================================
 * 📚 Phase 9: Context 与跨组件状态传播 - Part 4: Context 变更传播与优化
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-reconciler/src/ReactFiberNewContext.new.js
 *   - propagateContextChange (Line 198)
 *   - propagateContextChange_eager (Line 219)
 *   - scheduleContextWorkOnParentPath (Line 156)
 *
 * ⏱️ 预计时间：2-3 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: Context 变更触发更新的整体流程
// ============================================================

/**
 * 📊 Context 变更的整体流程
 */

const contextChangePropagationOverview = `
📊 Context 变更触发更新的整体流程

场景
═══════════════════════════════════════════════════════════════════════════════

<ThemeContext.Provider value={newValue}>  ← value 发生变化
  <ComponentTree />
</ThemeContext.Provider>

当 Provider 的 value 变化时，需要：
1. 找到所有依赖此 Context 的组件
2. 标记它们需要更新
3. 触发重新渲染


整体流程图
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Provider value 变化                                                       │
│           │                                                                 │
│           ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ updateContextProvider (beginWork 阶段)                              │  │
│   │                                                                     │  │
│   │ 1. pushProvider(newValue) → 设置 context._currentValue             │  │
│   │ 2. 比较 oldValue 和 newValue (Object.is)                           │  │
│   │ 3. 如果不同 → propagateContextChange()                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ propagateContextChange_eager (遍历子树)                             │  │
│   │                                                                     │  │
│   │ 深度优先遍历 Provider 的子 Fiber 树：                               │  │
│   │ • 检查每个 Fiber 的 dependencies.firstContext 链表                 │  │
│   │ • 如果发现依赖此 Context → 标记需要更新                            │  │
│   │ • 向上标记 childLanes（scheduleContextWorkOnParentPath）           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ 被标记的 Fiber 在后续 WorkLoop 中重新渲染                           │  │
│   │                                                                     │  │
│   │ • fiber.lanes 被合并了 renderLanes                                  │  │
│   │ • 祖先节点的 childLanes 被标记                                      │  │
│   │ • 确保 bailout 检查不会跳过这些组件                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: propagateContextChange_eager 详解
// ============================================================

/**
 * 📊 propagateContextChange_eager 实现
 */

const propagateContextChangeEager = `
📊 propagateContextChange_eager 实现

源码位置: ReactFiberNewContext.new.js (Line 219-354)
═══════════════════════════════════════════════════════════════════════════════

function propagateContextChange_eager<T>(
  workInProgress: Fiber,    // Provider Fiber
  context: ReactContext<T>, // 变化的 Context
  renderLanes: Lanes,       // 渲染优先级
): void {
  let fiber = workInProgress.child;

  // 深度优先遍历整个子树
  while (fiber !== null) {
    let nextFiber;

    // ⭐ 检查这个 Fiber 是否依赖此 Context
    const list = fiber.dependencies;
    if (list !== null) {
      nextFiber = fiber.child;

      // 遍历依赖链表
      let dependency = list.firstContext;
      while (dependency !== null) {
        // ⭐ 找到匹配的 Context！
        if (dependency.context === context) {

          // 1. 对于 ClassComponent，添加 ForceUpdate
          if (fiber.tag === ClassComponent) {
            const update = createUpdate(NoTimestamp, lane);
            update.tag = ForceUpdate;
            enqueueUpdate(fiber, update);
          }

          // 2. 合并 lanes，标记需要更新
          fiber.lanes = mergeLanes(fiber.lanes, renderLanes);
          if (fiber.alternate !== null) {
            fiber.alternate.lanes = mergeLanes(fiber.alternate.lanes, renderLanes);
          }

          // 3. 向上标记 childLanes
          scheduleContextWorkOnParentPath(fiber.return, renderLanes, workInProgress);

          // 4. 标记依赖列表的 lanes
          list.lanes = mergeLanes(list.lanes, renderLanes);

          break; // 找到就退出内层循环
        }
        dependency = dependency.next;
      }
    } else if (fiber.tag === ContextProvider) {
      // ⭐ 特殊处理：遇到相同 Context 的 Provider，停止向下搜索
      // 因为内层 Provider 会覆盖外层的值
      nextFiber = fiber.type === workInProgress.type ? null : fiber.child;
    } else {
      nextFiber = fiber.child;
    }

    // 继续遍历（深度优先）
    if (nextFiber !== null) {
      nextFiber.return = fiber;
    } else {
      // 没有子节点，回溯找兄弟节点
      nextFiber = fiber;
      while (nextFiber !== null) {
        if (nextFiber === workInProgress) {
          nextFiber = null;
          break;
        }
        const sibling = nextFiber.sibling;
        if (sibling !== null) {
          sibling.return = nextFiber.return;
          nextFiber = sibling;
          break;
        }
        nextFiber = nextFiber.return;
      }
    }
    fiber = nextFiber;
  }
}


关键理解
═══════════════════════════════════════════════════════════════════════════════

1. 遍历策略：深度优先遍历 Provider 的整个子树
2. 匹配方式：检查 fiber.dependencies.firstContext 链表
3. 标记方式：合并 lanes 到 fiber.lanes 和 alternate.lanes
4. 向上传播：调用 scheduleContextWorkOnParentPath 标记 childLanes
5. 内层 Provider 优化：遇到相同 Context 的 Provider 时停止搜索
`;

// ============================================================
// Part 3: scheduleContextWorkOnParentPath - 向上标记
// ============================================================

/**
 * 📊 scheduleContextWorkOnParentPath - 向上标记 childLanes
 */

const scheduleContextWorkOnParentPath = `
📊 scheduleContextWorkOnParentPath - 向上标记 childLanes

源码位置: ReactFiberNewContext.new.js (Line 156-196)
═══════════════════════════════════════════════════════════════════════════════

export function scheduleContextWorkOnParentPath(
  parent: Fiber | null,
  renderLanes: Lanes,
  propagationRoot: Fiber,
) {
  let node = parent;
  while (node !== null) {
    const alternate = node.alternate;

    // 如果 childLanes 还没有包含 renderLanes，就合并进去
    if (!isSubsetOfLanes(node.childLanes, renderLanes)) {
      node.childLanes = mergeLanes(node.childLanes, renderLanes);
      if (alternate !== null) {
        alternate.childLanes = mergeLanes(alternate.childLanes, renderLanes);
      }
    } else if (
      alternate !== null &&
      !isSubsetOfLanes(alternate.childLanes, renderLanes)
    ) {
      alternate.childLanes = mergeLanes(alternate.childLanes, renderLanes);
    } else {
      // 已经标记过了，通常可以停止
      // 但在 offscreen/fallback 树中可能需要继续
    }

    if (node === propagationRoot) {
      break;
    }
    node = node.return;
  }
}


为什么要标记 childLanes？
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   标记 childLanes 的目的：防止 bailout 跳过需要更新的子树                   │
│                                                                             │
│   bailoutOnAlreadyFinishedWork 中的检查:                                    │
│   ─────────────────────────────────────                                     │
│                                                                             │
│   function bailoutOnAlreadyFinishedWork(...) {                              │
│     // 检查子树是否有工作需要做                                             │
│     if (!includesSomeLane(renderLanes, workInProgress.childLanes)) {        │
│       // 没有子节点需要更新，完全跳过子树                                   │
│       return null;  // ← 如果不标记 childLanes，会错误地跳过！              │
│     }                                                                       │
│     // 克隆子节点，继续处理                                                 │
│     cloneChildFibers(current, workInProgress);                              │
│     return workInProgress.child;                                            │
│   }                                                                         │
│                                                                             │
│   如果不标记 childLanes：                                                   │
│   • 中间组件可能 bailout（因为 props/state 没变）                          │
│   • 它的 childLanes = 0，React 认为子树不需要更新                          │
│   • 依赖 Context 的深层组件就不会被访问到！                                 │
│                                                                             │
│   标记 childLanes 后：                                                      │
│   • 即使中间组件 bailout，也会检查 childLanes                               │
│   • 发现子树有工作需要做 → 继续向下遍历                                     │
│   • 依赖 Context 的组件会被正确访问和更新                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


图示: childLanes 的作用
═══════════════════════════════════════════════════════════════════════════════

Provider (value: "A" → "B")
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ IntermediateComponent                                             │
│                                                                   │
│ props 没变, state 没变 → 想 bailout                               │
│                                                                   │
│ 检查: childLanes 有 renderLanes 吗？                              │
│       ↓                                                           │
│ 如果 childLanes 被标记 → YES → 继续处理子树                       │
│ 如果 childLanes = 0    → NO  → 跳过子树（错误！）                 │
│                                                                   │
│ childLanes: 被 scheduleContextWorkOnParentPath 标记了！           │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ ContextConsumer (依赖 Provider 的 Context)                        │
│                                                                   │
│ fiber.lanes 被标记 → 会被重新渲染                                 │
│ useContext 读取新值 "B"                                           │
└───────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 4: 案例分析 - 深层嵌套的 Context 更新
// ============================================================

/**
 * 📊 案例 B: 深层嵌套的 Context 更新
 */

const deepNestedContextUpdate = `
📊 案例 B: 深层嵌套的 Context 更新

示例代码
═══════════════════════════════════════════════════════════════════════════════

const UserContext = createContext({ name: 'Guest' });

function App() {
  const [user, setUser] = useState({ name: 'Guest' });

  return (
    <UserContext.Provider value={user}>
      <Header />
      <Main />
      <button onClick={() => setUser({ name: 'John' })}>Login</button>
    </UserContext.Provider>
  );
}

// 中间组件，不使用 Context
function Header() {
  console.log('Header render');
  return (
    <header>
      <Navigation />
    </header>
  );
}

function Navigation() {
  console.log('Navigation render');
  return <nav>...</nav>;
}

// 深层组件，使用 Context
function Main() {
  console.log('Main render');
  return (
    <main>
      <Sidebar />
      <Content />
    </main>
  );
}

function Sidebar() {
  console.log('Sidebar render');
  return <aside>...</aside>;
}

function Content() {
  console.log('Content render');
  return (
    <section>
      <UserProfile />  {/* 使用 Context */}
    </section>
  );
}

function UserProfile() {
  const user = useContext(UserContext);  // ⭐ 依赖 UserContext
  console.log('UserProfile render');
  return <div>Welcome, {user.name}</div>;
}


点击 Login 后的更新过程
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   1. setUser 触发 App 重新渲染                                              │
│      scheduleUpdateOnFiber(AppFiber, ...)                                   │
│                                                                             │
│   2. beginWork(App)                                                         │
│      - 执行 App()，返回新的 children                                        │
│      - Provider 的 value 变了: { name: 'Guest' } → { name: 'John' }        │
│                                                                             │
│   3. beginWork(ContextProvider)                                             │
│      - updateContextProvider 检测到 value 变化                              │
│      - 调用 propagateContextChange_eager                                    │
│                                                                             │
│   4. propagateContextChange_eager 遍历子树:                                 │
│      ─────────────────────────────────────────────────────────────────      │
│      Header → dependencies: null → 继续向下                                 │
│        Navigation → dependencies: null → 没有子节点了，回溯                 │
│      Main → dependencies: null → 继续向下                                   │
│        Sidebar → dependencies: null → 没有子节点了，回溯                    │
│        Content → dependencies: null → 继续向下                              │
│          UserProfile → dependencies: { firstContext: UserContext }          │
│            ⭐ 找到匹配！                                                    │
│            - 标记 UserProfile.lanes = renderLanes                           │
│            - scheduleContextWorkOnParentPath(Content, ...)                  │
│                                                                             │
│   5. scheduleContextWorkOnParentPath 向上标记:                              │
│      UserProfile ← 已标记 lanes                                             │
│      Content.childLanes = renderLanes                                       │
│      Main.childLanes = renderLanes                                          │
│      Provider.childLanes = renderLanes                                      │
│                                                                             │
│   6. 继续 WorkLoop:                                                         │
│      - Header: props 没变 → bailout，但 childLanes = 0 → 完全跳过子树      │
│      - Main: props 没变 → 想 bailout，但 childLanes ≠ 0 → 继续处理         │
│        - Sidebar: childLanes = 0 → 跳过                                     │
│        - Content: childLanes ≠ 0 → 继续处理                                │
│          - UserProfile: lanes ≠ 0 → 重新渲染！                              │
│                                                                             │
│   7. 结果:                                                                  │
│      只有 App, Provider, UserProfile 重新执行                               │
│      Header, Navigation, Main, Sidebar, Content 都 bailout（跳过执行）      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


控制台输出对比
═══════════════════════════════════════════════════════════════════════════════

初次渲染:
─────────────────────────────────────────────────────────────────
Header render
Navigation render
Main render
Sidebar render
Content render
UserProfile render

点击 Login 后（Context 更新）:
─────────────────────────────────────────────────────────────────
UserProfile render    ← 只有 UserProfile 重新执行！

⭐ 关键：中间组件（Header, Main, Content 等）没有重新执行！
   React 通过 childLanes 机制实现了"精准更新"。
`;

// ============================================================
// Part 5: 性能优化注意事项
// ============================================================

/**
 * 📊 Context 性能优化注意事项
 */

const performanceOptimizations = `
📊 Context 性能优化注意事项

问题 1: Provider value 每次都是新对象
═══════════════════════════════════════════════════════════════════════════════

❌ 错误写法：

function App() {
  const [name, setName] = useState('John');

  return (
    // ⚠️ 每次渲染都创建新对象！
    <UserContext.Provider value={{ name, setName }}>
      <Content />
    </UserContext.Provider>
  );
}

问题：
- App 因任何原因重渲染时，value 是新对象
- { name, setName } !== { name, setName }（引用不同）
- propagateContextChange 被触发
- 所有消费者都会更新，即使 name 没变！


✅ 正确写法：

function App() {
  const [name, setName] = useState('John');

  // 使用 useMemo 缓存 value 对象
  const contextValue = useMemo(() => ({ name, setName }), [name]);

  return (
    <UserContext.Provider value={contextValue}>
      <Content />
    </UserContext.Provider>
  );
}


问题 2: 大 Context 导致过多组件更新
═══════════════════════════════════════════════════════════════════════════════

❌ 问题写法：

const AppContext = createContext({
  theme: 'light',
  user: null,
  locale: 'en',
  notifications: [],
  // ... 很多状态
});

// 任何一个值变化，所有消费者都更新


✅ 解决方案 1: 拆分 Context

const ThemeContext = createContext('light');
const UserContext = createContext(null);
const LocaleContext = createContext('en');

// 不同数据用不同 Context，互不影响


✅ 解决方案 2: 使用选择器（借助库）

// 使用 use-context-selector 等库
import { createContext, useContextSelector } from 'use-context-selector';

function UserName() {
  // 只订阅 name，其他属性变化不会触发更新
  const name = useContextSelector(UserContext, ctx => ctx.name);
  return <span>{name}</span>;
}


问题 3: 不必要的嵌套 Provider 重渲染
═══════════════════════════════════════════════════════════════════════════════

❌ 问题写法：

function App() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <button onClick={() => setCount(c => c + 1)}>{count}</button>
      {/* 每次 count 变化，ThemeProvider 都重新渲染 */}
      <ThemeProvider>
        <Content />
      </ThemeProvider>
    </div>
  );
}


✅ 正确写法：

function App() {
  return (
    <ThemeProvider>
      <Counter />
      <Content />
    </ThemeProvider>
  );
}

function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}

// 将 state 下移，避免影响 ThemeProvider
`;

// ============================================================
// Part 6: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 4 面试要点

Q1: Context 值变化时，React 是如何找到依赖它的组件的？
A: 通过 propagateContextChange_eager：
   1. 深度优先遍历 Provider 的整个子树
   2. 检查每个 Fiber 的 dependencies.firstContext 链表
   3. 如果发现依赖此 Context，标记 fiber.lanes
   4. 调用 scheduleContextWorkOnParentPath 向上标记 childLanes

Q2: 为什么要向上标记 childLanes？
A: 防止 bailout 跳过需要更新的子树。
   中间组件可能因为 props/state 没变而 bailout，
   但如果 childLanes 被标记，bailout 时会检查到子树有工作，
   继续向下遍历，确保依赖 Context 的组件被访问到。

Q3: 遇到嵌套的相同 Context Provider 时会怎样？
A: 会停止向该分支继续搜索。
   因为内层 Provider 会覆盖外层的值，
   内层 Provider 的消费者不受外层值变化的影响。

Q4: Context 更新是否会导致整棵树重渲染？
A: 不会。React 通过 lanes + childLanes 机制实现精准更新：
   - 只有依赖此 Context 的组件会重新执行
   - 中间组件可以 bailout（但会检查 childLanes）
   - 不依赖此 Context 的分支完全跳过

Q5: 使用 Context 有哪些性能优化建议？
A: 1. 用 useMemo 缓存 Provider 的 value 对象
   2. 按更新频率拆分成多个 Context
   3. 将频繁变化的 state 下移，避免影响 Provider
   4. 考虑使用选择器库（如 use-context-selector）

Q6: Object.is 比较值变化，如果传新对象但内容相同会怎样？
A: 会触发更新！Object.is 比较的是引用，不是深度比较。
   这就是为什么要用 useMemo 缓存 value 对象。
`;

export {
  contextChangePropagationOverview,
  propagateContextChangeEager,
  scheduleContextWorkOnParentPath,
  deepNestedContextUpdate,
  performanceOptimizations,
  interviewPoints,
};

