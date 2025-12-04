/**
 * ============================================================
 * 📚 Phase 7: 并发特性 - Part 4: 真实案例详解
 * ============================================================
 *
 * 本文件通过 2 个真实场景详细讲解并发特性的工作原理
 */

// ============================================================
// 案例 A: 搜索输入框 + 大列表过滤
// ============================================================

/**
 * 📊 场景描述
 */

const caseA_Description = `
📊 案例 A: 搜索输入框 + 大列表过滤

场景：
  - 用户在输入框中输入搜索词
  - 需要实时过滤 10000 条数据并显示结果

问题（不使用并发特性）：
  - 输入框卡顿
  - 用户体验差

解决方案：
  - 输入框更新：高优先级，立即响应
  - 列表过滤：低优先级，可延迟
`;

/**
 * 📊 组件代码
 */

const caseA_Code = `
📊 组件代码

import { useState, useTransition, useMemo } from 'react';

// 模拟大数据
const generateData = (count) => 
  Array.from({ length: count }, (_, i) => ({
    id: i,
    name: \`Item \${i}\`,
    description: \`Description for item \${i}\`
  }));

const ALL_DATA = generateData(10000);

function SearchableList() {
  // 输入框的值（高优先级）
  const [query, setQuery] = useState('');
  
  // 用于过滤的查询词（低优先级）
  const [deferredQuery, setDeferredQuery] = useState('');
  
  // isPending 用于显示加载状态
  const [isPending, startTransition] = useTransition();
  
  // 过滤数据
  const filteredData = useMemo(() => {
    return ALL_DATA.filter(item => 
      item.name.toLowerCase().includes(deferredQuery.toLowerCase())
    );
  }, [deferredQuery]);
  
  function handleChange(e) {
    const value = e.target.value;
    
    // ⭐ 高优先级：输入框立即更新
    setQuery(value);
    
    // ⭐ 低优先级：列表稍后更新
    startTransition(() => {
      setDeferredQuery(value);
    });
  }
  
  return (
    <div>
      <input 
        value={query} 
        onChange={handleChange} 
        placeholder="Search..."
      />
      
      {isPending && <div className="loading">Filtering...</div>}
      
      <ul style={{ opacity: isPending ? 0.7 : 1 }}>
        {filteredData.slice(0, 100).map(item => (
          <li key={item.id}>
            {item.name} - {item.description}
          </li>
        ))}
      </ul>
      
      <div>Showing {Math.min(100, filteredData.length)} of {filteredData.length} results</div>
    </div>
  );
}
`;

/**
 * 📊 时间线详解
 */

const caseA_Timeline = `
📊 案例 A 时间线

用户输入 "a" → "ab" 的完整流程
─────────────────────────────────────

时间 (ms)
0     5    10    15    20    25    30    35    40    45    50
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤

T=0ms: 用户输入 'a'
       │
       ├── handleChange('a') 触发
       │
       ├── setQuery('a')
       │   │
       │   └── scheduleUpdateOnFiber()
       │       └── lane = SyncLane / DefaultLane（高优先级）
       │
       └── startTransition(() => setDeferredQuery('a'))
           │
           ├── ReactCurrentBatchConfig.transition = {}
           │
           └── setDeferredQuery('a')
               │
               └── scheduleUpdateOnFiber()
                   └── requestUpdateLane()
                       └── 检测到 transition !== null
                           └── lane = TransitionLane1（低优先级）

T=0.5ms: ensureRootIsScheduled()
         │
         ├── 发现两个不同优先级的更新
         │
         ├── 高优先级 (SyncLane/DefaultLane)
         │   └── 立即调度
         │
         └── 低优先级 (TransitionLane)
             └── 稍后处理

T=1ms: 执行高优先级渲染
       │
       └── renderRootSync() 或 renderRootConcurrent()
           │
           └── 只处理 SyncLane 的更新
               │
               └── query = 'a' ✅
                   deferredQuery = '' (未更新)

T=2ms: commitRoot()
       │
       └── 输入框显示 'a' ✅
           isPending = true ✅ (startTransition 设置)

T=3ms: 开始处理低优先级更新
       │
       └── performConcurrentWorkOnRoot()
           │
           └── shouldTimeSlice = true (TransitionLane 可中断)
               │
               └── renderRootConcurrent()

T=5ms: 渲染中... (处理 Fiber 树)
       │
       └── workLoopConcurrent()
           └── 每处理一个 Fiber 检查 shouldYield()

T=10ms: ⚠️ 用户继续输入 'ab'
        │
        ├── handleChange('ab') 触发
        │
        ├── setQuery('ab')
        │   └── lane = SyncLane（高优先级）
        │
        └── startTransition(() => setDeferredQuery('ab'))
            └── lane = TransitionLane2（新的低优先级）

T=10.5ms: ensureRootIsScheduled()
          │
          ├── 发现新的高优先级更新
          │
          └── 取消当前的低优先级渲染！
              cancelCallback(existingCallbackNode)

T=11ms: 执行高优先级渲染
        │
        └── query = 'ab' ✅
            isPending = true（保持）

T=12ms: commitRoot()
        │
        └── 输入框显示 'ab' ✅

T=13ms: 重新开始低优先级渲染
        │
        └── 这次渲染 'ab' 的结果（旧的 'a' 结果被丢弃）

T=13-30ms: workLoopConcurrent()
           │
           └── 分多个时间片渲染

T=35ms: 低优先级渲染完成
        │
        └── deferredQuery = 'ab' ✅
            filteredData = [...] ✅

T=36ms: commitRoot()
        │
        └── 列表更新显示 'ab' 的过滤结果
            isPending = false ✅

关键点：
1. 输入框始终立即响应（T=2ms, T=12ms）
2. 旧的过滤结果被丢弃，避免过时渲染
3. isPending 在整个过渡期间为 true
`;

/**
 * 📊 函数调用链
 */

const caseA_CallStack = `
📊 案例 A 函数调用链

高优先级更新路径 (setQuery):
─────────────────────────────────────

onChange()
└── setQuery('a')
    └── dispatchSetState()
        📁 ReactFiberHooks.new.js:2476
        │
        └── scheduleUpdateOnFiber(fiber, lane)
            📁 ReactFiberWorkLoop.new.js:533
            │
            ├── markRootUpdated(root, lane)
            │   root.pendingLanes |= lane
            │
            └── ensureRootIsScheduled(root)
                📁 ReactFiberWorkLoop.new.js:696
                │
                └── scheduleSyncCallback() 或 scheduleCallback()

低优先级更新路径 (startTransition):
─────────────────────────────────────

startTransition()
📁 ReactFiberHooks.new.js:2002
│
├── setCurrentUpdatePriority(ContinuousEventPriority)
│
├── setPending(true)  // 高优先级
│   └── 立即设置 isPending = true
│
├── ReactCurrentBatchConfig.transition = {}
│
├── setPending(false) // 低优先级（在 transition 内）
│   └── 会延迟执行
│
└── setDeferredQuery('a')
    └── dispatchSetState()
        │
        └── requestUpdateLane(fiber)
            📁 ReactFiberWorkLoop.new.js
            │
            └── 检查 ReactCurrentBatchConfig.transition
                │
                └── !== null → claimNextTransitionLane()
                    📁 ReactFiberLane.new.js:493
                    └── return TransitionLane

渲染路径:
─────────────────────────────────────

performConcurrentWorkOnRoot(root)
📁 ReactFiberWorkLoop.new.js:829
│
├── getNextLanes(root)
│   📁 ReactFiberLane.new.js:232
│   └── 返回最高优先级的 lanes
│
├── shouldTimeSlice = !includesBlockingLane(lanes)
│
├── renderRootConcurrent(root, lanes)
│   📁 ReactFiberWorkLoop.new.js:1748
│   │
│   └── workLoopConcurrent()
│       📁 ReactFiberWorkLoop.new.js:1829
│       │
│       └── while (!shouldYield()) {
│             performUnitOfWork(workInProgress)
│           }
│
└── commitRoot(root)
    📁 ReactFiberWorkLoop.new.js:2044
`;

// ============================================================
// 案例 B: Suspense + 异步数据加载
// ============================================================

/**
 * 📊 场景描述
 */

const caseB_Description = `
📊 案例 B: Suspense + 异步数据加载

场景：
  - 用户切换 Tab 加载不同数据
  - 数据加载期间显示 loading
  - 使用 Transition 避免闪烁

问题（不使用 Transition）：
  - 切换 Tab 立即显示 Loading
  - 造成"闪烁"，体验差

解决方案：
  - 使用 startTransition 包裹 Tab 切换
  - 保持旧内容直到新内容就绪
`;

/**
 * 📊 组件代码
 */

const caseB_Code = `
📊 组件代码

import { Suspense, useState, useTransition } from 'react';

// 模拟数据获取（需要配合支持 Suspense 的数据库）
// 这里使用简化的实现来说明原理
const cache = new Map();

function fetchData(key) {
  if (!cache.has(key)) {
    let status = 'pending';
    let result;
    const promise = new Promise(resolve => {
      setTimeout(() => {
        result = { data: \`Data for \${key}\` };
        status = 'resolved';
        resolve(result);
      }, 1000);
    });
    
    cache.set(key, {
      read() {
        if (status === 'pending') throw promise;  // ⭐ 挂起
        return result;
      }
    });
  }
  return cache.get(key).read();
}

// Tab 内容组件（会挂起）
function TabContent({ tabId }) {
  const data = fetchData(tabId);  // 可能抛出 Promise
  return <div>{data.data}</div>;
}

// 主组件
function TabsWithSuspense() {
  const [currentTab, setCurrentTab] = useState('home');
  const [isPending, startTransition] = useTransition();
  
  function selectTab(tabId) {
    // ⭐ 使用 startTransition 包裹
    startTransition(() => {
      setCurrentTab(tabId);
    });
  }
  
  return (
    <div>
      <div className="tabs">
        {['home', 'profile', 'settings'].map(tab => (
          <button 
            key={tab}
            onClick={() => selectTab(tab)}
            style={{ 
              fontWeight: currentTab === tab ? 'bold' : 'normal',
              opacity: isPending ? 0.7 : 1
            }}
          >
            {tab}
          </button>
        ))}
      </div>
      
      <Suspense fallback={<div>Loading...</div>}>
        <TabContent tabId={currentTab} />
      </Suspense>
    </div>
  );
}

// 不使用 Transition 的版本（对比）
function TabsWithoutTransition() {
  const [currentTab, setCurrentTab] = useState('home');
  
  return (
    <div>
      <div className="tabs">
        {['home', 'profile', 'settings'].map(tab => (
          <button 
            key={tab}
            onClick={() => setCurrentTab(tab)}  // 直接更新
          >
            {tab}
          </button>
        ))}
      </div>
      
      <Suspense fallback={<div>Loading...</div>}>
        <TabContent tabId={currentTab} />
      </Suspense>
    </div>
  );
}
`;

/**
 * 📊 行为对比
 */

const caseB_Comparison = `
📊 有无 Transition 的行为对比

没有 Transition:
─────────────────────────────────────

T=0ms:   用户点击 "profile" Tab
T=1ms:   setCurrentTab('profile')
T=2ms:   开始渲染 profile Tab
T=3ms:   TabContent 抛出 Promise（数据未就绪）
T=4ms:   Suspense 捕获，显示 fallback
         ┌──────────────────────────┐
         │     Loading...           │  ← 立即显示！
         └──────────────────────────┘
T=1000ms: Promise resolve，数据就绪
T=1001ms: 重新渲染 TabContent
T=1002ms: 显示实际内容
         ┌──────────────────────────┐
         │   Data for profile       │
         └──────────────────────────┘

问题：用户看到 Loading 闪现，体验差

有 Transition:
─────────────────────────────────────

T=0ms:   用户点击 "profile" Tab
T=1ms:   startTransition(() => setCurrentTab('profile'))
         │
         └── isPending = true
T=2ms:   开始渲染 profile Tab (低优先级)
T=3ms:   TabContent 抛出 Promise
T=4ms:   ⭐ React 检测到在 Transition 中
         ⭐ 不显示 fallback，保持旧内容！
         ┌──────────────────────────┐
         │   Data for home          │  ← 保持旧内容
         │   (opacity: 0.7)         │  ← 通过 isPending 显示加载中
         └──────────────────────────┘
T=1000ms: Promise resolve
T=1001ms: 重新渲染，这次数据就绪
T=1002ms: 平滑切换到新内容
         ┌──────────────────────────┐
         │   Data for profile       │
         │   (opacity: 1)           │
         └──────────────────────────┘
         isPending = false

优势：无闪烁，平滑过渡
`;

/**
 * 📊 Fiber 树变化
 */

const caseB_FiberTree = `
📊 Suspense 在 Fiber 树中的表现

Fiber 树结构:
─────────────────────────────────────

                    FiberRoot
                        │
                    App Fiber
                        │
                ┌───────┴───────┐
                │               │
           Tabs Fiber     Suspense Fiber ← SuspenseComponent tag
                │               │
        ┌───────┴───┐     ┌─────┴─────┐
        │           │     │           │
    Button     Button   Primary    Fallback
    Fiber      Fiber    (child)   (fallback)
                          │
                    TabContent
                      Fiber

Suspense Fiber 的关键属性:
─────────────────────────────────────

SuspenseFiber {
  tag: SuspenseComponent (13),
  
  // 子节点
  child: TabContent Fiber,
  
  // 状态（控制显示 primary 还是 fallback）
  memoizedState: SuspenseState | null,
  
  // SuspenseState 结构:
  // {
  //   dehydrated: null,      // SSR 相关
  //   treeContext: null,     // 树上下文
  //   retryLane: RetryLane,  // 重试的 Lane
  // }
}

挂起时的处理:
─────────────────────────────────────

📁 packages/react-reconciler/src/ReactFiberBeginWork.new.js

1. TabContent 渲染时抛出 Promise
   throw promise

2. React 向上查找最近的 Suspense 边界
   throwException(root, value, lane)
   📁 ReactFiberThrow.new.js

3. 标记 Suspense Fiber
   workInProgress.flags |= ShouldCapture

4. 根据是否在 Transition 中决定行为
   
   普通更新:
     - 立即渲染 fallback
     - 显示 Loading
   
   Transition 更新:
     - 保持显示 primary（旧内容）
     - 记录挂起状态
     - 等待 Promise resolve

5. Promise resolve 后
   - 调用 ping 函数
   - 调度 RetryLane 更新
   - 重新渲染

Promise resolve 触发重新渲染:
─────────────────────────────────────

promise.then(() => {
  // 📁 ReactFiberWorkLoop.new.js
  
  // 标记需要重试
  markRootPinged(root, pingedLanes);
  
  // 调度更新
  ensureRootIsScheduled(root);
});
`;

/**
 * 📊 函数调用链
 */

const caseB_CallStack = `
📊 案例 B 函数调用链

挂起发生时:
─────────────────────────────────────

TabContent render
└── fetchData('profile')
    └── throw promise  // ⭐ 挂起！
        │
        └── 被 try/catch 捕获
            📁 ReactFiberWorkLoop.new.js - handleError()
            │
            └── throwException(root, value, workInProgress, lane)
                📁 ReactFiberThrow.new.js:434
                │
                ├── 检查 value 是否是 Thenable (Promise)
                │
                ├── 向上查找 Suspense 边界
                │   let suspenseBoundary = getSuspenseFallbackDirty...
                │
                ├── 标记为需要捕获
                │   suspenseBoundary.flags |= ShouldCapture
                │
                └── 附加 Promise 的回调
                    attachPingListener(root, wakeable, lane)

Transition 中的特殊处理:
─────────────────────────────────────

📁 ReactFiberBeginWork.new.js - updateSuspenseComponent

function updateSuspenseComponent(current, workInProgress) {
  const nextProps = workInProgress.pendingProps;
  
  // 检查是否应该显示 fallback
  let showFallback = false;
  
  if (didSuspend) {
    // 发生了挂起
    
    if (isTransitionLane(renderLanes)) {
      // ⭐ 在 Transition 中
      // 不显示 fallback，保持旧内容
      showFallback = false;
    } else {
      // 普通更新
      // 显示 fallback
      showFallback = true;
    }
  }
  
  if (showFallback) {
    // 渲染 fallback 子树
    return mountSuspenseFallbackChildren(...)
  } else {
    // 渲染 primary 子树
    return mountSuspensePrimaryChildren(...)
  }
}

Promise resolve 后:
─────────────────────────────────────

promise.then(resolve)
└── resolve()
    │
    └── pingSuspendedRoot(root, wakeable, pingedLanes)
        📁 ReactFiberWorkLoop.new.js:2972
        │
        ├── markRootPinged(root, pingedLanes)
        │   root.pingedLanes |= pingedLanes
        │
        └── ensureRootIsScheduled(root)
            └── scheduleCallback(priority, performConcurrentWorkOnRoot)
                └── 重新渲染，这次数据就绪
`;

// ============================================================
// Part 3: 总结
// ============================================================

const caseSummary = `
📊 案例总结

┌─────────────────────────────────────────────────────────────────────────────┐
│ 案例 A: 搜索输入框 + 列表过滤                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 核心 API: useTransition / startTransition                                   │
│                                                                             │
│ 关键点:                                                                     │
│ 1. 输入框使用高优先级（立即响应）                                            │
│ 2. 列表过滤使用低优先级（可延迟）                                            │
│ 3. 新输入会取消旧的过滤渲染                                                 │
│ 4. isPending 显示加载状态                                                   │
│                                                                             │
│ 涉及源码:                                                                   │
│ - ReactFiberHooks.new.js: startTransition, mountTransition                 │
│ - ReactFiberLane.new.js: claimNextTransitionLane                           │
│ - ReactFiberWorkLoop.new.js: shouldTimeSlice, ensureRootIsScheduled        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 案例 B: Suspense + 异步数据加载                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 核心 API: Suspense + useTransition                                          │
│                                                                             │
│ 关键点:                                                                     │
│ 1. Transition 中的 Suspense 不会立即显示 fallback                           │
│ 2. 保持旧内容直到新内容就绪                                                 │
│ 3. 避免闪烁，平滑过渡                                                       │
│ 4. Promise resolve 后自动重新渲染                                           │
│                                                                             │
│ 涉及源码:                                                                   │
│ - ReactFiberBeginWork.new.js: updateSuspenseComponent                      │
│ - ReactFiberThrow.new.js: throwException, attachPingListener               │
│ - ReactFiberWorkLoop.new.js: pingSuspendedRoot                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

export {
  caseA_Description,
  caseA_Code,
  caseA_Timeline,
  caseA_CallStack,
  caseB_Description,
  caseB_Code,
  caseB_Comparison,
  caseB_FiberTree,
  caseB_CallStack,
  caseSummary,
};

