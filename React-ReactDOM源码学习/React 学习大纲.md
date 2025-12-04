# React 18 源码深度学习指南

> 面向面试 + 实际开发的 React 源码学习路径
> 目标：不只是"了解"，而是"真正理解"React 的设计思想

---

## 🎯 学习目标

学完本指南，你将能够：

1. **面试应对**：回答 95%+ 的 React 原理面试题
2. **深入理解**：真正理解 React 的设计思想和实现原理
3. **实际开发**：写出更高质量的 React 代码，快速定位问题
4. **性能优化**：基于原理进行有针对性的性能优化
5. **架构能力**：学习顶级开源项目的工程化实践

---

## Part 1: 工程化架构设计

### 1.1 Monorepo 架构设计

#### 为什么选择 Monorepo？

```
📁 源码位置: package.json → "workspaces": ["packages/*"]

┌─────────────────────────────────────────────────────────────────┐
│                    Monorepo vs Multirepo 对比                   │
│                                                                 │
│  Monorepo（React 选择）          Multirepo                      │
│  ──────────────────────          ──────────                     │
│  ✅ 代码共享方便                  ❌ 需要发布再引用              │
│  ✅ 统一的构建流程                ❌ 每个仓库独立配置            │
│  ✅ 原子化提交（跨包修改）         ❌ 需要分别提交               │
│  ✅ 依赖关系清晰                  ❌ 依赖版本难以管理            │
│  ✅ 统一版本发布                  ❌ 版本不同步                  │
│                                                                 │
│  适用场景：                                                      │
│  - 包之间有紧密依赖（react-dom 依赖 react-reconciler）          │
│  - 需要频繁跨包修改                                              │
│  - 统一的发布节奏                                                │
└─────────────────────────────────────────────────────────────────┘
```

#### Yarn Workspaces 配置

```json
// 📁 package.json
{
  "private": true,
  "workspaces": ["packages/*"],  // 所有包都在 packages/ 下
  "scripts": {
    "build": "node ./scripts/rollup/build.js",
    "test": "node ./scripts/jest/jest-cli.js"
  }
}
```

**学习要点**：
- 理解为什么 React 项目需要 40+ 个包
- 理解包之间的依赖关系管理
- 理解统一版本发布的好处

---

### 1.2 构建系统深度解析

#### 为什么选择 Rollup 而不是 Webpack？

```
📁 源码位置: scripts/rollup/

┌─────────────────────────────────────────────────────────────────┐
│                    Rollup vs Webpack 对比                       │
│                                                                 │
│  场景        │ Rollup（React 选择）    │ Webpack                │
│  ───────────│─────────────────────────│──────────────────────   │
│  适用场景    │ 库/框架打包              │ 应用打包               │
│  Tree-shaking│ 原生支持，更彻底         │ 需配置，有限制         │
│  输出格式    │ ESM/CJS/UMD/IIFE        │ 主要 CJS              │
│  代码体积    │ 更小（无运行时）         │ 较大（有模块运行时）   │
│  构建速度    │ 快                      │ 较慢                   │
│  代码分割    │ 有限                    │ 强大                   │
│                                                                 │
│  💡 结论：库打包选 Rollup，应用打包选 Webpack/Vite              │
└─────────────────────────────────────────────────────────────────┘
```

#### 构建脚本结构

```
📁 scripts/rollup/
├── build.js              # 构建入口脚本
├── bundles.js            # ⭐ 包配置（定义每个包的构建选项）
├── modules.js            # 模块映射关系
├── packaging.js          # 打包后处理
├── forks.js              # 分支文件映射
├── plugins/              # Rollup 插件
│   ├── closure-plugin.js # Google Closure Compiler 压缩
│   ├── sizes-plugin.js   # 体积统计
│   └── use-forks-plugin.js # 条件编译
├── shims/                # 平台垫片
│   ├── facebook-www/     # Facebook 内部版本
│   └── react-native/     # React Native 版本
└── validate/             # 产物验证
```

#### bundles.js 核心配置解析

```javascript
// 📁 scripts/rollup/bundles.js

// 构建类型（不同环境/平台）
const bundleTypes = {
  NODE_ES2015: 'NODE_ES2015',    // Node.js ES2015
  NODE_ESM: 'NODE_ESM',          // ES Modules
  UMD_DEV: 'UMD_DEV',            // UMD 开发版
  UMD_PROD: 'UMD_PROD',          // UMD 生产版（压缩）
  UMD_PROFILING: 'UMD_PROFILING', // 性能分析版
  NODE_DEV: 'NODE_DEV',          // Node 开发版
  NODE_PROD: 'NODE_PROD',        // Node 生产版
  FB_WWW_DEV: 'FB_WWW_DEV',      // Facebook 内部开发
  FB_WWW_PROD: 'FB_WWW_PROD',    // Facebook 内部生产
  RN_OSS_DEV: 'RN_OSS_DEV',      // React Native 开源开发
  RN_OSS_PROD: 'RN_OSS_PROD',    // React Native 开源生产
};

// 模块类型
const moduleTypes = {
  ISOMORPHIC: 'ISOMORPHIC',      // 同构（如 react）
  RENDERER: 'RENDERER',          // 渲染器（如 react-dom）
  RENDERER_UTILS: 'RENDERER_UTILS', // 渲染器工具
  RECONCILER: 'RECONCILER',      // 协调器
};

// 示例：react 包的构建配置
{
  bundleTypes: [UMD_DEV, UMD_PROD, NODE_DEV, NODE_PROD, FB_WWW_DEV, ...],
  moduleType: ISOMORPHIC,
  entry: 'react',                // 入口
  global: 'React',               // UMD 全局变量名
  minifyWithProdErrorCodes: false,
  wrapWithModuleBoundaries: true,
  externals: ['ReactNativeInternalFeatureFlags'],
}
```

**学习要点**：
- React 需要为 10+ 种环境构建不同版本
- 理解 dev/prod/profiling 三种构建的区别
- 理解 Facebook 内部版本 vs 开源版本的差异

---

### 1.3 Babel 编译策略

#### Babel 配置

```javascript
// 📁 babel.config.js
module.exports = {
  plugins: [
    '@babel/plugin-syntax-jsx',                  // JSX 语法支持
    '@babel/plugin-transform-react-jsx',         // JSX 转换
    '@babel/plugin-transform-flow-strip-types',  // Flow 类型移除
    ['@babel/plugin-proposal-class-properties', {loose: true}],
    // ... 更多语法转换
  ],
};
```

#### 条件编译机制（核心！）

```javascript
// 📁 源码中随处可见的条件编译

// 1. __DEV__ 开发模式判断
if (__DEV__) {
  // 开发模式下的警告、验证
  console.warn('This is a development-only warning');
}
// 构建时：
// - DEV 构建：__DEV__ = true，代码保留
// - PROD 构建：__DEV__ = false，Dead Code Elimination 移除

// 2. __PROFILE__ 性能分析判断
if (__PROFILE__) {
  // 性能分析代码
  recordCommitTime();
}

// 3. __EXPERIMENTAL__ 实验特性
if (__EXPERIMENTAL__) {
  // 实验性 API
}
```

**构建流程**：
```
源码 → Babel 转换 → Rollup 打包 → Closure Compiler 压缩
                                          ↓
                              Dead Code Elimination
                              （移除 if(false) {...}）
```

---

### 1.4 构建优化策略

#### 1. Google Closure Compiler

```javascript
// 📁 scripts/rollup/plugins/closure-plugin.js

// 为什么用 Closure Compiler 而不是 Terser？
// - 更激进的压缩（属性重命名）
// - 更好的 Dead Code Elimination
// - 更小的产物体积

// 压缩效果对比（示例）
// Terser:    react.production.min.js  ~10KB
// Closure:   react.production.min.js  ~6KB
```

#### 2. 错误码压缩

```javascript
// 📁 scripts/error-codes/

// 开发版本（便于调试）
throw new Error('Invalid hook call. Hooks can only be called inside...');

// 生产版本（体积更小）
throw new Error(formatProdErrorMessage(321));
// 错误码 321 可在 https://reactjs.org/docs/error-decoder.html?invariant=321 查询

// 源码位置：scripts/error-codes/codes.json
{
  "321": "Invalid hook call. Hooks can only be called inside..."
}
```

#### 3. 产物分析

```
📁 构建产物结构

build/
├── node_modules/
│   ├── react/
│   │   ├── index.js                    # 入口
│   │   ├── cjs/
│   │   │   ├── react.development.js    # 开发版（~100KB，含警告）
│   │   │   └── react.production.min.js # 生产版（~6KB，压缩）
│   │   └── umd/
│   │       ├── react.development.js    # UMD 开发版
│   │       └── react.production.min.js # UMD 生产版
│   └── react-dom/
│       ├── index.js
│       ├── client.js
│       └── cjs/
│           ├── react-dom.development.js    # ~1MB
│           └── react-dom.production.min.js # ~130KB
```

---

### 1.5 本地开发流程

#### 开发命令

```bash
# 安装依赖
yarn install

# 构建所有包
yarn build

# 构建特定包
yarn build react react-dom --type=NODE_DEV

# 运行测试
yarn test

# 运行特定测试
yarn test ReactHooks

# 类型检查（Flow）
yarn flow

# 代码格式化
yarn prettier

# Lint 检查
yarn lint
```

#### 使用 fixtures 调试

```bash
# fixtures/ 包含各种测试场景
cd fixtures/dom
yarn install
yarn start
# 打开 http://localhost:3000 调试
```

---

### 1.6 测试体系

```
📁 测试相关文件

scripts/jest/
├── jest-cli.js           # Jest 入口
├── config.base.js        # 基础配置
├── matchers/             # 自定义 matchers
└── preprocessor.js       # 预处理器

packages/
└── */__tests__/          # 每个包的测试目录

fixtures/                 # 集成测试场景
├── dom/                  # DOM 测试
├── concurrent/           # 并发模式测试
├── ssr/                  # SSR 测试
└── ...
```

---

## Part 2: 包设计（40+ 个包）

### 2.1 包架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         React 包架构                                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        应用层 API                                  │ │
│  │     react          react-dom         react-native-renderer        │ │
│  │   (Hooks/JSX)     (Web 渲染)          (Native 渲染)               │ │
│  └────────────────────────────┬──────────────────────────────────────┘ │
│                               │                                         │
│                               │ 调用                                    │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        协调层                                      │ │
│  │                   react-reconciler                                 │ │
│  │              (Fiber/Hooks 实现/Diff/更新队列)                       │ │
│  └────────────────────────────┬──────────────────────────────────────┘ │
│                               │                                         │
│                               │ 调用                                    │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        调度层                                      │ │
│  │                      scheduler                                     │ │
│  │              (时间切片/优先级/任务调度)                             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        工具/共享层                                 │ │
│  │   shared         react-is      use-sync-external-store            │ │
│  │  (共享代码)      (类型判断)        (外部状态同步)                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心包详解

#### react 包

```
📁 packages/react/

src/
├── React.js              # ⭐ 入口文件（所有导出）
├── ReactBaseClasses.js   # Component/PureComponent
├── ReactElement.js       # createElement
├── ReactHooks.js         # ⭐ Hooks API 定义（重要！）
├── ReactContext.js       # createContext
├── ReactLazy.js          # lazy
├── ReactMemo.js          # memo
├── ReactForwardRef.js    # forwardRef
├── ReactChildren.js      # Children 工具
├── ReactCurrentDispatcher.js  # ⭐ dispatcher 指向
├── ReactCurrentOwner.js  # 当前渲染的组件
└── jsx/
    └── ReactJSXElement.js # 新 JSX 运行时
```

**关键设计：API 定义与实现分离**

```javascript
// 📁 packages/react/src/ReactHooks.js

// react 包只定义 API 接口，实现在 react-reconciler
export function useState(initialState) {
  const dispatcher = resolveDispatcher();  // 获取当前 dispatcher
  return dispatcher.useState(initialState); // 调用实现
}

// dispatcher 在渲染时由 react-reconciler 设置
function resolveDispatcher() {
  const dispatcher = ReactCurrentDispatcher.current;
  // dispatcher 指向 react-reconciler/src/ReactFiberHooks.js
  return dispatcher;
}
```

**为什么这样设计？**
- 允许不同环境有不同实现
- react-dom 和 react-native 可以有不同的 Hooks 实现
- 支持 DEV/PROD 不同行为

---

#### react-reconciler 包（核心！）

```
📁 packages/react-reconciler/src/

核心文件：
├── ReactFiber.new.js              # ⭐ Fiber 节点创建
├── ReactFiberWorkLoop.new.js      # ⭐⭐⭐ 工作循环（最核心！）
├── ReactFiberBeginWork.new.js     # ⭐⭐ beginWork（递阶段）
├── ReactFiberCompleteWork.new.js  # ⭐⭐ completeWork（归阶段）
├── ReactFiberCommitWork.new.js    # ⭐⭐ Commit 阶段
├── ReactFiberHooks.new.js         # ⭐⭐⭐ Hooks 实现（核心！）
├── ReactChildFiber.new.js         # ⭐⭐ Diff 算法
├── ReactFiberLane.new.js          # ⭐ Lane 优先级模型
├── ReactFiberRoot.new.js          # FiberRoot
├── ReactFiberReconciler.new.js    # 协调器入口

辅助文件：
├── ReactFiberFlags.js             # 副作用标记
├── ReactWorkTags.js               # Fiber 类型标签
├── ReactTypeOfMode.js             # 渲染模式
├── ReactHookEffectTags.js         # Effect 标签

注意：.new.js 和 .old.js
├── ReactFiberWorkLoop.new.js      # 新架构
└── ReactFiberWorkLoop.old.js      # 旧架构（通过 feature flag 切换）
```

**核心流程**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    React 渲染核心流程                           │
│                                                                 │
│   setState() / 初次渲染                                         │
│         │                                                       │
│         ▼                                                       │
│   scheduleUpdateOnFiber()                                       │
│         │                                                       │
│         ▼                                                       │
│   ensureRootIsScheduled()                                       │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Render 阶段（可中断）                         │  │
│   │                                                         │  │
│   │   performSyncWorkOnRoot() / performConcurrentWorkOnRoot()│  │
│   │         │                                               │  │
│   │         ▼                                               │  │
│   │   renderRootSync() / renderRootConcurrent()             │  │
│   │         │                                               │  │
│   │         ▼                                               │  │
│   │   workLoopSync() / workLoopConcurrent()                 │  │
│   │         │                                               │  │
│   │    ┌────┴────┐                                          │  │
│   │    │         │                                          │  │
│   │    ▼         ▼                                          │  │
│   │ beginWork() → completeWork()                            │  │
│   │   (递)         (归)                                      │  │
│   └─────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Commit 阶段（不可中断）                       │  │
│   │                                                         │  │
│   │   commitRoot()                                          │  │
│   │         │                                               │  │
│   │    ┌────┼────┬─────────┐                                │  │
│   │    │    │    │         │                                │  │
│   │    ▼    ▼    ▼         ▼                                │  │
│   │  Before  Mutation   Layout    异步调度                   │  │
│   │ Mutation  (DOM)     (DOM后)   useEffect                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

#### react-dom 包

```
📁 packages/react-dom/src/

client/
├── ReactDOM.js           # 客户端入口
├── ReactDOMRoot.js       # createRoot 实现
└── ReactDOMHostConfig.js # ⭐ HostConfig 实现（DOM 操作）

server/
├── ReactDOMServer.js     # 服务端入口
└── ReactDOMFizzServer.js # 流式 SSR

events/
├── DOMPluginEventSystem.js  # ⭐ 事件系统入口
├── SyntheticEvent.js        # 合成事件
├── getEventPriority.js      # 事件优先级
└── plugins/                 # 各种事件插件
```

**HostConfig 实现**：

```javascript
// 📁 packages/react-dom/src/client/ReactDOMHostConfig.js

// react-dom 通过实现 HostConfig 接口接入 react-reconciler
export function createInstance(type, props) {
  const element = document.createElement(type);
  // 设置属性...
  return element;
}

export function appendChild(parentInstance, child) {
  parentInstance.appendChild(child);
}

export function commitUpdate(domElement, updatePayload) {
  // 更新 DOM 属性
}

// ... 更多 DOM 操作
```

---

#### scheduler 包

```
📁 packages/scheduler/src/

├── Scheduler.js                # 调度器入口
├── SchedulerMinHeap.js         # 小顶堆（任务优先级队列）
├── SchedulerPriorities.js      # 优先级定义
└── forks/
    ├── Scheduler.js            # 通用实现
    └── SchedulerPostTask.js    # postTask API（浏览器标准）
```

**为什么独立成包？**
- 可以被非 React 项目使用
- 方便独立测试和优化
- 未来可能成为浏览器标准（scheduler.postTask）

---

#### shared 包

```
📁 packages/shared/

├── ReactSymbols.js        # ⭐ Symbol 定义（REACT_ELEMENT_TYPE 等）
├── ReactTypes.js          # 类型定义
├── ReactFeatureFlags.js   # ⭐ 特性开关（重要！）
├── ReactSharedInternals.js # 共享内部对象
├── objectIs.js            # Object.is polyfill
├── shallowEqual.js        # 浅比较
├── checkPropTypes.js      # PropTypes 检查
└── isValidElementType.js  # 元素类型验证
```

**ReactFeatureFlags 特性开关**：

```javascript
// 📁 packages/shared/ReactFeatureFlags.js

// 控制特性的开关，不同构建有不同配置
export const enableCache = __EXPERIMENTAL__;
export const enableTransitionTracing = false;
export const enableLazyContextPropagation = false;
export const enableSyncDefaultUpdates = true;

// 用于：
// 1. 灰度发布新特性
// 2. 为不同环境提供不同功能
// 3. A/B 测试
```

---

### 2.3 所有包一览表

| 分类 | 包名 | 说明 | 重要程度 |
|------|------|------|---------|
| **核心** | react | React API（Hooks、JSX、Component） | ⭐⭐⭐⭐ |
| **核心** | react-reconciler | 协调器（Fiber、Diff、更新队列） | ⭐⭐⭐⭐⭐ |
| **核心** | scheduler | 调度器（时间切片、优先级） | ⭐⭐⭐ |
| **核心** | shared | 共享代码（Symbol、工具函数） | ⭐⭐ |
| **渲染器** | react-dom | Web DOM 渲染 | ⭐⭐⭐⭐ |
| **渲染器** | react-native-renderer | React Native 渲染 | ⭐⭐ |
| **渲染器** | react-art | Canvas/SVG 渲染 | ⭐ |
| **渲染器** | react-test-renderer | 测试渲染器 | ⭐⭐ |
| **渲染器** | react-noop-renderer | 空渲染器（测试用） | ⭐ |
| **工具** | react-is | 类型判断 | ⭐⭐ |
| **工具** | use-sync-external-store | 外部状态同步 | ⭐⭐ |
| **工具** | use-subscription | 订阅管理 | ⭐ |
| **开发** | eslint-plugin-react-hooks | Hooks 规则检查 | ⭐⭐⭐ |
| **开发** | react-refresh | 快速刷新（HMR） | ⭐⭐ |
| **开发** | react-devtools* | DevTools 系列 | ⭐⭐ |
| **服务端** | react-server | Server Components 核心 | ⭐⭐ |
| **服务端** | react-server-dom-webpack | Webpack 集成 | ⭐⭐ |
| **服务端** | react-client | 客户端 RSC 消费 | ⭐⭐ |
| **实验性** | react-cache | 缓存（实验） | ⭐ |
| **实验性** | react-fetch | 数据获取（实验） | ⭐ |

---

## Part 3: 核心包设计深入

### 3.1 react 包设计哲学

#### 设计原则：API 定义与实现分离

```javascript
// 📁 packages/react/src/ReactHooks.js

import ReactCurrentDispatcher from './ReactCurrentDispatcher';

// react 包只定义 API，不包含实现
export function useState(initialState) {
  const dispatcher = resolveDispatcher();
  return dispatcher.useState(initialState);
}

export function useEffect(create, deps) {
  const dispatcher = resolveDispatcher();
  return dispatcher.useEffect(create, deps);
}

// dispatcher 是一个动态指针
function resolveDispatcher() {
  const dispatcher = ReactCurrentDispatcher.current;
  // 在渲染时，react-reconciler 会设置这个指针
  // 指向 ReactFiberHooks.js 中的实现
  return dispatcher;
}
```

**为什么这样设计？**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dispatcher 模式                              │
│                                                                 │
│   react 包（API 定义）                                          │
│        │                                                        │
│        │ ReactCurrentDispatcher.current                         │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Dispatcher 接口                             │  │
│   │  useState | useEffect | useContext | ...                │  │
│   └────────────────────────┬────────────────────────────────┘  │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│   HooksDispatcher    InvalidNestedHooks   ContextOnlyDispatcher│
│   (正常渲染)          (嵌套调用警告)        (服务端渲染)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

优势：
1. react 包可以保持稳定，实现可以独立演进
2. 不同环境（DOM/Native/SSR）可以有不同实现
3. DEV 模式可以注入额外检查
```

---

### 3.2 react-reconciler 核心设计

#### Fiber 数据结构

```javascript
// 📁 packages/react-reconciler/src/ReactFiber.new.js

function FiberNode(tag, pendingProps, key, mode) {
  // 实例相关
  this.tag = tag;                    // Fiber 类型
  this.key = key;                    // key
  this.elementType = null;           // 元素类型
  this.type = null;                  // 组件类型
  this.stateNode = null;             // DOM 节点/组件实例

  // Fiber 树结构
  this.return = null;                // 父节点
  this.child = null;                 // 第一个子节点
  this.sibling = null;               // 兄弟节点
  this.index = 0;                    // 索引

  this.ref = null;                   // ref

  // 状态相关
  this.pendingProps = pendingProps;  // 新 props
  this.memoizedProps = null;         // 旧 props
  this.updateQueue = null;           // 更新队列
  this.memoizedState = null;         // ⭐ Hooks 链表！
  this.dependencies = null;          // Context 依赖

  this.mode = mode;                  // 渲染模式

  // Effects
  this.flags = NoFlags;              // 副作用标记
  this.subtreeFlags = NoFlags;       // 子树副作用
  this.deletions = null;             // 要删除的子节点

  // 调度相关
  this.lanes = NoLanes;              // 优先级
  this.childLanes = NoLanes;         // 子树优先级

  // 双缓冲
  this.alternate = null;             // 另一棵树的对应节点
}
```

#### Hooks 实现机制

```javascript
// 📁 packages/react-reconciler/src/ReactFiberHooks.new.js

// Hooks 存储在 Fiber.memoizedState 上，是一个链表
// Hook 数据结构
type Hook = {
  memoizedState: any,       // 存储的状态
  baseState: any,           // 基础状态
  baseQueue: Update | null, // 基础更新队列
  queue: UpdateQueue | null,// 更新队列
  next: Hook | null,        // 下一个 Hook
};

// mount 阶段的 useState
function mountState(initialState) {
  // 1. 创建 Hook 节点
  const hook = mountWorkInProgressHook();
  
  // 2. 初始化状态
  if (typeof initialState === 'function') {
    initialState = initialState();
  }
  hook.memoizedState = hook.baseState = initialState;
  
  // 3. 创建更新队列
  const queue = {
    pending: null,
    lanes: NoLanes,
    dispatch: null,
    lastRenderedReducer: basicStateReducer,
    lastRenderedState: initialState,
  };
  hook.queue = queue;
  
  // 4. 绑定 dispatch
  const dispatch = dispatchSetState.bind(null, currentlyRenderingFiber, queue);
  queue.dispatch = dispatch;
  
  return [hook.memoizedState, dispatch];
}

// mountWorkInProgressHook: 创建并链接 Hook
function mountWorkInProgressHook() {
  const hook = {
    memoizedState: null,
    baseState: null,
    baseQueue: null,
    queue: null,
    next: null,
  };

  if (workInProgressHook === null) {
    // 第一个 Hook
    currentlyRenderingFiber.memoizedState = workInProgressHook = hook;
  } else {
    // 添加到链表尾部
    workInProgressHook = workInProgressHook.next = hook;
  }
  
  return workInProgressHook;
}
```

**为什么 Hooks 不能放在条件语句中？**

```
第一次渲染：
  Hook1 → Hook2 → Hook3
    ↑
  按顺序创建

第二次渲染（正确）：
  Hook1 → Hook2 → Hook3
    ↑       ↑       ↑
  按顺序匹配

第二次渲染（错误，条件语句跳过了 Hook2）：
  Hook1 → Hook3
    ↑       ↑
  按顺序匹配，但 Hook3 取到了 Hook2 的状态！
```

---

### 3.3 渲染器接入机制

#### HostConfig 接口

```javascript
// react-reconciler 定义接口，渲染器实现

// 📁 react-dom 的实现
export function createInstance(type, props, rootContainerInstance) {
  const element = document.createElement(type);
  // 设置属性
  return element;
}

export function appendChild(parentInstance, child) {
  parentInstance.appendChild(child);
}

export function commitUpdate(domElement, updatePayload, type, oldProps, newProps) {
  // 更新 DOM
}

// 📁 react-native 的实现（完全不同）
export function createInstance(type, props) {
  return UIManager.createView(type, props);
}

// 📁 自定义渲染器示例（渲染到 Canvas）
export function createInstance(type, props) {
  return new CanvasElement(type, props);
}
```

---

## Part 4: 学习路径与方法

### 4.1 源码阅读顺序

```
第一阶段：基础概念（2-3 天）
├── 1. shared/ReactSymbols.js         # 了解各种类型标识
├── 2. shared/ReactFeatureFlags.js    # 了解特性开关
├── 3. react/src/ReactElement.js      # 理解元素结构
└── 4. react/src/ReactHooks.js        # 理解 Hooks API

第二阶段：核心机制（1-2 周）
├── 5. react-reconciler/src/ReactFiber.new.js         # Fiber 结构
├── 6. react-reconciler/src/ReactFiberWorkLoop.new.js # ⭐ 工作循环
├── 7. react-reconciler/src/ReactFiberBeginWork.new.js # beginWork
├── 8. react-reconciler/src/ReactFiberHooks.new.js    # ⭐ Hooks 实现
└── 9. react-reconciler/src/ReactChildFiber.new.js    # Diff 算法

第三阶段：进阶内容（1 周）
├── 10. react-reconciler/src/ReactFiberLane.new.js    # Lane 优先级
├── 11. scheduler/src/Scheduler.js                    # 调度器
├── 12. react-dom/src/events/                         # 事件系统
└── 13. react-dom/src/client/ReactDOMHostConfig.js    # DOM 操作
```

### 4.2 调试技巧

```javascript
// 1. 添加 console.log
// 在 ReactFiberWorkLoop.new.js
function performUnitOfWork(unitOfWork) {
  console.log('Processing:', unitOfWork.type); // 添加日志
  // ...
}

// 2. 使用 debugger
function beginWork(current, workInProgress, renderLanes) {
  debugger; // 断点
  // ...
}

// 3. 使用 fixtures
cd fixtures/dom
yarn start
// 打开 DevTools 调试
```

### 4.3 学习检查清单

- [ ] 能画出 React 包的架构图
- [ ] 能解释 react 和 react-reconciler 的分离设计
- [ ] 能说出 Fiber 节点的关键属性
- [ ] 能解释 Hooks 为什么不能条件调用
- [ ] 能说出 Render/Commit 两阶段的工作
- [ ] 能解释 React 构建产物的差异
- [ ] 能解释条件编译（__DEV__）的作用

---

## 🔗 参考资源

- [React 技术揭秘](https://react.iamkasong.com/)（卡颂）
- [React 官方博客](https://react.dev/blog)
- [React 源码中的注释]（源码本身注释很详细）
- [Building a Custom React Renderer](https://github.com/nitin42/Making-a-custom-React-renderer)（自定义渲染器教程）
