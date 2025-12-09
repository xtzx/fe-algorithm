/**
 * ============================================================
 * 📚 Phase 0: React 架构设计深度解析
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 React 项目的工程化架构设计
 * 2. 掌握 Monorepo 多包开发模式
 * 3. 理解构建系统和优化策略
 * 4. 掌握核心包之间的依赖关系
 *
 * 📁 核心源码位置：
 * - package.json                     # 根配置
 * - scripts/rollup/build.js          # 构建入口
 * - scripts/rollup/bundles.js        # 包配置
 * - babel.config.js                  # Babel 配置
 * - packages/                        # 所有包
 *
 * ⏱️ 预计时间：4-6 小时
 */

// ============================================================
// Part 1: 工程化架构设计
// ============================================================

/**
 * =====================================================
 * 1.1 Monorepo 架构深度解析
 * =====================================================
 *
 * 📁 源码位置: package.json
 *
 * ```json
 * {
 *   "private": true,
 *   "workspaces": ["packages/*"]
 * }
 * ```
 *
 * 📊 为什么 React 选择 Monorepo？
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    Monorepo vs Multirepo                        │
 * │                                                                 │
 * │  问题场景：react-dom 需要修改 react-reconciler 的一个 API        │
 * │                                                                 │
 * │  Multirepo 流程：                                               │
 * │  1. 在 react-reconciler 仓库修改                                │
 * │  2. 发布新版本 react-reconciler@1.0.1                           │
 * │  3. 在 react-dom 仓库更新依赖                                   │
 * │  4. 测试 → 发布 react-dom@18.0.1                                │
 * │  5. 如果发现问题，再重复整个流程...                              │
 * │                                                                 │
 * │  Monorepo 流程：                                                │
 * │  1. 在同一个 PR 中修改两个包                                     │
 * │  2. 统一测试                                                    │
 * │  3. 一次性发布                                                  │
 * │                                                                 │
 * │  ✅ 效率提升 5-10 倍！                                          │
 * └─────────────────────────────────────────────────────────────────┘
 */

/**
 * 📊 Yarn Workspaces 工作原理
 *
 * 目录结构：
 * ```
 * react/
 * ├── node_modules/
 * │   ├── react -> ../packages/react          # 符号链接
 * │   ├── react-dom -> ../packages/react-dom  # 符号链接
 * │   ├── scheduler -> ../packages/scheduler  # 符号链接
 * │   └── ... (外部依赖)
 * └── packages/
 *     ├── react/
 *     ├── react-dom/
 *     └── scheduler/
 * ```
 *
 * 优势：
 * 1. 包之间可以直接 import，无需发布
 * 2. 依赖统一提升到根目录，减少重复
 * 3. 统一的 node_modules，版本一致性
 */

// ============================================================
// 1.2 构建系统深度解析
// ============================================================

/**
 * 📁 源码位置: scripts/rollup/build.js
 *
 * 📊 构建流程
 *
 * ```
 * yarn build
 *     │
 *     ▼
 * scripts/rollup/build.js
 *     │
 *     ├── 1. 解析命令行参数
 *     │      yarn build react react-dom --type=NODE_DEV
 *     │
 *     ├── 2. 加载包配置
 *     │      const Bundles = require('./bundles');
 *     │
 *     ├── 3. 遍历构建配置
 *     │      for (const bundle of Bundles.bundles) { ... }
 *     │
 *     ├── 4. Rollup 打包
 *     │      const result = await rollup.rollup(inputOptions);
 *     │
 *     ├── 5. Closure Compiler 压缩（生产版本）
 *     │
 *     └── 6. 输出到 build/ 目录
 * ```
 */

/**
 * 📊 Rollup 插件链
 *
 * 📁 源码位置: scripts/rollup/build.js (第 96-250 行)
 */

const rollupPluginChain = `
Rollup 插件执行顺序：

1. rollup-plugin-node-resolve
   │  解析 node_modules 中的模块
   ▼
2. rollup-plugin-babel
   │  Babel 转译（JSX、Flow 类型移除）
   ▼
3. rollup-plugin-commonjs
   │  转换 CommonJS 为 ES Modules
   ▼
4. rollup-plugin-replace
   │  替换环境变量（__DEV__、__PROFILE__）
   ▼
5. use-forks-plugin（自定义）
   │  条件编译，选择不同实现
   │  例：ReactFiberHooks.new.js vs ReactFiberHooks.old.js
   ▼
6. strip-unused-imports（自定义）
   │  移除未使用的 import
   ▼
7. rollup-plugin-prettier
   │  代码格式化（开发版本）
   ▼
8. closure-plugin（自定义）
   │  Google Closure Compiler 压缩（生产版本）
   ▼
9. sizes-plugin（自定义）
   │  输出体积统计
   ▼
输出文件
`;

/**
 * 📊 bundles.js 核心配置解析
 *
 * 📁 源码位置: scripts/rollup/bundles.js
 */

// 构建类型定义
const bundleTypes = {
  // Node.js 环境
  NODE_ES2015: 'NODE_ES2015',     // ES2015 语法（现代 Node）
  NODE_ESM: 'NODE_ESM',           // ES Modules 格式
  NODE_DEV: 'NODE_DEV',           // 开发版本（含警告）
  NODE_PROD: 'NODE_PROD',         // 生产版本（压缩）
  NODE_PROFILING: 'NODE_PROFILING', // 性能分析版

  // 浏览器 UMD
  UMD_DEV: 'UMD_DEV',             // UMD 开发版
  UMD_PROD: 'UMD_PROD',           // UMD 生产版
  UMD_PROFILING: 'UMD_PROFILING', // UMD 性能分析版

  // Facebook 内部
  FB_WWW_DEV: 'FB_WWW_DEV',       // Facebook 网站开发版
  FB_WWW_PROD: 'FB_WWW_PROD',     // Facebook 网站生产版
  FB_WWW_PROFILING: 'FB_WWW_PROFILING',

  // React Native
  RN_OSS_DEV: 'RN_OSS_DEV',       // RN 开源开发版
  RN_OSS_PROD: 'RN_OSS_PROD',     // RN 开源生产版
  RN_FB_DEV: 'RN_FB_DEV',         // RN Facebook 内部开发版
  RN_FB_PROD: 'RN_FB_PROD',       // RN Facebook 内部生产版
};

// 模块类型定义
const moduleTypes = {
  ISOMORPHIC: 'ISOMORPHIC',       // 同构代码（如 react）
  RENDERER: 'RENDERER',           // 渲染器（如 react-dom）
  RENDERER_UTILS: 'RENDERER_UTILS', // 渲染器工具
  RECONCILER: 'RECONCILER',       // 协调器
};

// react 包的构建配置示例
const reactBundleConfig = {
  bundleTypes: [
    'UMD_DEV',
    'UMD_PROD',
    'UMD_PROFILING',
    'NODE_DEV',
    'NODE_PROD',
    'FB_WWW_DEV',
    'FB_WWW_PROD',
    'FB_WWW_PROFILING',
    'RN_FB_DEV',
    'RN_FB_PROD',
    'RN_FB_PROFILING',
  ],
  moduleType: 'ISOMORPHIC',
  entry: 'react',
  global: 'React',              // UMD 全局变量名
  minifyWithProdErrorCodes: false,
  wrapWithModuleBoundaries: true,
  externals: ['ReactNativeInternalFeatureFlags'],
};

// ============================================================
// 1.3 条件编译机制（核心！）
// ============================================================

/**
 * 📁 源码中随处可见的条件编译
 *
 * 📊 条件编译变量
 *
 * 1. __DEV__
 *    - 开发模式标志
 *    - 构建时替换为 true/false
 *    - 用于：警告信息、参数校验、调试日志
 *
 * 2. __PROFILE__
 *    - 性能分析模式
 *    - 用于：Profiler 组件、性能指标收集
 *
 * 3. __EXPERIMENTAL__
 *    - 实验特性标志
 *    - 用于：新 API、未稳定功能
 */

// 条件编译示例
const conditionalCompilationExample = `
// 📁 源码示例（到处都有）

// 1. 开发模式警告
if (__DEV__) {
  console.warn(
    'Warning: Invalid prop \`%s\` supplied to \`%s\`.',
    propName,
    componentName
  );
}

// 2. 开发模式参数校验
function createElement(type, props, children) {
  if (__DEV__) {
    // 校验 type 是否合法
    if (type === undefined || type === null) {
      console.error('createElement: type is invalid');
    }
    // 校验 key 是否使用正确
    if (props && props.key !== undefined) {
      checkKeyStringCoercion(props.key);
    }
  }
  // 实际创建逻辑...
}

// 3. 性能分析代码
if (__PROFILE__) {
  recordCommitTime();
  recordLayoutEffectDuration(finishedWork);
}

// 4. 实验特性
if (__EXPERIMENTAL__) {
  // Server Components 相关代码
  exports.experimental_use = use;
}
`;

/**
 * 📊 构建时替换过程
 *
 * 📁 源码位置: scripts/rollup/build.js (rollup-plugin-replace)
 *
 * ```javascript
 * // 开发构建
 * replace({
 *   __DEV__: 'true',
 *   __PROFILE__: 'true',
 *   __EXPERIMENTAL__: 'true',
 * })
 *
 * // 生产构建
 * replace({
 *   __DEV__: 'false',           // 替换为 false
 *   __PROFILE__: 'false',
 *   __EXPERIMENTAL__: 'false',
 * })
 * ```
 *
 * 替换后：
 * ```javascript
 * if (false) {    // __DEV__ 被替换
 *   console.warn(...);
 * }
 * ```
 *
 * Closure Compiler Dead Code Elimination：
 * ```javascript
 * // 整个 if 块被移除！
 * ```
 */

// ============================================================
// 1.4 构建优化策略
// ============================================================

/**
 * 📊 优化策略 1: Google Closure Compiler
 *
 * 📁 源码位置: scripts/rollup/plugins/closure-plugin.js
 *
 * 为什么用 Closure Compiler 而不是 Terser？
 *
 * | 特性 | Closure Compiler | Terser |
 * |------|------------------|--------|
 * | 压缩率 | 更高（约 10-15%） | 标准 |
 * | 属性重命名 | 支持 | 不支持 |
 * | Dead Code | 更激进 | 保守 |
 * | 速度 | 较慢 | 快 |
 *
 * 配置：
 * ```javascript
 * const closureOptions = {
 *   compilation_level: 'SIMPLE',  // SIMPLE/ADVANCED
 *   language_in: 'ECMASCRIPT_2015',
 *   language_out: 'ECMASCRIPT5_STRICT',
 *   env: 'CUSTOM',
 *   warning_level: 'QUIET',
 * };
 * ```
 */

/**
 * 📊 优化策略 2: 错误码压缩
 *
 * 📁 源码位置: scripts/error-codes/
 *
 * ```
 * scripts/error-codes/
 * ├── codes.json           # 错误码映射
 * ├── extract-errors.js    # 提取错误信息
 * └── replace-invariant-error-codes.js  # 替换错误码
 * ```
 */

const errorCodeExample = `
// 📁 codes.json（部分）
{
  "1": "Invalid argument passed to %s",
  "130": "Element type is invalid: expected a string...",
  "321": "Invalid hook call. Hooks can only be called inside...",
  "423": "Rendered fewer hooks than expected..."
}

// 开发版本（便于调试）
throw new Error(
  'Invalid hook call. Hooks can only be called inside of the body ' +
  'of a function component. This could happen for one of the following reasons:\\n' +
  '1. You might have mismatching versions of React and the renderer...'
);

// 生产版本（体积优化）
throw new Error(formatProdErrorMessage(321));

// formatProdErrorMessage 返回：
// "Minified React error #321; visit https://reactjs.org/docs/error-decoder.html?invariant=321 for the full message"
`;

/**
 * 📊 优化策略 3: 分支文件（Forks）
 *
 * 📁 源码位置: scripts/rollup/forks.js
 *
 * React 使用"分支文件"为不同环境提供不同实现
 */

const forksExample = `
// 📁 packages/react-reconciler/src/

// 主文件（入口）
ReactFiberHooks.js

// 分支文件
ReactFiberHooks.new.js    // 新版实现（当前使用）
ReactFiberHooks.old.js    // 旧版实现（兼容）

// forks.js 配置
'react-reconciler/src/ReactFiberHooks': (bundleType) => {
  if (enableNewReconciler) {
    return 'react-reconciler/src/ReactFiberHooks.new.js';
  }
  return 'react-reconciler/src/ReactFiberHooks.old.js';
}

// 构建时根据 enableNewReconciler 选择使用哪个文件
`;

// ============================================================
// 1.5 产物分析
// ============================================================

/**
 * 📊 构建产物结构
 */

const buildOutputStructure = `
build/
├── node_modules/
│   ├── react/
│   │   ├── package.json
│   │   ├── index.js                      # 入口
│   │   ├── cjs/
│   │   │   ├── react.development.js      # CJS 开发版 (~100KB)
│   │   │   └── react.production.min.js   # CJS 生产版 (~6KB)
│   │   └── umd/
│   │       ├── react.development.js      # UMD 开发版
│   │       └── react.production.min.js   # UMD 生产版
│   │
│   ├── react-dom/
│   │   ├── package.json
│   │   ├── index.js
│   │   ├── client.js                     # createRoot 入口
│   │   ├── server.js                     # SSR 入口
│   │   └── cjs/
│   │       ├── react-dom.development.js      # ~1MB
│   │       └── react-dom.production.min.js   # ~130KB
│   │
│   └── scheduler/
│       ├── package.json
│       └── cjs/
│           ├── scheduler.development.js
│           └── scheduler.production.min.js

// 入口文件示例 (react/index.js)
'use strict';

if (process.env.NODE_ENV === 'production') {
  module.exports = require('./cjs/react.production.min.js');
} else {
  module.exports = require('./cjs/react.development.js');
}
`;

// ============================================================
// Part 2: 包设计
// ============================================================

/**
 * =====================================================
 * 2.1 包架构全景图
 * =====================================================
 */

const packageArchitecture = `
┌─────────────────────────────────────────────────────────────────────────┐
│                         React 包架构                                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        用户 API 层                                 │ │
│  │                                                                   │ │
│  │   react              react-dom            react-native-renderer   │ │
│  │   • createElement    • createRoot         • 原生组件渲染          │ │
│  │   • useState         • hydrate            • 桥接通信              │ │
│  │   • useEffect        • 事件系统                                   │ │
│  │   • Component        • DOM 操作                                   │ │
│  │                                                                   │ │
│  └────────────────────────────────┬──────────────────────────────────┘ │
│                                   │                                     │
│                                   │ 依赖                                │
│                                   ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        协调层                                      │ │
│  │                                                                   │ │
│  │                    react-reconciler                               │ │
│  │                                                                   │ │
│  │   • Fiber 架构（数据结构、双缓冲）                                 │ │
│  │   • Hooks 实现（useState、useEffect 的真正逻辑）                   │ │
│  │   • Diff 算法（reconcileChildFibers）                             │ │
│  │   • 更新队列（Update、UpdateQueue）                               │ │
│  │   • 工作循环（workLoop、beginWork、completeWork）                  │ │
│  │   • Commit 阶段（DOM 操作的调度）                                  │ │
│  │                                                                   │ │
│  └────────────────────────────────┬──────────────────────────────────┘ │
│                                   │                                     │
│                                   │ 依赖                                │
│                                   ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        调度层                                      │ │
│  │                                                                   │ │
│  │                      scheduler                                    │ │
│  │                                                                   │ │
│  │   • 任务优先级（5 个级别）                                         │ │
│  │   • 时间切片（默认 5ms）                                           │ │
│  │   • 任务队列（小顶堆）                                             │ │
│  │   • MessageChannel（调度实现）                                     │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        工具/共享层                                 │ │
│  │                                                                   │ │
│  │   shared            react-is          use-sync-external-store     │ │
│  │   • ReactSymbols    • isElement       • useSyncExternalStore      │ │
│  │   • ReactTypes      • isValidType     • 外部状态同步              │ │
│  │   • FeatureFlags    • 类型判断                                    │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
`;

/**
 * =====================================================
 * 2.2 核心包详解
 * =====================================================
 */

/**
 * 📊 react 包
 *
 * 📁 源码位置: packages/react/src/
 */

const reactPackageStructure = `
packages/react/src/
├── React.js                  # ⭐ 入口文件，所有导出
├── ReactBaseClasses.js       # Component、PureComponent
├── ReactElement.js           # createElement、isValidElement
├── ReactHooks.js             # ⭐ Hooks API 定义（非实现！）
├── ReactContext.js           # createContext
├── ReactLazy.js              # lazy
├── ReactMemo.js              # memo
├── ReactForwardRef.js        # forwardRef
├── ReactChildren.js          # Children.map/forEach/count
├── ReactCurrentDispatcher.js # ⭐ dispatcher 指针
├── ReactCurrentOwner.js      # 当前渲染的 Fiber
├── ReactSharedInternals.js   # 共享内部对象
├── ReactStartTransition.js   # startTransition
├── ReactAct.js               # act（测试用）
└── jsx/
    └── ReactJSXElement.js    # 新 JSX 运行时 (jsx, jsxs)
`;

/**
 * 📊 react-reconciler 包
 *
 * 📁 源码位置: packages/react-reconciler/src/
 */

const reconcilerPackageStructure = `
packages/react-reconciler/src/
│
├── ⭐ 核心文件
│   ├── ReactFiber.new.js              # Fiber 节点创建
│   ├── ReactFiberWorkLoop.new.js      # ⭐⭐⭐ 工作循环（最核心！）
│   ├── ReactFiberBeginWork.new.js     # ⭐⭐ beginWork（递阶段）
│   ├── ReactFiberCompleteWork.new.js  # ⭐⭐ completeWork（归阶段）
│   ├── ReactFiberCommitWork.new.js    # ⭐⭐ Commit 阶段
│   ├── ReactFiberHooks.new.js         # ⭐⭐⭐ Hooks 实现
│   └── ReactChildFiber.new.js         # ⭐⭐ Diff 算法
│
├── 数据结构
│   ├── ReactFiberRoot.new.js          # FiberRoot
│   ├── ReactFiberLane.new.js          # Lane 优先级
│   ├── ReactFiberFlags.js             # 副作用标记
│   └── ReactWorkTags.js               # Fiber 类型
│
├── 更新机制
│   ├── ReactFiberClassUpdateQueue.new.js  # 类组件更新队列
│   ├── ReactFiberConcurrentUpdates.new.js # 并发更新
│   └── ReactFiberSyncTaskQueue.new.js     # 同步任务队列
│
├── Context 相关
│   ├── ReactFiberContext.new.js       # Legacy Context
│   └── ReactFiberNewContext.new.js    # New Context API
│
├── Suspense 相关
│   ├── ReactFiberSuspenseComponent.new.js
│   ├── ReactFiberSuspenseContext.new.js
│   └── ReactFiberThrow.new.js         # 错误边界
│
└── 其他
    ├── ReactFiberReconciler.new.js    # 协调器入口
    ├── ReactFiberHostConfig.js        # ⭐ 宿主配置接口
    └── ReactInternalTypes.js          # 类型定义
`;

/**
 * 📊 react-dom 包
 *
 * 📁 源码位置: packages/react-dom/src/
 */

const reactDomPackageStructure = `
packages/react-dom/src/
│
├── client/
│   ├── ReactDOM.js               # 客户端入口
│   ├── ReactDOMRoot.js           # createRoot 实现
│   └── ReactDOMHostConfig.js     # ⭐ HostConfig 实现
│
├── server/
│   ├── ReactDOMServer.js         # 服务端入口
│   ├── ReactDOMFizzServer.js     # 流式 SSR
│   └── ReactDOMServerFormatConfig.js
│
├── events/
│   ├── DOMPluginEventSystem.js   # ⭐ 事件系统入口
│   ├── ReactDOMEventListener.js  # 事件监听
│   ├── SyntheticEvent.js         # 合成事件
│   ├── getEventPriority.js       # 事件优先级
│   └── plugins/                  # 事件插件
│       ├── SimpleEventPlugin.js
│       ├── ChangeEventPlugin.js
│       └── ...
│
└── shared/
    ├── DOMProperty.js            # DOM 属性处理
    ├── CSSProperty.js            # CSS 属性处理
    └── sanitizeURL.js            # URL 安全处理
`;

/**
 * 📊 scheduler 包
 *
 * 📁 源码位置: packages/scheduler/src/
 */

const schedulerPackageStructure = `
packages/scheduler/src/
├── Scheduler.js              # 调度器入口
├── SchedulerMinHeap.js       # 小顶堆（优先级队列）
├── SchedulerPriorities.js    # 优先级定义
└── forks/
    ├── Scheduler.js          # 通用实现
    └── SchedulerPostTask.js  # postTask API 实现
`;

// ============================================================
// Part 3: 核心包设计深入
// ============================================================

/**
 * =====================================================
 * 3.1 react 包设计哲学：API 定义与实现分离
 * =====================================================
 *
 * 📁 源码位置: packages/react/src/ReactHooks.js
 */

const dispatcherPatternExample = `
// 📁 packages/react/src/ReactHooks.js

import ReactCurrentDispatcher from './ReactCurrentDispatcher';

// react 包只定义 API，不包含实现！
export function useState(initialState) {
  const dispatcher = resolveDispatcher();  // 获取当前 dispatcher
  return dispatcher.useState(initialState); // 调用实现
}

export function useEffect(create, deps) {
  const dispatcher = resolveDispatcher();
  return dispatcher.useEffect(create, deps);
}

// dispatcher 是一个动态指针
function resolveDispatcher() {
  const dispatcher = ReactCurrentDispatcher.current;
  if (__DEV__) {
    if (dispatcher === null) {
      console.error(
        'Invalid hook call. Hooks can only be called inside ' +
        'of the body of a function component...'
      );
    }
  }
  return dispatcher;
}

// 📁 packages/react/src/ReactCurrentDispatcher.js
const ReactCurrentDispatcher = {
  current: null,  // 在渲染时由 react-reconciler 设置
};
`;

/**
 * 📊 Dispatcher 模式的优势
 *
 * ```
 *                    react 包
 *                       │
 *   ReactCurrentDispatcher.current
 *                       │
 *                       ▼
 *              ┌─────────────────┐
 *              │   Dispatcher    │
 *              └────────┬────────┘
 *                       │
 *      ┌────────────────┼────────────────┐
 *      │                │                │
 *      ▼                ▼                ▼
 * Hooks Dispatcher  Invalid Hooks   Server Hooks
 * (正常渲染)        Dispatcher      Dispatcher
 *                  (错误提示)       (SSR)
 *
 * 在 react-reconciler 渲染时：
 * ReactCurrentDispatcher.current = HooksDispatcherOnMount;
 * // 或
 * ReactCurrentDispatcher.current = HooksDispatcherOnUpdate;
 * ```
 */

/**
 * =====================================================
 * 3.2 react-reconciler 核心：协调器入口
 * =====================================================
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberReconciler.new.js
 */

const reconcilerEntryExample = `
// 📁 ReactFiberReconciler.new.js（简化版）

// 1. createContainer - 创建 FiberRoot
export function createContainer(containerInfo, tag, hydrate) {
  return createFiberRoot(containerInfo, tag, hydrate);
}

// 2. updateContainer - 触发更新
export function updateContainer(element, container, parentComponent, callback) {
  const current = container.current;  // 获取 rootFiber
  const eventTime = requestEventTime();
  const lane = requestUpdateLane(current);

  // 创建更新对象
  const update = createUpdate(eventTime, lane);
  update.payload = { element };
  update.callback = callback;

  // 加入更新队列
  enqueueUpdate(current, update, lane);

  // 调度更新
  scheduleUpdateOnFiber(current, lane, eventTime);

  return lane;
}

// 3. batchedUpdates - 批量更新
export { batchedUpdates } from './ReactFiberWorkLoop.new';

// 4. flushSync - 同步刷新
export { flushSync } from './ReactFiberWorkLoop.new';
`;

/**
 * =====================================================
 * 3.3 HostConfig 接口：渲染器如何接入
 * =====================================================
 *
 * 📁 源码位置: packages/react-dom/src/client/ReactDOMHostConfig.js
 */

const hostConfigExample = `
// react-reconciler 定义了 HostConfig 接口
// 不同渲染器（react-dom、react-native）实现这些接口

// 📁 react-dom 的实现
export function createInstance(type, props, rootContainerInstance) {
  // 创建 DOM 元素
  const domElement = document.createElement(type);
  // 设置属性
  updateFiberProps(domElement, props);
  return domElement;
}

export function appendChild(parentInstance, child) {
  parentInstance.appendChild(child);
}

export function insertBefore(parentInstance, child, beforeChild) {
  parentInstance.insertBefore(child, beforeChild);
}

export function removeChild(parentInstance, child) {
  parentInstance.removeChild(child);
}

export function commitUpdate(
  domElement,
  updatePayload,
  type,
  oldProps,
  newProps
) {
  // 更新 DOM 属性
  updateProperties(domElement, updatePayload, type, oldProps, newProps);
}

export function commitTextUpdate(textInstance, oldText, newText) {
  textInstance.nodeValue = newText;
}

// react-native 的实现完全不同
// export function createInstance(type, props) {
//   return UIManager.createView(type, props);
// }
`;

// ============================================================
// Part 4: 面试题与实践
// ============================================================

/**
 * 💡 面试题
 */

const interviewQuestions = `
💡 Q1: React 为什么选择 Monorepo？
A:
   1. 代码共享方便（shared 包）
   2. 原子化提交（一次修改多个包）
   3. 统一的构建和测试流程
   4. 依赖版本一致性
   5. 方便跨包重构

💡 Q2: React 为什么用 Rollup 而不是 Webpack？
A:
   1. Rollup 适合库打包，Webpack 适合应用打包
   2. Rollup 原生支持 Tree-shaking
   3. Rollup 输出更小（无模块运行时）
   4. 支持多种输出格式（ESM/CJS/UMD）

💡 Q3: __DEV__ 是如何工作的？
A:
   1. 源码中使用 if (__DEV__) { ... }
   2. 构建时 rollup-plugin-replace 替换为 true/false
   3. 生产构建替换为 false 后
   4. Closure Compiler 的 Dead Code Elimination 移除整个 if 块

💡 Q4: react 和 react-reconciler 为什么分离？
A:
   1. react 定义 API，react-reconciler 实现逻辑
   2. Dispatcher 模式允许不同实现
   3. 支持多平台（DOM/Native/测试）
   4. DEV/PROD 可以有不同行为

💡 Q5: 如何实现自定义渲染器？
A:
   1. 安装 react-reconciler
   2. 实现 HostConfig 接口（createInstance、appendChild 等）
   3. 创建渲染器实例
   4. 示例：react-three-fiber、ink（终端渲染）

💡 Q6: React 的错误码压缩是怎么实现的？
A:
   1. 开发版本使用完整错误信息
   2. 生产版本替换为错误码（如 321）
   3. formatProdErrorMessage(321) 返回链接
   4. 用户可以在官网查询完整信息
`;

// ============================================================
// 学习检查清单
// ============================================================

/**
 * ✅ Phase 0 学习检查
 *
 * 工程化架构：
 * - [ ] 理解 Monorepo 的优势和 Yarn Workspaces 工作原理
 * - [ ] 理解构建流程（Rollup 插件链）
 * - [ ] 理解条件编译机制（__DEV__）
 * - [ ] 理解构建优化策略（Closure Compiler、错误码）
 *
 * 包设计：
 * - [ ] 能画出包架构图
 * - [ ] 理解每个核心包的职责
 * - [ ] 理解 react 和 react-reconciler 的分离设计
 *
 * 核心设计：
 * - [ ] 理解 Dispatcher 模式
 * - [ ] 理解 HostConfig 接口
 * - [ ] 能说出 react-reconciler 的核心文件
 */

export {
  bundleTypes,
  moduleTypes,
  reactBundleConfig,
  rollupPluginChain,
  conditionalCompilationExample,
  errorCodeExample,
  forksExample,
  buildOutputStructure,
  packageArchitecture,
  reactPackageStructure,
  reconcilerPackageStructure,
  reactDomPackageStructure,
  schedulerPackageStructure,
  dispatcherPatternExample,
  reconcilerEntryExample,
  hostConfigExample,
  interviewQuestions,
};
