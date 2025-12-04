/**
 * ============================================================
 * 📚 微前端架构
 * ============================================================
 *
 * 面试考察重点：
 * 1. 微前端的概念和价值
 * 2. 主流方案对比
 * 3. 核心问题解决（沙箱、样式隔离、通信）
 * 4. 实践中的坑和解决方案
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是微前端？
 *
 * 将前端应用拆分为多个独立的子应用，可独立开发、部署、运行。
 *
 * 📊 微前端的价值
 *
 * 1. 技术栈无关：不同子应用可用不同框架
 * 2. 独立开发部署：团队自治
 * 3. 增量升级：渐进式重构
 * 4. 独立运行时：子应用隔离
 *
 * 📊 适用场景
 *
 * ✅ 适合：
 * - 大型系统拆分
 * - 多团队协作
 * - 老系统渐进重构
 * - 聚合多个独立系统
 *
 * ❌ 不适合：
 * - 小型项目
 * - 单一团队维护
 * - 对性能要求极高
 */

// ============================================================
// 2. 主流方案对比
// ============================================================

/**
 * 📊 微前端方案对比
 *
 * ┌──────────────────┬─────────────────────────────────────────────────┐
 * │ 方案              │ 特点                                            │
 * ├──────────────────┼─────────────────────────────────────────────────┤
 * │ iframe           │ 天然隔离，但体验差（白屏、通信复杂、SEO 差）       │
 * │ qiankun          │ 基于 single-spa，成熟稳定，阿里出品               │
 * │ micro-app        │ 类 WebComponent，京东出品，接入简单               │
 * │ Module Federation│ Webpack 5 原生，运行时共享模块                    │
 * │ wujie            │ 基于 WebComponent + iframe，腾讯出品             │
 * │ Garfish          │ 字节出品，支持多框架                             │
 * └──────────────────┴─────────────────────────────────────────────────┘
 */

// ============================================================
// 3. qiankun 核心原理
// ============================================================

/**
 * 📊 qiankun 工作流程
 *
 * 1. 注册子应用（路由 + 入口）
 * 2. 加载子应用（HTML Entry）
 * 3. 解析并执行子应用的 JS/CSS
 * 4. 创建沙箱隔离环境
 * 5. 挂载子应用到指定容器
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                         主应用                                  │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                    路由分发                              │   │
 * │  └──────────────┬──────────────────┬──────────────────────┘   │
 * │                 │                  │                          │
 * │       ┌─────────▼──────┐  ┌────────▼───────┐                  │
 * │       │   子应用 A     │  │   子应用 B      │                  │
 * │       │  (React)      │  │   (Vue)        │                  │
 * │       │  [JS 沙箱]     │  │  [JS 沙箱]      │                  │
 * │       │  [样式隔离]    │  │  [样式隔离]     │                  │
 * │       └───────────────┘  └────────────────┘                  │
 * │                                                               │
 * └─────────────────────────────────────────────────────────────────┘
 */

// 主应用配置
const qiankunMainApp = `
// main.ts
import { registerMicroApps, start } from 'qiankun';

registerMicroApps([
  {
    name: 'app-react',
    entry: '//localhost:3001',
    container: '#subapp-container',
    activeRule: '/react',
    props: { token: 'xxx' },
  },
  {
    name: 'app-vue',
    entry: '//localhost:3002',
    container: '#subapp-container',
    activeRule: '/vue',
  },
]);

// 启动
start({
  sandbox: {
    strictStyleIsolation: true, // Shadow DOM 隔离
    // experimentalStyleIsolation: true, // scoped CSS
  },
  prefetch: 'all', // 预加载
});
`;

// 子应用配置
const qiankunSubApp = `
// React 子应用
// src/index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

let root: ReactDOM.Root | null = null;

// 独立运行时
if (!window.__POWERED_BY_QIANKUN__) {
  root = ReactDOM.createRoot(document.getElementById('root')!);
  root.render(<App />);
}

// 生命周期钩子
export async function bootstrap() {
  console.log('React app bootstrapped');
}

export async function mount(props: any) {
  const { container, token } = props;
  const dom = container 
    ? container.querySelector('#root') 
    : document.getElementById('root');
  
  root = ReactDOM.createRoot(dom);
  root.render(<App token={token} />);
}

export async function unmount() {
  root?.unmount();
  root = null;
}

// webpack 配置
module.exports = {
  output: {
    library: 'app-react',
    libraryTarget: 'umd',
    // jsonpFunction: 'webpackJsonp_app_react', // Webpack 4
    chunkLoadingGlobal: 'webpackJsonp_app_react', // Webpack 5
  },
  devServer: {
    headers: {
      'Access-Control-Allow-Origin': '*', // 允许跨域
    },
  },
};
`;

// ============================================================
// 4. JS 沙箱实现
// ============================================================

/**
 * 📊 JS 沙箱类型
 *
 * 1. 快照沙箱（SnapshotSandbox）
 *    - 进入时保存 window 快照
 *    - 退出时恢复快照
 *    - 不支持多实例
 *
 * 2. 代理沙箱（ProxySandbox）
 *    - 基于 Proxy 拦截
 *    - 支持多实例
 *    - qiankun 默认方案
 */

// 快照沙箱实现
class SnapshotSandbox {
  private windowSnapshot: Record<string, any> = {};
  private modifyPropsMap: Record<string, any> = {};
  private active = false;

  activate() {
    // 1. 保存当前 window 快照
    this.windowSnapshot = {};
    for (const key in window) {
      this.windowSnapshot[key] = (window as any)[key];
    }

    // 2. 恢复之前的修改
    Object.keys(this.modifyPropsMap).forEach(key => {
      (window as any)[key] = this.modifyPropsMap[key];
    });

    this.active = true;
  }

  deactivate() {
    // 记录修改，恢复快照
    this.modifyPropsMap = {};
    for (const key in window) {
      if ((window as any)[key] !== this.windowSnapshot[key]) {
        this.modifyPropsMap[key] = (window as any)[key];
        (window as any)[key] = this.windowSnapshot[key];
      }
    }

    this.active = false;
  }
}

// 代理沙箱实现
class ProxySandbox {
  private proxy: Window;
  private running = false;
  private fakeWindow: Record<string, any> = {};

  constructor() {
    const rawWindow = window;
    const fakeWindow = this.fakeWindow;

    this.proxy = new Proxy(fakeWindow, {
      get(target, key) {
        // 优先从 fakeWindow 获取
        if (key in target) {
          return target[key as string];
        }
        // 否则从真实 window 获取
        const value = (rawWindow as any)[key];
        return typeof value === 'function' ? value.bind(rawWindow) : value;
      },

      set(target, key, value) {
        if (this.running) {
          target[key as string] = value;
        }
        return true;
      },

      has(target, key) {
        return key in target || key in rawWindow;
      },
    }) as unknown as Window;
  }

  activate() {
    this.running = true;
  }

  deactivate() {
    this.running = false;
  }

  getProxy() {
    return this.proxy;
  }
}

// ============================================================
// 5. 样式隔离
// ============================================================

/**
 * 📊 样式隔离方案
 *
 * 1. Shadow DOM
 *    - 完全隔离
 *    - 但弹窗等挂载到 body 的组件会有问题
 *
 * 2. CSS Scoped
 *    - 给选择器加前缀
 *    - 类似 Vue scoped
 *
 * 3. CSS Modules
 *    - 编译时处理
 *    - 类名 hash 化
 *
 * 4. CSS-in-JS
 *    - 运行时生成
 *    - 天然隔离
 */

// 运行时 CSS 作用域
function scopedCSS(css: string, prefix: string): string {
  // 简化实现：给选择器加前缀
  return css.replace(
    /([^{}]+)\{/g,
    (match, selector) => {
      const scoped = selector
        .split(',')
        .map((s: string) => `${prefix} ${s.trim()}`)
        .join(', ');
      return `${scoped}{`;
    }
  );
}

// ============================================================
// 6. 应用通信
// ============================================================

/**
 * 📊 微前端通信方案
 *
 * 1. props 传递
 *    - 主应用通过 props 传递给子应用
 *
 * 2. 全局状态
 *    - qiankun 的 initGlobalState
 *
 * 3. 自定义事件
 *    - CustomEvent + addEventListener
 *
 * 4. 发布订阅
 *    - EventEmitter
 */

// qiankun 全局状态
const globalStateExample = `
// 主应用
import { initGlobalState, MicroAppStateActions } from 'qiankun';

const state = { user: null, token: '' };
const actions: MicroAppStateActions = initGlobalState(state);

// 监听变化
actions.onGlobalStateChange((newState, prev) => {
  console.log('Global state changed:', newState);
});

// 设置状态
actions.setGlobalState({ user: { name: 'Tom' } });

// 子应用
export function mount(props: any) {
  props.onGlobalStateChange((state, prev) => {
    console.log('Sub app received:', state);
  });
  
  props.setGlobalState({ count: 1 });
}
`;

// 自定义事件通信
class MicroAppEventBus {
  private events: Map<string, Function[]> = new Map();

  // 发送到主应用
  static dispatchToMain(type: string, detail: any) {
    window.dispatchEvent(new CustomEvent(`micro-app:${type}`, { detail }));
  }

  // 发送到子应用
  static dispatchToSub(type: string, detail: any) {
    window.dispatchEvent(new CustomEvent(`main-app:${type}`, { detail }));
  }

  // 监听
  static listen(type: string, handler: (e: CustomEvent) => void) {
    window.addEventListener(type, handler as EventListener);
    return () => window.removeEventListener(type, handler as EventListener);
  }
}

// ============================================================
// 7. Module Federation
// ============================================================

/**
 * 📊 Module Federation
 *
 * Webpack 5 原生支持，运行时共享模块
 *
 * 优势：
 * - 真正的模块共享
 * - 无需额外框架
 * - 支持双向依赖
 */

const moduleFederationConfig = `
// 远程应用（提供模块）
// webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'remote_app',
      filename: 'remoteEntry.js',
      exposes: {
        './Button': './src/components/Button',
        './utils': './src/utils',
      },
      shared: {
        react: { singleton: true, requiredVersion: '^18.0.0' },
        'react-dom': { singleton: true, requiredVersion: '^18.0.0' },
      },
    }),
  ],
};

// 主应用（消费模块）
module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'host_app',
      remotes: {
        remote_app: 'remote_app@http://localhost:3001/remoteEntry.js',
      },
      shared: {
        react: { singleton: true },
        'react-dom': { singleton: true },
      },
    }),
  ],
};

// 使用远程模块
const RemoteButton = React.lazy(() => import('remote_app/Button'));

function App() {
  return (
    <Suspense fallback="Loading...">
      <RemoteButton />
    </Suspense>
  );
}
`;

// ============================================================
// 8. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. 路由冲突
 *    - 主应用和子应用路由要规划好
 *    - 子应用使用 memory 路由或加 base
 *
 * 2. 样式污染
 *    - 全局样式会互相影响
 *    - 使用 Shadow DOM 或 CSS Scoped
 *
 * 3. JS 全局变量污染
 *    - 子应用可能修改 window
 *    - 使用沙箱隔离
 *
 * 4. 资源加载问题
 *    - 相对路径变绝对路径
 *    - 配置 publicPath
 *
 * 5. 性能问题
 *    - 首次加载慢
 *    - 使用预加载
 *    - 共享公共依赖
 *
 * 6. 弹窗/Modal 问题
 *    - Shadow DOM 下弹窗样式丢失
 *    - 挂载到主应用 body
 */

// ============================================================
// 9. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 微前端有哪些实现方案？各自优缺点？
 * A:
 *    iframe：隔离好但体验差
 *    qiankun：成熟但有侵入性
 *    Module Federation：运行时共享，但需 Webpack 5
 *
 * Q2: 如何实现 JS 沙箱？
 * A:
 *    快照沙箱：保存/恢复 window
 *    代理沙箱：Proxy 拦截，支持多实例
 *
 * Q3: 样式隔离有哪些方案？
 * A:
 *    Shadow DOM、CSS Scoped、CSS Modules、CSS-in-JS
 *
 * Q4: 子应用之间如何通信？
 * A:
 *    props、全局状态、CustomEvent、EventBus
 *
 * Q5: 微前端的性能优化？
 * A:
 *    - 预加载（prefetch）
 *    - 共享依赖
 *    - 按需加载
 *    - 资源缓存
 */

// ============================================================
// 10. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：大型中后台系统微前端改造
 *
 * 背景：
 * - 多个业务线，各自独立开发
 * - 技术栈不统一（React、Vue）
 * - 需要统一门户入口
 *
 * 方案：
 * 1. 主应用：负责布局、路由、登录、权限
 * 2. 子应用：各业务模块
 * 3. 共享：公共组件库、工具函数、用户状态
 *
 * 技术选型：
 * - 框架：qiankun
 * - 样式隔离：experimentalStyleIsolation
 * - 通信：全局状态 + EventBus
 * - 部署：各子应用独立部署
 */

const practicalExample = `
// 主应用架构
src/
├── layouts/
│   └── BasicLayout.tsx      # 统一布局
├── components/
│   └── SubAppContainer.tsx  # 子应用容器
├── micro/
│   ├── apps.ts              # 子应用注册配置
│   ├── lifecycle.ts         # 生命周期钩子
│   └── globalState.ts       # 全局状态
├── utils/
│   └── auth.ts              # 统一认证
└── App.tsx

// 子应用注册
const apps = [
  {
    name: 'dashboard',
    entry: process.env.DASHBOARD_URL,
    activeRule: '/dashboard',
  },
  {
    name: 'order',
    entry: process.env.ORDER_URL,
    activeRule: '/order',
  },
  {
    name: 'legacy-system', // 旧系统
    entry: process.env.LEGACY_URL,
    activeRule: '/legacy',
  },
];
`;

export {
  SnapshotSandbox,
  ProxySandbox,
  scopedCSS,
  MicroAppEventBus,
  qiankunMainApp,
  qiankunSubApp,
  globalStateExample,
  moduleFederationConfig,
  practicalExample,
};

