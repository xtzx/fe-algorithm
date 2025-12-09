/**
 * ============================================================
 * 📚 Phase 11: SSR & Hydration - Part 1: SSR 概览与 API
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-dom/server/*.js (服务端渲染)
 * - packages/react-dom/src/client/ReactDOMRoot.js (hydrateRoot)
 *
 * ⏱️ 预计时间：1-2 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: 什么是 SSR（服务端渲染）
// ============================================================

/**
 * 📊 SSR 概念
 */

const ssrConcept = `
📊 什么是 SSR（Server-Side Rendering）

定义
═══════════════════════════════════════════════════════════════════════════════

SSR 是指在服务器端执行 React 组件的渲染逻辑，生成完整的 HTML 字符串，
然后将这个 HTML 发送给浏览器。浏览器收到后可以立即显示内容，
而不需要等待 JavaScript 下载和执行。


CSR vs SSR 对比
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CSR (Client-Side Rendering) 客户端渲染:                                   │
│   ──────────────────────────────────────                                    │
│                                                                             │
│   1. 浏览器请求页面                                                         │
│   2. 服务器返回空的 HTML (<div id="root"></div>)                           │
│   3. 浏览器下载 JS bundle                                                   │
│   4. JS 执行，React 渲染组件                                                │
│   5. 用户看到内容                                                           │
│                                                                             │
│   时间线: ├─ 白屏 ─┤──────────────────────┤                                 │
│           请求     JS 加载完成             首次内容显示                      │
│                                                                             │
│   问题:                                                                     │
│   • 首屏白屏时间长                                                          │
│   • SEO 不友好（搜索引擎看到空 HTML）                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   SSR (Server-Side Rendering) 服务端渲染:                                   │
│   ──────────────────────────────────────                                    │
│                                                                             │
│   1. 浏览器请求页面                                                         │
│   2. 服务器执行 React 渲染，生成完整 HTML                                   │
│   3. 服务器返回带内容的 HTML                                                │
│   4. 浏览器立即显示内容（可见但不可交互）                                   │
│   5. 浏览器下载 JS bundle                                                   │
│   6. JS 执行，React "hydrate" 接管 DOM                                     │
│   7. 页面可交互                                                             │
│                                                                             │
│   时间线: ├─┤──────────────────────────────┤                                │
│           请求 首次内容显示               可交互                             │
│                (FCP 快)                  (hydrate 完成)                     │
│                                                                             │
│   优势:                                                                     │
│   • 首屏内容显示快（FCP 提前）                                              │
│   • SEO 友好（搜索引擎看到完整 HTML）                                       │
│   • 在低端设备/慢网络下体验更好                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


同构（Isomorphic/Universal）应用
═══════════════════════════════════════════════════════════════════════════════

同构是指同一套 React 组件代码可以同时在服务端和客户端运行：

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   React 组件 <App />                                                        │
│        │                                                                    │
│        ├─── 服务端执行 ───► ReactDOMServer.renderToString(<App />)          │
│        │                      ↓                                             │
│        │                    HTML 字符串 → 发送给浏览器                       │
│        │                                                                    │
│        └─── 客户端执行 ───► hydrateRoot(container, <App />)                 │
│                               ↓                                             │
│                             接管已有 DOM，绑定事件                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: SSR 的优势与场景
// ============================================================

/**
 * 📊 SSR 的优势
 */

const ssrAdvantages = `
📊 SSR 的优势与适用场景

SSR 的主要优势
═══════════════════════════════════════════════════════════════════════════════

1. 更快的首次内容绘制 (FCP - First Contentful Paint)
   • 用户更快看到有意义的内容
   • 减少白屏时间
   • 提升用户体验

2. SEO 友好
   • 搜索引擎爬虫可以直接解析 HTML 内容
   • 不依赖 JavaScript 执行
   • 对内容类网站尤其重要

3. 社交媒体分享优化
   • Open Graph、Twitter Card 等元数据可以被正确解析
   • 分享链接时能显示预览图和描述

4. 低端设备/慢网络支持
   • 减少客户端计算负担
   • 内容更快可见


适用场景
═══════════════════════════════════════════════════════════════════════════════

✅ 适合 SSR:
─────────────────────────────────────────────────────────────────
• 内容为主的网站（博客、新闻、文档）
• 需要 SEO 的营销页面
• 首屏加载时间敏感的应用
• 电商产品详情页


❌ 不太需要 SSR:
─────────────────────────────────────────────────────────────────
• 纯后台管理系统
• 需要登录才能访问的内部应用
• 高度交互的单页应用（如在线编辑器）


SSR 的代价
═══════════════════════════════════════════════════════════════════════════════

• 服务器计算负担增加
• 需要 Node.js 服务器（不能纯静态托管）
• 开发复杂度增加（需要考虑服务端/客户端环境差异）
• 需要处理数据获取的时机
`;

// ============================================================
// Part 3: ReactDOMServer API
// ============================================================

/**
 * 📊 ReactDOMServer API
 */

const reactDOMServerAPIs = `
📊 ReactDOMServer API 概览

旧版 API（仍然支持）
═══════════════════════════════════════════════════════════════════════════════

1. renderToString(element)
─────────────────────────────────────────────────────────────────
import { renderToString } from 'react-dom/server';

const html = renderToString(<App />);

• 同步执行，返回完整 HTML 字符串
• 简单但阻塞，适合简单场景
• 不支持 Suspense 数据获取


2. renderToStaticMarkup(element)
─────────────────────────────────────────────────────────────────
import { renderToStaticMarkup } from 'react-dom/server';

const html = renderToStaticMarkup(<App />);

• 类似 renderToString，但不添加 React 内部属性
• 生成的 HTML 无法 hydrate
• 适合纯静态内容（如邮件模板）


3. renderToNodeStream(element) [已废弃]
─────────────────────────────────────────────────────────────────
import { renderToNodeStream } from 'react-dom/server';

const stream = renderToNodeStream(<App />);
stream.pipe(res);

• 流式输出，边渲染边发送
• React 18 中已废弃，推荐使用新 API


React 18 新版 API
═══════════════════════════════════════════════════════════════════════════════

4. renderToPipeableStream(element, options)  ⭐ 推荐
─────────────────────────────────────────────────────────────────
import { renderToPipeableStream } from 'react-dom/server';

const { pipe, abort } = renderToPipeableStream(<App />, {
  bootstrapScripts: ['/main.js'],
  onShellReady() {
    // shell 就绪时开始流式输出
    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/html');
    pipe(res);
  },
  onShellError(error) {
    // shell 渲染出错
    res.statusCode = 500;
    res.send('<h1>Server Error</h1>');
  },
  onAllReady() {
    // 所有内容（包括 Suspense）都就绪
    // 适合爬虫/静态生成场景
  },
  onError(error) {
    console.error(error);
  },
});

// 可选：设置超时
setTimeout(() => abort(), 10000);

• Node.js 环境使用
• 支持 Streaming SSR + Suspense
• 可以边渲染边发送，提升 TTFB


5. renderToReadableStream(element, options)
─────────────────────────────────────────────────────────────────
import { renderToReadableStream } from 'react-dom/server';

const stream = await renderToReadableStream(<App />, {
  bootstrapScripts: ['/main.js'],
});
return new Response(stream, {
  headers: { 'Content-Type': 'text/html' },
});

• Web Streams API（适用于 Edge Runtime、Deno）
• 与 renderToPipeableStream 功能类似
• 适合 Cloudflare Workers、Vercel Edge 等环境
`;

// ============================================================
// Part 4: Hydration API
// ============================================================

/**
 * 📊 Hydration API
 */

const hydrationAPIs = `
📊 Hydration（注水）API

什么是 Hydration？
═══════════════════════════════════════════════════════════════════════════════

Hydration（注水）是指客户端 React 接管服务端渲染的 HTML 的过程：

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   服务端渲染的 HTML:                                                        │
│   <div id="root">                                                           │
│     <h1>Hello</h1>                                                          │
│     <button>Click me</button>                                               │
│   </div>                                                                    │
│                                                                             │
│   此时:                                                                     │
│   • DOM 存在，用户可见                                                      │
│   • 但没有事件监听，按钮点击无反应                                          │
│   • 不是"活"的 React 应用                                                  │
│                                                                             │
│   Hydration 之后:                                                           │
│   • React 遍历已有 DOM                                                      │
│   • 与 Fiber 树对齐                                                         │
│   • 复用 DOM 节点（而不是重新创建）                                         │
│   • 绑定事件监听器                                                          │
│   • 应用变成"活"的                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


React 18 的 hydrateRoot
═══════════════════════════════════════════════════════════════════════════════

import { hydrateRoot } from 'react-dom/client';

hydrateRoot(document.getElementById('root'), <App />);

• React 18 的新 API
• 支持并发特性
• 支持 Selective Hydration（选择性注水）


旧版 hydrate（已废弃）
═══════════════════════════════════════════════════════════════════════════════

import { hydrate } from 'react-dom';

hydrate(<App />, document.getElementById('root'));

• Legacy API，仍然可用但不推荐
• 不支持并发特性


createRoot vs hydrateRoot
═══════════════════════════════════════════════════════════════════════════════

┌────────────────────┬────────────────────────────┬────────────────────────────┐
│ API                │ createRoot                 │ hydrateRoot                │
├────────────────────┼────────────────────────────┼────────────────────────────┤
│ 适用场景           │ 纯客户端渲染               │ SSR 后的客户端接管          │
├────────────────────┼────────────────────────────┼────────────────────────────┤
│ DOM 处理           │ 清空容器，创建新 DOM       │ 复用已有 DOM               │
├────────────────────┼────────────────────────────┼────────────────────────────┤
│ 内部实现           │ createContainer            │ createHydrationContainer   │
├────────────────────┼────────────────────────────┼────────────────────────────┤
│ 渲染模式           │ isHydrating = false        │ isHydrating = true         │
└────────────────────┴────────────────────────────┴────────────────────────────┘
`;

// ============================================================
// Part 5: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 1 面试要点

Q1: 什么是 SSR？为什么需要 SSR？
A: SSR 是在服务端执行 React 渲染生成 HTML 的技术。
   优势：
   - 更快的首屏内容显示（FCP 提前）
   - SEO 友好
   - 社交媒体分享优化
   - 低端设备/慢网络下更好的体验

Q2: CSR 和 SSR 的主要区别是什么？
A: - CSR: 服务器返回空 HTML，客户端 JS 渲染内容
   - SSR: 服务器返回完整 HTML，客户端 hydrate 接管

   SSR 首屏更快可见，但需要服务器计算；
   CSR 首屏白屏长，但服务器负担轻。

Q3: renderToString 和 renderToPipeableStream 的区别？
A: - renderToString: 同步，返回完整 HTML 字符串，阻塞
   - renderToPipeableStream: 流式输出，支持 Suspense，边渲染边发送

   React 18 推荐使用 renderToPipeableStream。

Q4: 什么是 Hydration（注水）？
A: Hydration 是客户端 React 接管服务端渲染 HTML 的过程：
   - 遍历已有 DOM
   - 与 Fiber 树对齐
   - 复用 DOM 节点（而不是重新创建）
   - 绑定事件监听器

   使 SSR 的静态 HTML 变成可交互的 React 应用。

Q5: createRoot 和 hydrateRoot 的区别？
A: - createRoot: 纯客户端渲染，会清空容器
   - hydrateRoot: SSR 后使用，复用已有 DOM

   内部实现上，hydrateRoot 调用 createHydrationContainer，
   开启 isHydrating 模式，尝试复用 DOM 而不是新建。

Q6: 什么时候应该使用 SSR？
A: 适合：内容为主的网站、需要 SEO、首屏敏感的应用
   不太需要：后台管理、内部系统、高度交互的 SPA
`;

export {
  ssrConcept,
  ssrAdvantages,
  reactDOMServerAPIs,
  hydrationAPIs,
  interviewPoints,
};

