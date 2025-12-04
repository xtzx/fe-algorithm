/**
 * ============================================================
 * 📚 加载性能优化
 * ============================================================
 *
 * 面试考察重点：
 * 1. 资源加载优化策略
 * 2. 关键渲染路径优化
 * 3. 代码分割与懒加载
 * 4. 图片优化策略
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 加载性能优化的目标
 *
 * 核心目标：让用户更快看到内容、更快可以交互
 *
 * 优化方向：
 * 1. 减少资源体积（压缩、裁剪）
 * 2. 减少请求数量（合并、内联）
 * 3. 优化加载顺序（关键资源优先）
 * 4. 利用缓存（浏览器缓存、CDN）
 * 5. 加速传输（HTTP/2、CDN）
 */

// ============================================================
// 2. 关键渲染路径优化
// ============================================================

/**
 * 📊 关键渲染路径（Critical Rendering Path）
 *
 * HTML → DOM →
 *              → Render Tree → Layout → Paint
 * CSS → CSSOM →
 *
 * 【阻塞行为】
 * - CSS 阻塞渲染（但不阻塞 DOM 解析）
 * - JS 阻塞 DOM 解析（除非 async/defer）
 * - JS 执行前需要 CSSOM 完成（如果 JS 在 CSS 后面）
 *
 * ⚠️ 易错点：
 * - CSS 不阻塞 DOM 解析，但阻塞渲染
 * - JS 阻塞 DOM 解析，因为 JS 可能修改 DOM
 * - JS 依赖 CSSOM，因为 JS 可能访问样式
 */

/**
 * 📊 script 标签的 async 和 defer
 *
 * 普通 script：
 * HTML ──┬── 暂停 ──┬── 继续 ──
 *        │         │
 *     下载 JS   执行 JS
 *
 * async：
 * HTML ────────────────────────
 *        │    │
 *     下载 JS  执行（下载完立即执行，可能打断 HTML 解析）
 *
 * defer：
 * HTML ────────────────────────┬── 执行 JS
 *        │                     │
 *     下载 JS              DOMContentLoaded
 *
 * 💡 面试追问：
 * Q: async 和 defer 怎么选？
 * A:
 * - async：独立脚本，不依赖 DOM 和其他脚本（如统计脚本）
 * - defer：需要 DOM 或有依赖顺序（大部分业务脚本）
 * - 多个 defer 会按顺序执行，多个 async 不保证顺序
 */

// ============================================================
// 3. 资源预加载
// ============================================================

/**
 * 📊 资源提示（Resource Hints）
 *
 * ┌─────────────────┬─────────────────────────────────────────────────────────┐
 * │ 类型             │ 作用                                                     │
 * ├─────────────────┼─────────────────────────────────────────────────────────┤
 * │ dns-prefetch    │ 预解析 DNS                                               │
 * │ preconnect      │ 预建立连接（DNS + TCP + TLS）                             │
 * │ prefetch        │ 预获取资源（低优先级，可能下个页面用）                      │
 * │ preload         │ 预加载资源（高优先级，当前页面一定要用）                    │
 * │ prerender       │ 预渲染页面（Chrome 已废弃，改用 Speculation Rules）        │
 * └─────────────────┴─────────────────────────────────────────────────────────┘
 */

const resourceHintsExample = `
<!-- DNS 预解析：提前解析第三方域名 -->
<link rel="dns-prefetch" href="https://cdn.example.com">

<!-- 预连接：比 dns-prefetch 更进一步，建立完整连接 -->
<link rel="preconnect" href="https://cdn.example.com" crossorigin>

<!-- 预加载：当前页面关键资源 -->
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/hero.jpg" as="image">
<link rel="preload" href="/critical.css" as="style">
<link rel="preload" href="/main.js" as="script">

<!-- 预获取：下个页面可能用到的资源 -->
<link rel="prefetch" href="/next-page.js">

<!-- 模块预加载 -->
<link rel="modulepreload" href="/module.js">
`;

/**
 * ⚠️ preload 注意事项
 *
 * 1. 必须指定 as 属性，否则会重复下载
 * 2. 字体需要 crossorigin，即使同源
 * 3. 预加载后 3s 内未使用会有控制台警告
 * 4. 不要预加载太多资源，会占用带宽
 *
 * 💡 追问：preload 和 prefetch 的优先级？
 * A: preload 是高优先级，会抢占带宽
 *    prefetch 是低优先级，空闲时才加载
 */

// ============================================================
// 4. 代码分割与懒加载
// ============================================================

/**
 * 📊 代码分割策略
 *
 * 1. 入口分割：多入口打包
 * 2. 动态导入：import() 语法
 * 3. 公共代码提取：splitChunks
 */

// 路由懒加载（React）
const routeLazyLoadExample = `
import { lazy, Suspense } from 'react';

// 路由组件懒加载
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));

// 使用
<Suspense fallback={<Loading />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/settings" element={<Settings />} />
  </Routes>
</Suspense>
`;

// 组件懒加载
const componentLazyLoadExample = `
import { lazy, Suspense } from 'react';

// 大组件懒加载
const HeavyChart = lazy(() => import('./HeavyChart'));

function Dashboard() {
  const [showChart, setShowChart] = useState(false);

  return (
    <div>
      <button onClick={() => setShowChart(true)}>显示图表</button>
      {showChart && (
        <Suspense fallback={<Skeleton />}>
          <HeavyChart />
        </Suspense>
      )}
    </div>
  );
}
`;

// Webpack 魔法注释
const webpackMagicComments = `
// 预加载（当前页面需要）
const Modal = lazy(() => import(
  /* webpackPreload: true */
  './Modal'
));

// 预获取（未来可能需要）
const Settings = lazy(() => import(
  /* webpackPrefetch: true */
  './Settings'
));

// 自定义 chunk 名
const Dashboard = lazy(() => import(
  /* webpackChunkName: "dashboard" */
  './Dashboard'
));
`;

/**
 * 📊 SplitChunks 配置最佳实践
 */
const splitChunksConfig = `
// webpack.config.js
optimization: {
  splitChunks: {
    chunks: 'all',
    cacheGroups: {
      // 第三方库
      vendors: {
        test: /[\\\\/]node_modules[\\\\/]/,
        name: 'vendors',
        priority: 10,
      },
      // React 相关
      react: {
        test: /[\\\\/]node_modules[\\\\/](react|react-dom)[\\\\/]/,
        name: 'react',
        priority: 20,
      },
      // 公共代码
      common: {
        minChunks: 2,
        name: 'common',
        priority: 5,
      },
    },
  },
}
`;

// ============================================================
// 5. 图片优化
// ============================================================

/**
 * 📊 图片优化策略
 *
 * 1. 格式选择
 *    - JPEG：照片
 *    - PNG：透明图、图标
 *    - WebP：通用（比 JPEG 小 25-35%）
 *    - AVIF：新一代（比 WebP 更小，但兼容性差）
 *    - SVG：矢量图标
 *
 * 2. 响应式图片
 *    - srcset + sizes
 *    - <picture> 元素
 *
 * 3. 懒加载
 *    - loading="lazy"
 *    - Intersection Observer
 *
 * 4. 压缩
 *    - 有损/无损压缩
 *    - 构建时自动压缩
 */

const responsiveImageExample = `
<!-- srcset：根据屏幕密度/尺寸选择图片 -->
<img
  src="image-800.jpg"
  srcset="
    image-400.jpg 400w,
    image-800.jpg 800w,
    image-1200.jpg 1200w
  "
  sizes="(max-width: 600px) 400px, 800px"
  alt="响应式图片"
>

<!-- picture：更精细的控制 -->
<picture>
  <!-- AVIF 优先 -->
  <source type="image/avif" srcset="image.avif">
  <!-- WebP 次之 -->
  <source type="image/webp" srcset="image.webp">
  <!-- 兜底 -->
  <img src="image.jpg" alt="图片">
</picture>

<!-- 原生懒加载 -->
<img src="image.jpg" loading="lazy" alt="懒加载图片">
`;

// 图片懒加载实现
class ImageLazyLoader {
  private observer: IntersectionObserver;

  constructor() {
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadImage(entry.target as HTMLImageElement);
            this.observer.unobserve(entry.target);
          }
        });
      },
      {
        rootMargin: '50px 0px', // 提前 50px 开始加载
        threshold: 0.01,
      }
    );
  }

  observe(images: NodeListOf<HTMLImageElement>) {
    images.forEach(img => this.observer.observe(img));
  }

  private loadImage(img: HTMLImageElement) {
    const src = img.dataset.src;
    if (src) {
      img.src = src;
      img.removeAttribute('data-src');
    }
  }

  destroy() {
    this.observer.disconnect();
  }
}

// ============================================================
// 6. 其他优化策略
// ============================================================

/**
 * 📊 压缩优化
 *
 * 1. Gzip：通用，压缩率约 70%
 * 2. Brotli：更好，压缩率比 Gzip 高 15-25%
 *
 * Nginx 配置：
 * gzip on;
 * gzip_types text/plain text/css application/json application/javascript;
 * gzip_min_size 1024;
 *
 * brotli on;
 * brotli_types text/plain text/css application/json application/javascript;
 */

/**
 * 📊 HTTP/2 优化
 *
 * - 多路复用：不需要合并请求了
 * - 头部压缩：减少重复头部
 * - 服务器推送：提前推送资源
 *
 * ⚠️ 注意：
 * - HTTP/2 下雪碧图、文件合并反而可能降低性能
 * - 小文件可以独立请求，利用缓存
 */

/**
 * 📊 Service Worker 缓存
 */
const serviceWorkerCache = `
// sw.js
const CACHE_NAME = 'v1';
const ASSETS = [
  '/',
  '/index.html',
  '/main.js',
  '/style.css',
];

// 安装时缓存
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS);
    })
  );
});

// 请求时优先使用缓存
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
`;

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. preload 滥用
 *    - 预加载太多资源反而抢占带宽
 *    - 只预加载首屏关键资源
 *
 * 2. 懒加载首屏内容
 *    - 首屏内容应该直接加载
 *    - 懒加载反而增加 LCP
 *
 * 3. 代码分割粒度不当
 *    - 太细：请求数过多
 *    - 太粗：失去分割意义
 *    - 建议：按路由分割 + 公共代码提取
 *
 * 4. 忽略移动端网络
 *    - 3G 网络下体验差异巨大
 *    - 应该在弱网环境测试
 *
 * 5. Tree Shaking 失效
 *    - 使用 ES Module 才能 Tree Shaking
 *    - 检查第三方库是否支持
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 首屏优化的完整方案？
 * A:
 * 1. 减少关键资源：CSS 内联、JS defer
 * 2. 减少资源体积：压缩、Tree Shaking
 * 3. 优化加载顺序：preload 关键资源
 * 4. 利用缓存：CDN、Service Worker
 * 5. 服务端渲染：SSR/SSG
 *
 * Q2: 如何分析加载性能瓶颈？
 * A:
 * 1. Chrome DevTools Network 面板
 *    - 瀑布图分析
 *    - 慢速网络模拟
 * 2. Lighthouse 分析
 * 3. WebPageTest 详细分析
 * 4. 真实用户监控（RUM）
 *
 * Q3: CDN 的作用和原理？
 * A:
 * - 就近访问：用户访问最近的节点
 * - 缓存加速：边缘节点缓存资源
 * - 负载均衡：分散源站压力
 * - 原理：DNS 解析到最近节点 IP
 *
 * Q4: 如何做构建产物分析？
 * A:
 * - webpack-bundle-analyzer：可视化分析
 * - source-map-explorer：分析 source map
 * - 检查：大文件、重复依赖、未使用代码
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：大型 SPA 首屏优化
 *
 * 问题：首屏加载 5s+，用户流失严重
 *
 * 分析：
 * - main.js 2MB+
 * - 首屏图片未优化
 * - 第三方库全量引入
 *
 * 优化：
 * 1. 代码分割
 *    - 路由懒加载
 *    - 第三方库独立 chunk
 *
 * 2. Tree Shaking
 *    - lodash → lodash-es
 *    - 按需引入 antd
 *
 * 3. 图片优化
 *    - WebP + 压缩
 *    - 首图 preload
 *    - 非首屏懒加载
 *
 * 4. 缓存策略
 *    - CDN 加速
 *    - 长效缓存（contenthash）
 *
 * 结果：首屏 5s → 1.5s
 */

export {
  resourceHintsExample,
  routeLazyLoadExample,
  componentLazyLoadExample,
  webpackMagicComments,
  splitChunksConfig,
  responsiveImageExample,
  ImageLazyLoader,
  serviceWorkerCache,
};

