/**
 * ============================================================
 * 📚 HTTP 缓存策略
 * ============================================================
 *
 * 面试考察重点：
 * 1. 强缓存与协商缓存
 * 2. 缓存相关的 HTTP 头
 * 3. 缓存策略设计
 * 4. Service Worker 缓存
 */

// ============================================================
// 1. 缓存分类
// ============================================================

/**
 * 📊 浏览器缓存分类
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                     浏览器缓存                               │
 * │                                                             │
 * │  ┌───────────────────┐     ┌───────────────────┐           │
 * │  │    强缓存          │     │    协商缓存        │           │
 * │  │  （本地缓存）       │     │  （对比缓存）      │           │
 * │  │                   │     │                   │           │
 * │  │ 直接使用缓存       │     │ 向服务器验证       │           │
 * │  │ 不发送请求         │     │ 是否可用          │           │
 * │  │                   │     │                   │           │
 * │  │ Expires           │     │ Last-Modified     │           │
 * │  │ Cache-Control     │     │ ETag              │           │
 * │  └───────────────────┘     └───────────────────┘           │
 * │                                                             │
 * │  优先级：强缓存 > 协商缓存                                    │
 * └─────────────────────────────────────────────────────────────┘
 */

/**
 * 📊 缓存判断流程
 *
 * ┌─────────────────┐
 * │   发起请求       │
 * └────────┬────────┘
 *          │
 *          ▼
 * ┌─────────────────┐
 * │  检查强缓存      │
 * │ Cache-Control   │
 * │ Expires         │
 * └────────┬────────┘
 *          │
 *    ┌─────┴─────┐
 *    │           │
 *   有效       无效/过期
 *    │           │
 *    ▼           ▼
 * ┌──────┐  ┌─────────────────┐
 * │200    │  │  协商缓存        │
 * │(from  │  │ If-None-Match   │
 * │cache) │  │ If-Modified-Since│
 * └──────┘  └────────┬────────┘
 *                    │
 *              ┌─────┴─────┐
 *              │           │
 *            未修改       已修改
 *              │           │
 *              ▼           ▼
 *          ┌──────┐    ┌──────┐
 *          │ 304  │    │ 200  │
 *          │使用  │    │新资源│
 *          │缓存  │    │      │
 *          └──────┘    └──────┘
 */

// ============================================================
// 2. 强缓存
// ============================================================

/**
 * 📊 Expires（HTTP/1.0）
 *
 * 响应头：
 * Expires: Wed, 21 Oct 2024 07:28:00 GMT
 *
 * 特点：
 * - 指定绝对过期时间
 * - 依赖客户端时间（可能不准确）
 * - 优先级低于 Cache-Control
 */

/**
 * 📊 Cache-Control（HTTP/1.1）
 *
 * 响应头：
 * Cache-Control: max-age=31536000
 *
 * 常用指令：
 *
 * ┌─────────────────────┬───────────────────────────────────────┐
 * │ 指令                 │ 说明                                  │
 * ├─────────────────────┼───────────────────────────────────────┤
 * │ max-age=<seconds>   │ 缓存有效期（相对时间）                  │
 * │ s-maxage=<seconds>  │ 共享缓存（CDN）有效期                  │
 * │ no-cache            │ 强制协商缓存（不是不缓存！）            │
 * │ no-store            │ 完全不缓存                            │
 * │ private             │ 只能浏览器缓存，不能 CDN 缓存          │
 * │ public              │ 可以被任何缓存                         │
 * │ must-revalidate     │ 过期后必须验证                         │
 * │ immutable           │ 不会变化，不需要验证                    │
 * └─────────────────────┴───────────────────────────────────────┘
 *
 * 常见组合：
 * - 静态资源：max-age=31536000, immutable
 * - HTML：no-cache 或 max-age=0, must-revalidate
 * - 敏感数据：no-store
 */

// ============================================================
// 3. 协商缓存
// ============================================================

/**
 * 📊 Last-Modified / If-Modified-Since
 *
 * 首次请求：
 * 响应头：Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
 *
 * 再次请求：
 * 请求头：If-Modified-Since: Wed, 21 Oct 2024 07:28:00 GMT
 *
 * 服务器判断：
 * - 文件修改时间 ≤ If-Modified-Since → 304
 * - 文件修改时间 > If-Modified-Since → 200 + 新资源
 *
 * 缺点：
 * - 精度只到秒
 * - 文件内容不变但修改时间变了
 * - 分布式环境时间可能不一致
 */

/**
 * 📊 ETag / If-None-Match
 *
 * 首次请求：
 * 响应头：ETag: "abc123"
 *
 * 再次请求：
 * 请求头：If-None-Match: "abc123"
 *
 * 服务器判断：
 * - ETag 相同 → 304
 * - ETag 不同 → 200 + 新资源
 *
 * ETag 生成方式：
 * - 内容哈希
 * - 最后修改时间 + 文件大小
 * - 版本号
 *
 * 优点：
 * - 精确判断内容是否变化
 * - 解决 Last-Modified 的问题
 *
 * 缺点：
 * - 计算 ETag 有开销
 * - 分布式环境需要保证一致性
 *
 * 优先级：ETag > Last-Modified
 */

// ============================================================
// 4. 缓存位置
// ============================================================

/**
 * 📊 缓存存储位置
 *
 * 优先级从高到低：
 *
 * 1. Service Worker Cache
 *    - 可编程控制
 *    - 优先级最高
 *    - 离线可用
 *
 * 2. Memory Cache（内存缓存）
 *    - 读取最快
 *    - Tab 关闭即清除
 *    - 小文件优先
 *
 * 3. Disk Cache（磁盘缓存）
 *    - 容量大
 *    - 持久化
 *    - 大文件优先
 *
 * 4. Push Cache（HTTP/2 推送缓存）
 *    - 生命周期很短
 *    - 只在 HTTP/2 中
 */

// ============================================================
// 5. 缓存策略设计
// ============================================================

/**
 * 📊 不同资源的缓存策略
 *
 * HTML 文件：
 * Cache-Control: no-cache
 * 或 Cache-Control: max-age=0, must-revalidate
 * 原因：入口文件，需要获取最新版本
 *
 * CSS/JS（带哈希）：
 * Cache-Control: max-age=31536000, immutable
 * 原因：文件名含哈希，内容不变名不变
 *
 * 图片/字体：
 * Cache-Control: max-age=31536000
 * 原因：通常不会频繁更新
 *
 * API 请求：
 * Cache-Control: no-store
 * 或根据业务设置较短的 max-age
 * 原因：数据实时性要求
 */

/**
 * 📊 文件哈希策略
 *
 * 目的：利用强缓存，同时能及时更新
 *
 * 实现：
 * 1. 构建时给文件名添加内容哈希
 *    main.abc123.js
 *    styles.def456.css
 *
 * 2. 设置长期强缓存
 *    max-age=31536000
 *
 * 3. HTML 不缓存或协商缓存
 *    引用最新的带哈希文件名
 *
 * 更新流程：
 * 1. 修改 JS 代码
 * 2. 重新构建，生成新哈希 main.xyz789.js
 * 3. HTML 更新引用
 * 4. 用户获取新 HTML，加载新 JS
 * 5. 旧 JS 自然过期
 */

// ============================================================
// 6. CDN 缓存
// ============================================================

/**
 * 📊 CDN 缓存策略
 *
 * 浏览器 → CDN 节点 → 源站
 *
 * CDN 相关头部：
 * - s-maxage：CDN 缓存时间（优先于 max-age）
 * - Vary：根据请求头缓存不同版本
 * - Age：资源在 CDN 的缓存时间
 *
 * 示例：
 * Cache-Control: public, max-age=60, s-maxage=3600
 * 浏览器缓存 60 秒，CDN 缓存 1 小时
 */

/**
 * 📊 CDN 缓存刷新
 *
 * 1. URL 刷新：
 *    刷新指定 URL 的缓存
 *
 * 2. 目录刷新：
 *    刷新整个目录下的缓存
 *
 * 3. 预热：
 *    提前将资源推送到 CDN 节点
 *
 * 最佳实践：
 * - 静态资源使用哈希文件名
 * - 避免频繁刷新 CDN 缓存
 * - 合理设置 TTL
 */

// ============================================================
// 7. Service Worker 缓存
// ============================================================

/**
 * 📊 缓存策略模式
 */

// 1. Cache First（缓存优先）
const cacheFirst = `
  self.addEventListener('fetch', (event) => {
    event.respondWith(
      caches.match(event.request)
        .then((response) => {
          // 有缓存，返回缓存
          if (response) {
            return response;
          }
          // 没缓存，请求网络
          return fetch(event.request);
        })
    );
  });
  
  // 适用：静态资源、不常更新的内容
`;

// 2. Network First（网络优先）
const networkFirst = `
  self.addEventListener('fetch', (event) => {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          // 网络请求成功，更新缓存
          const clone = response.clone();
          caches.open('v1').then((cache) => {
            cache.put(event.request, clone);
          });
          return response;
        })
        .catch(() => {
          // 网络失败，使用缓存
          return caches.match(event.request);
        })
    );
  });
  
  // 适用：需要最新数据，但也要离线支持
`;

// 3. Stale While Revalidate（先返回缓存，后台更新）
const staleWhileRevalidate = `
  self.addEventListener('fetch', (event) => {
    event.respondWith(
      caches.open('v1').then((cache) => {
        return cache.match(event.request).then((response) => {
          const fetchPromise = fetch(event.request).then((networkResponse) => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
          // 立即返回缓存，后台更新
          return response || fetchPromise;
        });
      })
    );
  });
  
  // 适用：频繁访问，允许短暂不一致
`;

// 4. Cache Only
const cacheOnly = `
  event.respondWith(caches.match(event.request));
  
  // 适用：离线页面、预缓存的静态资源
`;

// 5. Network Only
const networkOnly = `
  event.respondWith(fetch(event.request));
  
  // 适用：必须从网络获取的请求
`;

// ============================================================
// 8. 高频面试题
// ============================================================

/**
 * 题目 1：强缓存和协商缓存的区别？
 *
 * 强缓存：
 * - 直接使用缓存，不发送请求
 * - Expires、Cache-Control
 * - 返回 200（from cache）
 *
 * 协商缓存：
 * - 发送请求验证是否可用
 * - Last-Modified/If-Modified-Since
 * - ETag/If-None-Match
 * - 返回 304 或 200
 *
 * 执行顺序：
 * 先检查强缓存，失效后检查协商缓存
 */

/**
 * 题目 2：Cache-Control 的常用指令？
 *
 * max-age：缓存有效期（秒）
 * no-cache：强制协商缓存
 * no-store：不缓存
 * public：可被任何缓存
 * private：只能浏览器缓存
 * immutable：内容不会变
 */

/**
 * 题目 3：ETag 和 Last-Modified 的区别？
 *
 * Last-Modified：
 * - 基于文件修改时间
 * - 精度只到秒
 * - 可能有误判
 *
 * ETag：
 * - 基于内容
 * - 更精确
 * - 计算有开销
 *
 * 优先级：ETag > Last-Modified
 * 建议同时使用
 */

/**
 * 题目 4：如何设计前端缓存策略？
 *
 * 1. HTML：协商缓存或短期缓存
 *    Cache-Control: no-cache
 *
 * 2. CSS/JS/图片：
 *    - 文件名添加哈希
 *    - 长期强缓存
 *    Cache-Control: max-age=31536000, immutable
 *
 * 3. API：根据业务需求
 *    - 实时数据：no-store
 *    - 可缓存数据：短期缓存
 *
 * 4. 使用 CDN 加速
 *
 * 5. Service Worker 离线缓存
 */

export {
  cacheFirst,
  networkFirst,
  staleWhileRevalidate,
  cacheOnly,
  networkOnly,
};

