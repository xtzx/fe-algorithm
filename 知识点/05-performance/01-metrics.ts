/**
 * ============================================================
 * 📚 性能指标体系
 * ============================================================
 *
 * 面试考察重点：
 * 1. 核心 Web 指标（Core Web Vitals）
 * 2. 性能指标的测量方法
 * 3. 指标背后的用户体验含义
 * 4. 如何建立性能监控体系
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是性能指标？
 *
 * 性能指标是衡量用户体验的量化数据：
 * - 加载速度：页面多快能用？
 * - 交互响应：点击多久有反馈？
 * - 视觉稳定：页面会不会跳动？
 *
 * 📊 为什么需要指标？
 * - 没有度量就没有优化
 * - 量化问题，追踪改进
 * - 建立性能基准，防止劣化
 */

// ============================================================
// 2. Core Web Vitals（核心 Web 指标）
// ============================================================

/**
 * 📊 Google 定义的三大核心指标（2024）
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                        Core Web Vitals                                 │
 * ├────────────────────────────────────────────────────────────────────────┤
 * │                                                                        │
 * │   LCP                    INP                     CLS                   │
 * │   Largest Contentful     Interaction to          Cumulative Layout     │
 * │   Paint                  Next Paint              Shift                 │
 * │                                                                        │
 * │   加载性能               交互响应性               视觉稳定性            │
 * │   ─────────             ─────────               ─────────             │
 * │   最大内容绘制           下一次绘制交互           累积布局偏移          │
 * │                                                                        │
 * │   Good: ≤2.5s           Good: ≤200ms            Good: ≤0.1            │
 * │   Poor: >4s             Poor: >500ms            Poor: >0.25           │
 * │                                                                        │
 * └────────────────────────────────────────────────────────────────────────┘
 *
 * ⚠️ 注意：INP 在 2024 年 3 月正式取代 FID 成为核心指标
 */

/**
 * 📊 LCP（Largest Contentful Paint）- 最大内容绘制
 *
 * 【定义】视口内最大内容元素完成渲染的时间
 *
 * 【包含元素】
 * - <img> 图片
 * - <svg> 内的 <image>
 * - <video> 封面图
 * - 带有 background-image 的元素
 * - 文本块元素（<p>、<h1> 等）
 *
 * 【标准】
 * - Good: ≤ 2.5s
 * - Needs Improvement: 2.5s - 4s
 * - Poor: > 4s
 *
 * ⚠️ 易错点：
 * - LCP 元素可能会变化（先是文字，后是图片）
 * - 只统计视口内的元素
 * - 用户交互后停止统计
 *
 * 💡 面试追问：
 * Q: LCP 和 FCP 有什么区别？
 * A: FCP 是首次绘制任意内容，LCP 是最大内容。
 *    FCP 可能只是 loading 图标，LCP 更能反映"有意义"的内容加载完成。
 *
 * Q: 如何优化 LCP？
 * A:
 * - 优化服务器响应时间（TTFB）
 * - 预加载关键资源（<link rel="preload">）
 * - 优化图片（压缩、响应式、懒加载非首屏）
 * - 移除阻塞渲染的资源
 */

/**
 * 📊 INP（Interaction to Next Paint）- 交互到下一次绘制
 *
 * 【定义】用户交互到下一帧渲染的延迟时间（取整个页面生命周期中最差的交互）
 *
 * 【包含交互】
 * - 点击（click）
 * - 触摸（tap）
 * - 键盘输入（keypress）
 * - 不包括：hover、scroll
 *
 * 【标准】
 * - Good: ≤ 200ms
 * - Needs Improvement: 200ms - 500ms
 * - Poor: > 500ms
 *
 * 【INP 组成】
 * INP = Input Delay + Processing Time + Presentation Delay
 *       输入延迟   +   处理时间      +   呈现延迟
 *
 * ⚠️ 易错点：
 * - INP 统计的是整个页面最慢的交互（P98）
 * - 不是首次交互，而是整个生命周期
 * - 主线程繁忙会增加 Input Delay
 *
 * 💡 面试追问：
 * Q: INP 和 FID 有什么区别？
 * A:
 * - FID 只测量首次交互的输入延迟
 * - INP 测量整个生命周期所有交互的响应性
 * - INP 更全面，包括处理时间和渲染时间
 *
 * Q: 如何优化 INP？
 * A:
 * - 减少长任务（Long Task > 50ms）
 * - 使用 requestIdleCallback 延迟非关键任务
 * - 优化事件处理器
 * - 使用 Web Worker 处理计算密集型任务
 */

/**
 * 📊 CLS（Cumulative Layout Shift）- 累积布局偏移
 *
 * 【定义】页面生命周期内所有意外布局偏移的累积分数
 *
 * 【计算公式】
 * CLS = Impact Fraction × Distance Fraction
 *       影响比例       ×   距离比例
 *
 * 【标准】
 * - Good: ≤ 0.1
 * - Needs Improvement: 0.1 - 0.25
 * - Poor: > 0.25
 *
 * 【什么不算布局偏移】
 * - 用户主动触发的（点击展开、输入导致的）
 * - 动画（transform 不会触发）
 *
 * ⚠️ 易错点：
 * - CLS 是累积值，不是单次最大值
 * - 只统计视口内可见元素
 * - 新增元素不算偏移，已有元素移动才算
 *
 * 💡 面试追问：
 * Q: 常见导致 CLS 的原因？
 * A:
 * - 图片/视频没有尺寸属性
 * - 动态插入的内容（广告、弹窗）
 * - 异步加载的字体（FOIT/FOUT）
 * - 动态注入的 DOM
 *
 * Q: 如何优化 CLS？
 * A:
 * - 图片/视频添加 width/height 或 aspect-ratio
 * - 预留广告位空间
 * - 使用 font-display: optional 或预加载字体
 * - 避免在已有内容上方插入新内容
 */

// ============================================================
// 3. 其他重要性能指标
// ============================================================

/**
 * 📊 加载相关指标
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │ 时间线：                                                               │
 * │                                                                        │
 * │ 请求 ──► TTFB ──► FP ──► FCP ──► LCP ──► TTI ──► Load                 │
 * │  │        │       │       │       │       │        │                   │
 * │  │        │       │       │       │       │        └─ 完全加载         │
 * │  │        │       │       │       │       └─ 可交互                    │
 * │  │        │       │       │       └─ 最大内容绘制                      │
 * │  │        │       │       └─ 首次内容绘制                              │
 * │  │        │       └─ 首次绘制（任意像素）                              │
 * │  │        └─ 首字节时间                                                │
 * │  └─ 请求开始                                                           │
 * └────────────────────────────────────────────────────────────────────────┘
 */

/**
 * TTFB（Time to First Byte）- 首字节时间
 *
 * 从请求到收到第一个字节的时间
 * 包括：DNS、TCP、TLS、服务器处理
 *
 * Good: ≤ 800ms
 *
 * 💡 追问：TTFB 慢怎么排查？
 * - 检查 DNS 解析时间
 * - 检查服务器响应时间
 * - 考虑使用 CDN
 */

/**
 * FCP（First Contentful Paint）- 首次内容绘制
 *
 * 首次渲染任何内容（文字、图片、canvas 等）
 *
 * Good: ≤ 1.8s
 *
 * 💡 追问：白屏时间怎么计算？
 * 白屏时间 ≈ FP 或 FCP 时间
 */

/**
 * TTI（Time to Interactive）- 可交互时间
 *
 * 页面完全可交互的时间点：
 * - FCP 已完成
 * - 主线程空闲（5s 内没有长任务）
 * - 已注册事件处理器
 *
 * Good: ≤ 3.8s
 *
 * ⚠️ 注意：TTI 已从 Lighthouse 核心指标移除，但仍有参考价值
 */

/**
 * TBT（Total Blocking Time）- 总阻塞时间
 *
 * FCP 到 TTI 之间，长任务阻塞主线程的总时间
 * 长任务 = 执行时间 > 50ms 的任务
 * 阻塞时间 = 任务时间 - 50ms
 *
 * Good: ≤ 200ms
 *
 * 💡 追问：TBT 和 INP 的关系？
 * TBT 高通常意味着 INP 也差，因为主线程被阻塞
 */

// ============================================================
// 4. 代码实现 - 性能指标采集
// ============================================================

/**
 * 💻 使用 Performance API 采集指标
 */

// 4.1 采集 Core Web Vitals
function collectWebVitals() {
  // LCP
  const lcpObserver = new PerformanceObserver((list) => {
    const entries = list.getEntries();
    const lastEntry = entries[entries.length - 1];
    console.log('LCP:', lastEntry.startTime);
  });
  lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });

  // CLS
  let clsValue = 0;
  const clsObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      // 只统计非用户触发的偏移
      if (!(entry as any).hadRecentInput) {
        clsValue += (entry as any).value;
      }
    }
    console.log('CLS:', clsValue);
  });
  clsObserver.observe({ type: 'layout-shift', buffered: true });

  // FCP
  const fcpObserver = new PerformanceObserver((list) => {
    const entries = list.getEntries();
    const fcp = entries.find(e => e.name === 'first-contentful-paint');
    if (fcp) {
      console.log('FCP:', fcp.startTime);
    }
  });
  fcpObserver.observe({ type: 'paint', buffered: true });
}

// 4.2 采集资源加载时间
function collectResourceTiming() {
  const resources = performance.getEntriesByType('resource');
  
  resources.forEach(resource => {
    const timing = resource as PerformanceResourceTiming;
    console.log({
      name: timing.name,
      type: timing.initiatorType,
      duration: timing.duration,
      transferSize: timing.transferSize,
      // 各阶段耗时
      dns: timing.domainLookupEnd - timing.domainLookupStart,
      tcp: timing.connectEnd - timing.connectStart,
      ttfb: timing.responseStart - timing.requestStart,
      download: timing.responseEnd - timing.responseStart,
    });
  });
}

// 4.3 采集长任务
function collectLongTasks() {
  const longTaskObserver = new PerformanceObserver((list) => {
    list.getEntries().forEach(entry => {
      console.log('Long Task:', {
        duration: entry.duration,
        startTime: entry.startTime,
        // 任务来源（如果支持）
        attribution: (entry as any).attribution,
      });
    });
  });
  longTaskObserver.observe({ type: 'longtask', buffered: true });
}

// 4.4 完整的性能监控类
class PerformanceMonitor {
  private metrics: Record<string, number> = {};
  private observers: PerformanceObserver[] = [];

  start() {
    this.collectNavigationTiming();
    this.collectPaintTiming();
    this.collectLCP();
    this.collectCLS();
    this.collectLongTasks();
  }

  private collectNavigationTiming() {
    // 等待页面加载完成
    window.addEventListener('load', () => {
      setTimeout(() => {
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        
        this.metrics['dns'] = navigation.domainLookupEnd - navigation.domainLookupStart;
        this.metrics['tcp'] = navigation.connectEnd - navigation.connectStart;
        this.metrics['ttfb'] = navigation.responseStart - navigation.requestStart;
        this.metrics['domParse'] = navigation.domInteractive - navigation.responseEnd;
        this.metrics['domReady'] = navigation.domContentLoadedEventEnd - navigation.fetchStart;
        this.metrics['load'] = navigation.loadEventEnd - navigation.fetchStart;
      }, 0);
    });
  }

  private collectPaintTiming() {
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach(entry => {
        if (entry.name === 'first-paint') {
          this.metrics['fp'] = entry.startTime;
        }
        if (entry.name === 'first-contentful-paint') {
          this.metrics['fcp'] = entry.startTime;
        }
      });
    });
    observer.observe({ type: 'paint', buffered: true });
    this.observers.push(observer);
  }

  private collectLCP() {
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.metrics['lcp'] = lastEntry.startTime;
    });
    observer.observe({ type: 'largest-contentful-paint', buffered: true });
    this.observers.push(observer);
  }

  private collectCLS() {
    let clsValue = 0;
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!(entry as any).hadRecentInput) {
          clsValue += (entry as any).value;
        }
      }
      this.metrics['cls'] = clsValue;
    });
    observer.observe({ type: 'layout-shift', buffered: true });
    this.observers.push(observer);
  }

  private collectLongTasks() {
    let totalBlockingTime = 0;
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach(entry => {
        // 超过 50ms 的部分算阻塞时间
        const blockingTime = entry.duration - 50;
        if (blockingTime > 0) {
          totalBlockingTime += blockingTime;
        }
      });
      this.metrics['tbt'] = totalBlockingTime;
    });
    observer.observe({ type: 'longtask', buffered: true });
    this.observers.push(observer);
  }

  getMetrics() {
    return { ...this.metrics };
  }

  // 上报数据
  report() {
    const metrics = this.getMetrics();
    // 使用 sendBeacon 确保数据发送
    navigator.sendBeacon('/api/performance', JSON.stringify(metrics));
  }

  destroy() {
    this.observers.forEach(o => o.disconnect());
  }
}

// ============================================================
// 5. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 只关注实验室数据，忽略真实用户数据
 *    - Lighthouse 是模拟环境，真实用户设备和网络差异大
 *    - 应该同时关注 RUM（Real User Monitoring）数据
 *
 * 2. 只看平均值，忽略分位数
 *    - P50 可能很好，但 P95 可能很差
 *    - 应该关注 P75、P90、P95
 *
 * 3. 指标优化不等于体验优化
 *    - 可能为了指标做出伤害体验的事
 *    - 比如延迟加载首屏内容来降低 LCP
 *
 * 4. 忽略移动端
 *    - 移动端通常是性能瓶颈
 *    - CPU 和网络都比桌面端差
 *
 * 5. 采样率设置不当
 *    - 100% 采集可能影响性能
 *    - 但采样率太低可能遗漏问题
 */

// ============================================================
// 6. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何建立性能基准？
 * A:
 * - 竞品对比：和竞争对手比
 * - 历史对比：和自己历史数据比
 * - 行业标准：Google 的 Good/Poor 阈值
 * - 用户研究：用户满意度和性能的关系
 *
 * Q2: 性能指标如何与业务指标关联？
 * A:
 * - 建立数据看板，关联性能和业务指标
 * - 分析：LCP 每提升 100ms，转化率提升 X%
 * - Amazon：每 100ms 延迟损失 1% 销售额
 * - Google：500ms 延迟减少 20% 流量
 *
 * Q3: 实验室数据和真实用户数据有什么区别？
 * A:
 * - 实验室（Lab）：Lighthouse，可复现，控制环境
 * - 真实（Field）：RUM，真实用户，设备多样
 * - 两者都要关注，实验室用于调试，真实用于监控
 *
 * Q4: 如何设计性能监控告警？
 * A:
 * - 绝对阈值：超过 3s 告警
 * - 相对阈值：比上周同期差 20% 告警
 * - 分位数告警：P95 超标告警
 * - 分维度：区分设备、网络、地域
 */

// ============================================================
// 7. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：首页性能优化项目
 *
 * 背景：首页 LCP 从 4.5s 优化到 2.0s
 *
 * 步骤：
 * 1. 建立监控：接入 Web Vitals 采集
 * 2. 分析数据：发现 LCP 元素是轮播图
 * 3. 定位问题：图片太大 + 无预加载
 * 4. 优化措施：
 *    - 图片压缩 + WebP
 *    - 首图 preload
 *    - 非首屏图片懒加载
 * 5. 验证效果：LCP 降到 2.0s，转化率提升 15%
 */

/**
 * 🏢 场景 2：性能监控体系建设
 *
 * 技术方案：
 * 1. 数据采集：
 *    - SDK：web-vitals + 自研埋点
 *    - 采样率：10%（可配置）
 *
 * 2. 数据上报：
 *    - 时机：页面隐藏时（visibilitychange）
 *    - 方式：sendBeacon
 *
 * 3. 数据存储：
 *    - 时序数据库（InfluxDB/ClickHouse）
 *
 * 4. 数据展示：
 *    - Grafana 看板
 *    - 分维度：页面、设备、网络、地域
 *
 * 5. 告警：
 *    - P95 超过阈值
 *    - 环比/同比异常
 */

// 使用 web-vitals 库（推荐）
const webVitalsExample = `
import { onLCP, onINP, onCLS } from 'web-vitals';

function sendToAnalytics(metric) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    rating: metric.rating,
    delta: metric.delta,
    id: metric.id,
  });
  
  // 使用 sendBeacon 确保数据发送
  navigator.sendBeacon('/analytics', body);
}

onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onCLS(sendToAnalytics);
`;

export {
  collectWebVitals,
  collectResourceTiming,
  collectLongTasks,
  PerformanceMonitor,
  webVitalsExample,
};

