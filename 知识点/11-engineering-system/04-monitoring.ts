/**
 * ============================================================
 * 📚 监控告警体系
 * ============================================================
 *
 * 面试考察重点：
 * 1. 错误监控
 * 2. 性能监控
 * 3. 用户行为监控
 * 4. 告警机制
 */

// ============================================================
// 1. 监控体系概览
// ============================================================

/**
 * 📊 前端监控体系
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                      前端监控体系                               │
 * │                                                                 │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                    数据采集层                            │   │
 * │  │  错误监控 │ 性能监控 │ 用户行为 │ 自定义埋点              │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * │                          │                                      │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                    数据上报层                            │   │
 * │  │  批量上报 │ 采样 │ 压缩 │ 离线存储                        │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * │                          │                                      │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                    数据处理层                            │   │
 * │  │  清洗 │ 聚合 │ 存储 │ 分析                               │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * │                          │                                      │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │                    可视化与告警                          │   │
 * │  │  Dashboard │ 告警规则 │ 通知渠道                         │   │
 * │  └─────────────────────────────────────────────────────────┘   │
 * └─────────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. 错误监控
// ============================================================

/**
 * 📊 错误类型
 *
 * - JS 运行时错误
 * - Promise 未捕获错误
 * - 资源加载错误
 * - 接口错误
 * - 框架错误（React Error Boundary）
 */

// 错误监控 SDK
class ErrorMonitor {
  private queue: ErrorEvent[] = [];
  private config: MonitorConfig;

  constructor(config: MonitorConfig) {
    this.config = config;
    this.init();
  }

  private init() {
    // 1. JS 运行时错误
    window.addEventListener('error', (event) => {
      if (event.target && (event.target as HTMLElement).tagName) {
        // 资源加载错误
        this.report({
          type: 'resource',
          message: `Failed to load ${(event.target as HTMLElement).tagName}`,
          url: (event.target as HTMLImageElement).src || (event.target as HTMLScriptElement).href,
        });
      } else {
        // JS 错误
        this.report({
          type: 'javascript',
          message: event.message,
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          stack: event.error?.stack,
        });
      }
    }, true);

    // 2. Promise 未捕获错误
    window.addEventListener('unhandledrejection', (event) => {
      this.report({
        type: 'promise',
        message: event.reason?.message || String(event.reason),
        stack: event.reason?.stack,
      });
    });

    // 3. 框架错误边界
    this.setupReactErrorBoundary();
  }

  private setupReactErrorBoundary() {
    // 通过 ErrorBoundary 组件捕获
  }

  private report(error: ErrorEvent) {
    // 添加公共信息
    const enrichedError = {
      ...error,
      timestamp: Date.now(),
      url: location.href,
      userAgent: navigator.userAgent,
      userId: this.config.userId,
      sessionId: this.config.sessionId,
    };

    this.queue.push(enrichedError);
    this.flush();
  }

  private flush() {
    if (this.queue.length === 0) return;

    // 使用 sendBeacon 确保数据发送
    const data = JSON.stringify(this.queue);
    if (navigator.sendBeacon) {
      navigator.sendBeacon(this.config.reportUrl, data);
    } else {
      fetch(this.config.reportUrl, {
        method: 'POST',
        body: data,
        keepalive: true,
      });
    }

    this.queue = [];
  }
}

interface ErrorEvent {
  type: string;
  message: string;
  filename?: string;
  lineno?: number;
  colno?: number;
  stack?: string;
  url?: string;
  timestamp?: number;
  userAgent?: string;
  userId?: string;
  sessionId?: string;
}

interface MonitorConfig {
  reportUrl: string;
  userId?: string;
  sessionId?: string;
}

// ============================================================
// 3. 性能监控
// ============================================================

/**
 * 📊 性能指标采集
 *
 * Core Web Vitals：
 * - LCP（Largest Contentful Paint）
 * - FID（First Input Delay）
 * - CLS（Cumulative Layout Shift）
 *
 * 其他指标：
 * - FCP（First Contentful Paint）
 * - TTFB（Time to First Byte）
 * - 资源加载时间
 */

class PerformanceMonitor {
  private metrics: Record<string, number> = {};

  constructor() {
    this.observeWebVitals();
    this.observeResources();
  }

  private observeWebVitals() {
    // LCP
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.metrics.LCP = lastEntry.startTime;
    });
    lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });

    // FID
    const fidObserver = new PerformanceObserver((list) => {
      const entry = list.getEntries()[0] as PerformanceEventTiming;
      this.metrics.FID = entry.processingStart - entry.startTime;
    });
    fidObserver.observe({ type: 'first-input', buffered: true });

    // CLS
    let clsValue = 0;
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as any[]) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      this.metrics.CLS = clsValue;
    });
    clsObserver.observe({ type: 'layout-shift', buffered: true });

    // FCP
    const fcpObserver = new PerformanceObserver((list) => {
      const entry = list.getEntries().find(e => e.name === 'first-contentful-paint');
      if (entry) {
        this.metrics.FCP = entry.startTime;
      }
    });
    fcpObserver.observe({ type: 'paint', buffered: true });
  }

  private observeResources() {
    const resourceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as PerformanceResourceTiming[]) {
        // 资源加载慢于 3s 的记录
        if (entry.duration > 3000) {
          this.reportSlowResource({
            name: entry.name,
            duration: entry.duration,
            type: entry.initiatorType,
          });
        }
      }
    });
    resourceObserver.observe({ type: 'resource', buffered: true });
  }

  private reportSlowResource(resource: { name: string; duration: number; type: string }) {
    console.log('Slow resource:', resource);
    // 上报慢资源
  }

  getMetrics() {
    return this.metrics;
  }
}

// web-vitals 库的使用
const webVitalsExample = `
import { onCLS, onFID, onLCP, onFCP, onTTFB } from 'web-vitals';

function sendToAnalytics(metric) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    id: metric.id,
    delta: metric.delta,
  });

  navigator.sendBeacon('/analytics', body);
}

onCLS(sendToAnalytics);
onFID(sendToAnalytics);
onLCP(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);
`;

// ============================================================
// 4. 用户行为监控
// ============================================================

/**
 * 📊 行为采集类型
 *
 * - 页面访问（PV/UV）
 * - 点击事件
 * - 滚动深度
 * - 停留时长
 * - 用户路径
 */

class BehaviorMonitor {
  private sessionId: string;
  private events: BehaviorEvent[] = [];

  constructor() {
    this.sessionId = this.generateSessionId();
    this.init();
  }

  private generateSessionId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private init() {
    // 页面访问
    this.trackPageView();

    // 点击事件
    document.addEventListener('click', this.handleClick.bind(this), true);

    // 页面离开
    window.addEventListener('beforeunload', () => {
      this.trackPageLeave();
      this.flush();
    });

    // 定时上报
    setInterval(() => this.flush(), 30000);
  }

  private trackPageView() {
    this.track('pageview', {
      url: location.href,
      referrer: document.referrer,
      title: document.title,
    });
  }

  private handleClick(event: MouseEvent) {
    const target = event.target as HTMLElement;

    // 提取点击信息
    const data = {
      tagName: target.tagName,
      className: target.className,
      id: target.id,
      text: target.innerText?.slice(0, 50),
      xpath: this.getXPath(target),
      x: event.clientX,
      y: event.clientY,
    };

    this.track('click', data);
  }

  private trackPageLeave() {
    this.track('pageleave', {
      url: location.href,
      duration: performance.now(),
    });
  }

  private getXPath(element: HTMLElement): string {
    const paths: string[] = [];
    let current: HTMLElement | null = element;

    while (current && current.nodeType === Node.ELEMENT_NODE) {
      let index = 1;
      let sibling = current.previousElementSibling;

      while (sibling) {
        if (sibling.tagName === current.tagName) {
          index++;
        }
        sibling = sibling.previousElementSibling;
      }

      paths.unshift(`${current.tagName.toLowerCase()}[${index}]`);
      current = current.parentElement;
    }

    return `/${paths.join('/')}`;
  }

  track(eventType: string, data: Record<string, any>) {
    this.events.push({
      type: eventType,
      data,
      timestamp: Date.now(),
      sessionId: this.sessionId,
      url: location.href,
    });
  }

  private flush() {
    if (this.events.length === 0) return;

    const data = JSON.stringify(this.events);
    navigator.sendBeacon('/api/behavior', data);
    this.events = [];
  }
}

interface BehaviorEvent {
  type: string;
  data: Record<string, any>;
  timestamp: number;
  sessionId: string;
  url: string;
}

// ============================================================
// 5. 告警机制
// ============================================================

/**
 * 📊 告警配置
 */

const alertConfigExample = `
// 告警规则配置
const alertRules = [
  {
    name: 'JS 错误率告警',
    metric: 'js_error_rate',
    condition: 'gt',
    threshold: 0.01, // 错误率超过 1%
    window: '5m',     // 5 分钟窗口
    severity: 'critical',
    channels: ['slack', 'email', 'phone'],
  },
  {
    name: 'API 错误率告警',
    metric: 'api_error_rate',
    condition: 'gt',
    threshold: 0.05,
    window: '5m',
    severity: 'warning',
    channels: ['slack', 'email'],
  },
  {
    name: 'LCP 性能告警',
    metric: 'lcp_p95',
    condition: 'gt',
    threshold: 2500, // 2.5s
    window: '1h',
    severity: 'warning',
    channels: ['slack'],
  },
];

// 告警通知模板
const alertTemplate = {
  slack: {
    blocks: [
      {
        type: 'header',
        text: { type: 'plain_text', text: '🚨 前端告警' },
      },
      {
        type: 'section',
        fields: [
          { type: 'mrkdwn', text: '*告警名称:*\\n{{name}}' },
          { type: 'mrkdwn', text: '*严重程度:*\\n{{severity}}' },
          { type: 'mrkdwn', text: '*当前值:*\\n{{value}}' },
          { type: 'mrkdwn', text: '*阈值:*\\n{{threshold}}' },
        ],
      },
    ],
  },
};
`;

// ============================================================
// 6. 监控平台选型
// ============================================================

/**
 * 📊 监控平台对比
 *
 * ┌─────────────────┬────────────────────────────────────────────────┐
 * │ 平台            │ 特点                                           │
 * ├─────────────────┼────────────────────────────────────────────────┤
 * │ Sentry          │ 错误监控首选，Source Map 支持好                 │
 * │ Datadog         │ 全栈监控，APM 能力强                            │
 * │ New Relic       │ 企业级，功能全面                               │
 * │ 阿里云 ARMS     │ 国内首选，接入简单                              │
 * │ 自建方案        │ 定制化，但维护成本高                            │
 * └─────────────────┴────────────────────────────────────────────────┘
 */

// Sentry 接入
const sentrySetupExample = `
import * as Sentry from '@sentry/react';

Sentry.init({
  dsn: 'https://xxx@sentry.io/xxx',
  integrations: [
    new Sentry.BrowserTracing(),
    new Sentry.Replay(),
  ],
  tracesSampleRate: 0.1, // 10% 性能采样
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
  environment: process.env.NODE_ENV,
  release: process.env.VERSION,
});

// React Error Boundary
const App = () => (
  <Sentry.ErrorBoundary fallback={<ErrorFallback />}>
    <MyApp />
  </Sentry.ErrorBoundary>
);
`;

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. 数据采集过多
 *    - 影响性能
 *    - 采样和聚合
 *
 * 2. 告警风暴
 *    - 阈值设置不合理
 *    - 告警聚合和收敛
 *
 * 3. Source Map 泄露
 *    - 生产环境不上传
 *    - 或限制访问权限
 *
 * 4. 隐私合规
 *    - 不采集敏感信息
 *    - 符合 GDPR
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何监控 JS 错误？
 * A:
 *    - window.onerror
 *    - addEventListener('error')
 *    - unhandledrejection
 *    - React Error Boundary
 *
 * Q2: 什么是 Core Web Vitals？
 * A:
 *    - LCP：最大内容绘制 < 2.5s
 *    - FID：首次输入延迟 < 100ms
 *    - CLS：累计布局偏移 < 0.1
 *
 * Q3: 如何设计告警策略？
 * A:
 *    - 分级（P0-P3）
 *    - 分渠道（电话/短信/邮件）
 *    - 聚合收敛
 *    - 值班机制
 *
 * Q4: 如何保证数据上报可靠性？
 * A:
 *    - navigator.sendBeacon
 *    - 离线存储
 *    - 重试机制
 */

export {
  ErrorMonitor,
  PerformanceMonitor,
  BehaviorMonitor,
  webVitalsExample,
  alertConfigExample,
  sentrySetupExample,
};

