/**
 * ============================================================
 * 📚 性能监控方案
 * ============================================================
 *
 * 面试考察重点：
 * 1. 性能数据采集
 * 2. 监控系统设计
 * 3. 告警策略
 * 4. 性能分析与归因
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 为什么需要性能监控？
 *
 * 1. 发现问题：及时发现性能劣化
 * 2. 定位问题：找到性能瓶颈
 * 3. 验证优化：量化优化效果
 * 4. 建立基准：设定性能目标
 *
 * 📊 监控类型
 *
 * 1. 实验室数据（Lab Data）
 *    - Lighthouse、WebPageTest
 *    - 可控环境，可复现
 *    - 适合开发调试
 *
 * 2. 真实用户数据（Field Data / RUM）
 *    - 真实用户设备和网络
 *    - 反映实际体验
 *    - 适合生产监控
 */

// ============================================================
// 2. 完整的性能监控 SDK
// ============================================================

interface PerformanceMetrics {
  // 导航时间
  dns?: number;
  tcp?: number;
  ssl?: number;
  ttfb?: number;
  domParse?: number;
  domReady?: number;
  loadComplete?: number;
  
  // Core Web Vitals
  fcp?: number;
  lcp?: number;
  fid?: number;
  inp?: number;
  cls?: number;
  
  // 自定义指标
  [key: string]: number | undefined;
}

interface ResourceMetric {
  name: string;
  type: string;
  duration: number;
  size: number;
  protocol: string;
}

interface ErrorInfo {
  type: 'js' | 'resource' | 'promise' | 'api';
  message: string;
  stack?: string;
  url?: string;
  time: number;
}

class PerformanceSDK {
  private metrics: PerformanceMetrics = {};
  private resources: ResourceMetric[] = [];
  private errors: ErrorInfo[] = [];
  private observers: PerformanceObserver[] = [];
  private config: {
    reportUrl: string;
    sampleRate: number;
    enableResource: boolean;
    enableError: boolean;
  };

  constructor(config: Partial<PerformanceSDK['config']> = {}) {
    this.config = {
      reportUrl: '/api/performance',
      sampleRate: 1, // 采样率 0-1
      enableResource: true,
      enableError: true,
      ...config,
    };

    // 采样
    if (Math.random() > this.config.sampleRate) {
      return;
    }

    this.init();
  }

  private init() {
    this.collectNavigationTiming();
    this.collectPaintTiming();
    this.collectWebVitals();
    
    if (this.config.enableResource) {
      this.collectResourceTiming();
    }
    
    if (this.config.enableError) {
      this.collectErrors();
    }

    // 页面卸载时上报
    this.setupReporting();
  }

  // ==================== 数据采集 ====================

  private collectNavigationTiming() {
    const callback = () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      if (!navigation) return;

      this.metrics.dns = navigation.domainLookupEnd - navigation.domainLookupStart;
      this.metrics.tcp = navigation.connectEnd - navigation.connectStart;
      this.metrics.ssl = navigation.secureConnectionStart > 0
        ? navigation.connectEnd - navigation.secureConnectionStart
        : 0;
      this.metrics.ttfb = navigation.responseStart - navigation.requestStart;
      this.metrics.domParse = navigation.domInteractive - navigation.responseEnd;
      this.metrics.domReady = navigation.domContentLoadedEventEnd - navigation.fetchStart;
      this.metrics.loadComplete = navigation.loadEventEnd - navigation.fetchStart;
    };

    // load 事件后采集
    if (document.readyState === 'complete') {
      setTimeout(callback, 0);
    } else {
      window.addEventListener('load', () => setTimeout(callback, 0));
    }
  }

  private collectPaintTiming() {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-paint') {
            this.metrics.fp = entry.startTime;
          }
          if (entry.name === 'first-contentful-paint') {
            this.metrics.fcp = entry.startTime;
          }
        }
      });
      observer.observe({ type: 'paint', buffered: true });
      this.observers.push(observer);
    } catch (e) {
      console.warn('Paint timing not supported');
    }
  }

  private collectWebVitals() {
    // LCP
    try {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        this.metrics.lcp = lastEntry.startTime;
      });
      lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });
      this.observers.push(lcpObserver);
    } catch (e) {
      console.warn('LCP not supported');
    }

    // FID (First Input Delay)
    try {
      const fidObserver = new PerformanceObserver((list) => {
        const entry = list.getEntries()[0] as any;
        this.metrics.fid = entry.processingStart - entry.startTime;
      });
      fidObserver.observe({ type: 'first-input', buffered: true });
      this.observers.push(fidObserver);
    } catch (e) {
      console.warn('FID not supported');
    }

    // CLS
    try {
      let clsValue = 0;
      const clsObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as any).hadRecentInput) {
            clsValue += (entry as any).value;
          }
        }
        this.metrics.cls = clsValue;
      });
      clsObserver.observe({ type: 'layout-shift', buffered: true });
      this.observers.push(clsObserver);
    } catch (e) {
      console.warn('CLS not supported');
    }

    // INP (Interaction to Next Paint)
    try {
      let maxINP = 0;
      const inpObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const duration = (entry as any).duration;
          if (duration > maxINP) {
            maxINP = duration;
            this.metrics.inp = duration;
          }
        }
      });
      inpObserver.observe({ type: 'event', buffered: true });
      this.observers.push(inpObserver);
    } catch (e) {
      console.warn('INP not supported');
    }
  }

  private collectResourceTiming() {
    const processResources = () => {
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
      
      this.resources = resources.map(r => ({
        name: r.name,
        type: r.initiatorType,
        duration: r.duration,
        size: r.transferSize,
        protocol: r.nextHopProtocol,
      }));
    };

    window.addEventListener('load', () => setTimeout(processResources, 0));
  }

  private collectErrors() {
    // JS 错误
    window.addEventListener('error', (event) => {
      if (event.target && (event.target as HTMLElement).tagName) {
        // 资源加载错误
        this.errors.push({
          type: 'resource',
          message: `Failed to load: ${(event.target as HTMLImageElement).src || (event.target as HTMLScriptElement).href}`,
          url: window.location.href,
          time: Date.now(),
        });
      } else {
        // JS 运行时错误
        this.errors.push({
          type: 'js',
          message: event.message,
          stack: event.error?.stack,
          url: event.filename,
          time: Date.now(),
        });
      }
    }, true);

    // Promise 错误
    window.addEventListener('unhandledrejection', (event) => {
      this.errors.push({
        type: 'promise',
        message: event.reason?.message || String(event.reason),
        stack: event.reason?.stack,
        time: Date.now(),
      });
    });
  }

  // ==================== 数据上报 ====================

  private setupReporting() {
    // 页面隐藏时上报
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.report();
      }
    });

    // 页面卸载时上报（兜底）
    window.addEventListener('pagehide', () => this.report());
  }

  private report() {
    const data = {
      metrics: this.metrics,
      resources: this.resources.slice(0, 50), // 只上报前 50 个资源
      errors: this.errors.slice(0, 20),
      page: {
        url: window.location.href,
        referrer: document.referrer,
        title: document.title,
      },
      device: {
        userAgent: navigator.userAgent,
        connection: (navigator as any).connection?.effectiveType,
        deviceMemory: (navigator as any).deviceMemory,
        hardwareConcurrency: navigator.hardwareConcurrency,
      },
      timestamp: Date.now(),
    };

    // 使用 sendBeacon 确保数据发送
    const success = navigator.sendBeacon(
      this.config.reportUrl,
      JSON.stringify(data)
    );

    // 兜底：fetch keepalive
    if (!success) {
      fetch(this.config.reportUrl, {
        method: 'POST',
        body: JSON.stringify(data),
        keepalive: true,
      }).catch(() => {});
    }
  }

  // ==================== 自定义指标 ====================

  // 标记时间点
  mark(name: string) {
    performance.mark(name);
  }

  // 测量两个标记之间的时间
  measure(name: string, startMark: string, endMark?: string) {
    try {
      const measure = performance.measure(name, startMark, endMark);
      this.metrics[name] = measure.duration;
      return measure.duration;
    } catch (e) {
      console.warn('Measure failed:', e);
      return null;
    }
  }

  // 手动设置指标
  setMetric(name: string, value: number) {
    this.metrics[name] = value;
  }

  // ==================== 清理 ====================

  destroy() {
    this.observers.forEach(o => o.disconnect());
    this.observers = [];
  }
}

// ============================================================
// 3. 告警策略
// ============================================================

/**
 * 📊 告警策略设计
 *
 * 1. 阈值告警
 *    - LCP > 4s
 *    - CLS > 0.25
 *    - INP > 500ms
 *
 * 2. 环比告警
 *    - 比上周同期差 20%
 *
 * 3. 分位数告警
 *    - P95 超过阈值
 *
 * 4. 分维度告警
 *    - 按设备、网络、地域分别告警
 */

interface AlertRule {
  metric: string;
  operator: '>' | '<' | '>=' | '<=';
  threshold: number;
  severity: 'warning' | 'critical';
  message: string;
}

const alertRules: AlertRule[] = [
  {
    metric: 'lcp',
    operator: '>',
    threshold: 4000,
    severity: 'critical',
    message: 'LCP 超过 4s，严重影响用户体验',
  },
  {
    metric: 'lcp',
    operator: '>',
    threshold: 2500,
    severity: 'warning',
    message: 'LCP 超过 2.5s，需要关注',
  },
  {
    metric: 'cls',
    operator: '>',
    threshold: 0.25,
    severity: 'critical',
    message: 'CLS 超过 0.25，页面布局不稳定',
  },
  {
    metric: 'inp',
    operator: '>',
    threshold: 500,
    severity: 'critical',
    message: 'INP 超过 500ms，交互响应慢',
  },
];

function checkAlerts(metrics: PerformanceMetrics): AlertRule[] {
  const triggered: AlertRule[] = [];
  
  for (const rule of alertRules) {
    const value = metrics[rule.metric];
    if (value === undefined) continue;
    
    let shouldAlert = false;
    switch (rule.operator) {
      case '>': shouldAlert = value > rule.threshold; break;
      case '<': shouldAlert = value < rule.threshold; break;
      case '>=': shouldAlert = value >= rule.threshold; break;
      case '<=': shouldAlert = value <= rule.threshold; break;
    }
    
    if (shouldAlert) {
      triggered.push(rule);
    }
  }
  
  return triggered;
}

// ============================================================
// 4. 性能分析与归因
// ============================================================

/**
 * 📊 性能归因分析
 *
 * 1. 按维度分析
 *    - 设备：移动端 vs 桌面端
 *    - 网络：4G vs 3G vs WiFi
 *    - 地域：一线城市 vs 其他
 *    - 浏览器：Chrome vs Safari
 *
 * 2. 时间趋势分析
 *    - 按小时/天/周聚合
 *    - 发现周期性问题
 *
 * 3. 漏斗分析
 *    - 首屏时间 → 可交互时间 → 完全加载
 *    - 找出卡点
 */

interface PerformanceReport {
  period: string;
  metrics: {
    lcp: { p50: number; p75: number; p95: number };
    cls: { p50: number; p75: number; p95: number };
    inp: { p50: number; p75: number; p95: number };
  };
  dimensions: {
    device: Record<string, number>;
    network: Record<string, number>;
    browser: Record<string, number>;
  };
  slowestResources: ResourceMetric[];
  topErrors: { message: string; count: number }[];
}

// ============================================================
// 5. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 采样率设置不当
 *    - 100%：影响性能
 *    - 0.1%：数据不够
 *    - 建议：5%-10%
 *
 * 2. 只看平均值
 *    - 平均值被极值影响
 *    - 应该看 P50、P75、P95
 *
 * 3. 忽略移动端
 *    - 移动端问题更严重
 *    - 需要分设备分析
 *
 * 4. 数据丢失
 *    - 页面卸载时数据丢失
 *    - 必须用 sendBeacon
 *
 * 5. 时区问题
 *    - 用户时区不同
 *    - 统一用服务器时间
 */

// ============================================================
// 6. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何保证监控数据准确性？
 * A:
 * - 多来源校验（SDK + 服务端日志）
 * - 异常值过滤
 * - 采样要随机
 * - 验证数据完整性
 *
 * Q2: 如何降低监控对性能的影响？
 * A:
 * - 采样
 * - 延迟上报（visibilitychange）
 * - 数据压缩
 * - 使用 sendBeacon
 *
 * Q3: 如何设计性能看板？
 * A:
 * - 核心指标趋势图
 * - 分维度对比
 * - 告警列表
 * - 慢资源 TOP N
 * - 错误率统计
 *
 * Q4: 如何做性能劣化检测？
 * A:
 * - 发布前后对比
 * - A/B 测试
 * - 环比/同比分析
 * - 设置性能预算
 */

// ============================================================
// 7. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：前端性能监控体系建设
 *
 * 1. 数据采集层
 *    - 自研 SDK 或 web-vitals
 *    - 采样率 10%
 *    - 上报时机：visibilitychange
 *
 * 2. 数据处理层
 *    - 数据清洗、聚合
 *    - 计算分位数
 *    - 维度分析
 *
 * 3. 存储层
 *    - 时序数据库（InfluxDB/ClickHouse）
 *    - 保留 30 天明细，90 天聚合
 *
 * 4. 展示层
 *    - Grafana 看板
 *    - 核心指标大盘
 *    - 维度下钻
 *
 * 5. 告警层
 *    - 阈值告警
 *    - 环比告警
 *    - 钉钉/飞书通知
 */

// 使用示例
const sdkUsage = `
// 初始化
const sdk = new PerformanceSDK({
  reportUrl: 'https://monitor.example.com/api/report',
  sampleRate: 0.1, // 10% 采样
  enableResource: true,
  enableError: true,
});

// 自定义业务指标
sdk.mark('pageReady');
// ... 业务逻辑 ...
sdk.mark('dataLoaded');
sdk.measure('businessMetric', 'pageReady', 'dataLoaded');

// 手动设置指标
sdk.setMetric('customMetric', 1234);
`;

export {
  PerformanceSDK,
  alertRules,
  checkAlerts,
  sdkUsage,
};

export type {
  PerformanceMetrics,
  ResourceMetric,
  ErrorInfo,
  AlertRule,
  PerformanceReport,
};

