/**
 * ============================================================
 * 📚 可视化与大屏
 * ============================================================
 *
 * 面试考察重点：
 * 1. 可视化技术选型
 * 2. Canvas/SVG/WebGL 对比
 * 3. 大数据渲染优化
 * 4. 大屏适配方案
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 可视化技术栈
 *
 * 📊 渲染技术对比
 *
 * ┌──────────────┬────────────────────┬────────────────────┬────────────────────┐
 * │ 特性          │ SVG                │ Canvas 2D          │ WebGL              │
 * ├──────────────┼────────────────────┼────────────────────┼────────────────────┤
 * │ 渲染方式      │ 矢量（DOM）         │ 像素（位图）         │ GPU 加速           │
 * │ 元素数量      │ < 1000             │ < 10000            │ 百万级             │
 * │ 交互         │ 原生 DOM 事件       │ 需要手动计算         │ 需要手动计算        │
 * │ 动画         │ CSS/SMIL           │ requestAnimationFrame│ Shader           │
 * │ 适用场景      │ 图标、简单图表      │ 复杂图表、游戏       │ 3D、大数据量        │
 * │ 学习成本      │ 低                 │ 中                  │ 高                 │
 * └──────────────┴────────────────────┴────────────────────┴────────────────────┘
 *
 * 选型建议：
 * - 元素少、需要交互：SVG
 * - 元素多、动画复杂：Canvas
 * - 3D、超大数据量：WebGL
 */

// ============================================================
// 2. 图表库选型
// ============================================================

/**
 * 📊 主流图表库对比
 *
 * ┌─────────────────┬────────────────────────────────────────────────┐
 * │ 库              │ 特点                                           │
 * ├─────────────────┼────────────────────────────────────────────────┤
 * │ ECharts         │ 百度出品，功能全面，配置丰富，大屏首选           │
 * │ D3.js           │ 底层库，灵活度高，学习曲线陡                    │
 * │ Chart.js        │ 轻量，简单易用，适合简单场景                    │
 * │ AntV G2/G6      │ 蚂蚁出品，图形语法，关系图强                    │
 * │ Highcharts      │ 商业库，文档全面                               │
 * │ Three.js        │ 3D 图形库                                      │
 * └─────────────────┴────────────────────────────────────────────────┘
 */

// ============================================================
// 3. Canvas 核心 API
// ============================================================

/**
 * 📊 Canvas 基础
 */

class CanvasRenderer {
  private ctx: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private dpr: number;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.dpr = window.devicePixelRatio || 1;
    this.setupHiDPI();
  }

  // 高清屏适配
  private setupHiDPI() {
    const { width, height } = this.canvas.getBoundingClientRect();
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  // 绘制矩形
  drawRect(x: number, y: number, width: number, height: number, color: string) {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, width, height);
  }

  // 绘制圆
  drawCircle(x: number, y: number, radius: number, color: string) {
    this.ctx.beginPath();
    this.ctx.arc(x, y, radius, 0, Math.PI * 2);
    this.ctx.fillStyle = color;
    this.ctx.fill();
  }

  // 绘制线
  drawLine(points: { x: number; y: number }[], color: string, width: number = 1) {
    if (points.length < 2) return;

    this.ctx.beginPath();
    this.ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
      this.ctx.lineTo(points[i].x, points[i].y);
    }

    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = width;
    this.ctx.stroke();
  }

  // 绘制文字
  drawText(text: string, x: number, y: number, options: {
    font?: string;
    color?: string;
    align?: CanvasTextAlign;
  } = {}) {
    const { font = '14px sans-serif', color = '#333', align = 'left' } = options;
    this.ctx.font = font;
    this.ctx.fillStyle = color;
    this.ctx.textAlign = align;
    this.ctx.fillText(text, x, y);
  }

  // 清空画布
  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}

// ============================================================
// 4. 大数据渲染优化
// ============================================================

/**
 * 📊 大数据量渲染策略
 *
 * 1. 数据采样：降低数据点数量
 * 2. 分层渲染：静态/动态分离
 * 3. 离屏渲染：预渲染到离屏 Canvas
 * 4. 增量渲染：分帧渲染
 * 5. 虚拟化：只渲染可见区域
 * 6. WebWorker：数据处理放到 Worker
 */

// 数据采样 - 最大最小值采样
function lttbSampling(data: number[], threshold: number): number[] {
  if (data.length <= threshold) return data;

  const sampled: number[] = [];
  const bucketSize = (data.length - 2) / (threshold - 2);

  sampled.push(data[0]); // 保留第一个点

  for (let i = 0; i < threshold - 2; i++) {
    const start = Math.floor((i + 1) * bucketSize) + 1;
    const end = Math.floor((i + 2) * bucketSize) + 1;

    // 找出区间内的最大最小值
    let maxValue = -Infinity;
    let maxIndex = start;

    for (let j = start; j < end && j < data.length; j++) {
      if (data[j] > maxValue) {
        maxValue = data[j];
        maxIndex = j;
      }
    }

    sampled.push(data[maxIndex]);
  }

  sampled.push(data[data.length - 1]); // 保留最后一个点
  return sampled;
}

// 增量渲染
function incrementalRender(
  items: any[],
  renderFn: (item: any) => void,
  batchSize: number = 1000
): Promise<void> {
  return new Promise((resolve) => {
    let index = 0;

    function renderBatch() {
      const end = Math.min(index + batchSize, items.length);

      for (; index < end; index++) {
        renderFn(items[index]);
      }

      if (index < items.length) {
        requestAnimationFrame(renderBatch);
      } else {
        resolve();
      }
    }

    renderBatch();
  });
}

// 离屏渲染
class OffscreenRenderer {
  private offscreenCanvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;

  constructor(width: number, height: number) {
    this.offscreenCanvas = new OffscreenCanvas(width, height);
    this.ctx = this.offscreenCanvas.getContext('2d')!;
  }

  // 预渲染静态内容
  preRender(renderFn: (ctx: OffscreenCanvasRenderingContext2D) => void) {
    renderFn(this.ctx);
  }

  // 获取图像位图
  async getImageBitmap(): Promise<ImageBitmap> {
    return this.offscreenCanvas.transferToImageBitmap();
  }

  // 绘制到主 Canvas
  drawToMain(mainCtx: CanvasRenderingContext2D, x: number = 0, y: number = 0) {
    mainCtx.drawImage(this.offscreenCanvas, x, y);
  }
}

// ============================================================
// 5. 大屏适配方案
// ============================================================

/**
 * 📊 大屏适配方案对比
 *
 * 1. scale 缩放
 *    - 优点：简单，等比缩放
 *    - 缺点：可能有留白或裁剪
 *
 * 2. rem + vw/vh
 *    - 优点：灵活
 *    - 缺点：需要计算
 *
 * 3. CSS 缩放 + 定位
 *    - 优点：精确控制
 *    - 缺点：复杂
 */

// 方案 1：scale 缩放
function scaleScreen(designWidth: number, designHeight: number) {
  const container = document.getElementById('app');
  if (!container) return;

  const scaleX = window.innerWidth / designWidth;
  const scaleY = window.innerHeight / designHeight;
  const scale = Math.min(scaleX, scaleY);

  container.style.transform = `scale(${scale})`;
  container.style.transformOrigin = 'left top';
  container.style.width = `${designWidth}px`;
  container.style.height = `${designHeight}px`;

  // 居中
  const marginLeft = (window.innerWidth - designWidth * scale) / 2;
  const marginTop = (window.innerHeight - designHeight * scale) / 2;
  container.style.marginLeft = `${marginLeft}px`;
  container.style.marginTop = `${marginTop}px`;
}

// 监听窗口变化
const resizeHandler = () => scaleScreen(1920, 1080);
window.addEventListener('resize', resizeHandler);
resizeHandler();

// 方案 2：rem 适配
function setRemUnit(designWidth: number = 1920) {
  const html = document.documentElement;
  const clientWidth = html.clientWidth;
  html.style.fontSize = `${(clientWidth / designWidth) * 100}px`;
}

// ============================================================
// 6. 动画与性能
// ============================================================

/**
 * 📊 动画实现方式
 *
 * 1. requestAnimationFrame
 * 2. CSS Animation
 * 3. Web Animations API
 * 4. 第三方库（GSAP）
 */

// 流畅动画基类
class Animator {
  private animationId: number | null = null;
  private startTime: number = 0;
  private duration: number;
  private easing: (t: number) => number;
  private onUpdate: (progress: number) => void;
  private onComplete?: () => void;

  constructor(options: {
    duration: number;
    easing?: (t: number) => number;
    onUpdate: (progress: number) => void;
    onComplete?: () => void;
  }) {
    this.duration = options.duration;
    this.easing = options.easing || ((t) => t);
    this.onUpdate = options.onUpdate;
    this.onComplete = options.onComplete;
  }

  start() {
    this.startTime = performance.now();
    this.tick();
  }

  private tick = () => {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(elapsed / this.duration, 1);
    const easedProgress = this.easing(progress);

    this.onUpdate(easedProgress);

    if (progress < 1) {
      this.animationId = requestAnimationFrame(this.tick);
    } else {
      this.onComplete?.();
    }
  };

  stop() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
}

// 常用缓动函数
const easings = {
  linear: (t: number) => t,
  easeInQuad: (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
  easeOutCubic: (t: number) => --t * t * t + 1,
  easeOutElastic: (t: number) =>
    Math.pow(2, -10 * t) * Math.sin(((t - 0.075) * (2 * Math.PI)) / 0.3) + 1,
};

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. Canvas 模糊
 *    - 高清屏需要考虑 devicePixelRatio
 *    - 设置 canvas 的实际宽高
 *
 * 2. 大数据卡顿
 *    - 使用数据采样
 *    - 增量渲染
 *    - WebWorker 处理数据
 *
 * 3. 内存泄漏
 *    - 及时清理动画
 *    - 销毁图表实例
 *
 * 4. 大屏适配变形
 *    - 使用等比缩放
 *    - 处理留白区域
 *
 * 5. 交互性能
 *    - Canvas 事件委托
 *    - 减少重绘区域
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: SVG 和 Canvas 如何选择？
 * A:
 *    SVG：元素少、需要交互、矢量图
 *    Canvas：元素多、动画复杂、像素操作
 *
 * Q2: Canvas 如何实现事件交互？
 * A:
 *    - 监听 canvas 事件
 *    - 根据坐标判断点击的元素
 *    - 维护元素的包围盒
 *
 * Q3: 如何优化大数据量图表性能？
 * A:
 *    - 数据采样
 *    - 分层/离屏渲染
 *    - 增量渲染
 *    - WebGL
 *
 * Q4: 大屏如何适配不同分辨率？
 * A:
 *    - scale 等比缩放
 *    - rem + vw/vh
 *    - 处理边界留白
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：实时监控大屏
 */

const dashboardExample = `
// 架构设计
┌─────────────────────────────────────────────────────────────────┐
│                        实时监控大屏                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     数据层                                │  │
│  │  WebSocket 订阅 ──► 数据聚合 ──► 缓存 ──► 更新视图         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     渲染层                                │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │ 折线图   │  │ 柱状图   │  │ 地图     │  │ 数字滚动 │     │  │
│  │  │(Canvas) │  │(Canvas) │  │(WebGL)  │  │ (DOM)   │     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     适配层                                │  │
│  │              scale(1920x1080) + 居中                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

// 关键代码
class Dashboard {
  private charts: Map<string, any> = new Map();
  private ws: WebSocket;

  constructor() {
    this.initResize();
    this.initCharts();
    this.initWebSocket();
  }

  private initResize() {
    const resize = () => scaleScreen(1920, 1080);
    window.addEventListener('resize', resize);
    resize();
  }

  private initCharts() {
    // 初始化各图表
  }

  private initWebSocket() {
    this.ws = new WebSocket('wss://api.example.com/realtime');
    this.ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      this.updateChart(data.chartId, data.value);
    };
  }

  private updateChart(chartId: string, data: any) {
    const chart = this.charts.get(chartId);
    if (chart) {
      chart.setOption({ series: [{ data }] });
    }
  }
}
`;

export {
  CanvasRenderer,
  lttbSampling,
  incrementalRender,
  OffscreenRenderer,
  scaleScreen,
  setRemUnit,
  Animator,
  easings,
  dashboardExample,
};

