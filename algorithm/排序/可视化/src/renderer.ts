/**
 * Canvas 渲染器
 */

import type { VisualizerState } from './visualizer';

// ============================================================================
// 颜色常量
// ============================================================================

export const COLORS = {
  default: '#4A90D9',    // 蓝色 - 默认
  comparing: '#F5A623',  // 黄色 - 比较中
  swapping: '#D0021B',   // 红色 - 交换中
  sorted: '#7ED321',     // 绿色 - 已排序
  pivot: '#9013FE',      // 紫色 - pivot
  background: '#1a1a2e', // 深色背景
  text: '#ffffff',       // 文字颜色
};

// ============================================================================
// 渲染器配置
// ============================================================================

export interface RendererConfig {
  barGap: number;        // 柱子间距
  paddingX: number;      // 水平内边距
  paddingY: number;      // 垂直内边距
  showValues: boolean;   // 是否显示数值
}

const DEFAULT_CONFIG: RendererConfig = {
  barGap: 2,
  paddingX: 20,
  paddingY: 40,
  showValues: true,
};

// ============================================================================
// Canvas 渲染器
// ============================================================================

export class CanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private config: RendererConfig;

  constructor(canvas: HTMLCanvasElement, config: Partial<RendererConfig> = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D context');
    }
    this.ctx = ctx;
    this.config = { ...DEFAULT_CONFIG, ...config };

    // 处理高 DPI
    this.setupHiDPI();
  }

  /**
   * 设置高 DPI 支持
   */
  private setupHiDPI(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();

    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;

    this.ctx.scale(dpr, dpr);

    // 设置 CSS 大小
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
  }

  /**
   * 获取显示尺寸
   */
  private getDisplaySize(): { width: number; height: number } {
    const rect = this.canvas.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }

  /**
   * 清空画布
   */
  clear(): void {
    const { width, height } = this.getDisplaySize();
    this.ctx.fillStyle = COLORS.background;
    this.ctx.fillRect(0, 0, width, height);
  }

  /**
   * 渲染状态
   */
  render(state: VisualizerState): void {
    this.clear();

    const { array, comparing, swapping, sorted, pivot } = state;
    if (array.length === 0) return;

    const { width, height } = this.getDisplaySize();
    const { barGap, paddingX, paddingY, showValues } = this.config;

    // 计算柱子尺寸
    const availableWidth = width - paddingX * 2;
    const availableHeight = height - paddingY * 2;
    const barWidth = (availableWidth - barGap * (array.length - 1)) / array.length;
    const maxValue = Math.max(...array);

    // 绘制每个柱子
    array.forEach((value, index) => {
      const x = paddingX + index * (barWidth + barGap);
      const barHeight = (value / maxValue) * availableHeight;
      const y = height - paddingY - barHeight;

      // 确定颜色
      let color = COLORS.default;
      if (sorted.includes(index)) {
        color = COLORS.sorted;
      } else if (swapping.includes(index)) {
        color = COLORS.swapping;
      } else if (comparing.includes(index)) {
        color = COLORS.comparing;
      } else if (index === pivot) {
        color = COLORS.pivot;
      }

      // 绘制柱子
      this.ctx.fillStyle = color;
      this.ctx.fillRect(x, y, barWidth, barHeight);

      // 绘制数值（如果空间足够）
      if (showValues && barWidth > 15) {
        this.ctx.fillStyle = COLORS.text;
        this.ctx.font = `${Math.min(barWidth * 0.6, 12)}px monospace`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'bottom';
        this.ctx.fillText(
          String(value),
          x + barWidth / 2,
          y - 2
        );
      }
    });
  }

  /**
   * 渲染消息
   */
  renderMessage(message: string): void {
    const { width, height } = this.getDisplaySize();

    this.ctx.fillStyle = COLORS.text;
    this.ctx.font = '16px sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(message, width / 2, height / 2);
  }

  /**
   * 调整画布大小
   */
  resize(): void {
    this.setupHiDPI();
  }
}

// ============================================================================
// SVG 渲染器（备选方案）
// ============================================================================

export class SVGRenderer {
  private svg: SVGSVGElement;
  private config: RendererConfig;

  constructor(container: HTMLElement, config: Partial<RendererConfig> = {}) {
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.setAttribute('width', '100%');
    this.svg.setAttribute('height', '100%');
    container.appendChild(this.svg);
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * 清空
   */
  clear(): void {
    this.svg.innerHTML = '';
  }

  /**
   * 渲染状态
   */
  render(state: VisualizerState): void {
    this.clear();

    const { array, comparing, swapping, sorted, pivot } = state;
    if (array.length === 0) return;

    const rect = this.svg.getBoundingClientRect();
    const { barGap, paddingX, paddingY } = this.config;

    const availableWidth = rect.width - paddingX * 2;
    const availableHeight = rect.height - paddingY * 2;
    const barWidth = (availableWidth - barGap * (array.length - 1)) / array.length;
    const maxValue = Math.max(...array);

    // 背景
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', '100%');
    bg.setAttribute('height', '100%');
    bg.setAttribute('fill', COLORS.background);
    this.svg.appendChild(bg);

    // 柱子
    array.forEach((value, index) => {
      const x = paddingX + index * (barWidth + barGap);
      const barHeight = (value / maxValue) * availableHeight;
      const y = rect.height - paddingY - barHeight;

      let color = COLORS.default;
      if (sorted.includes(index)) {
        color = COLORS.sorted;
      } else if (swapping.includes(index)) {
        color = COLORS.swapping;
      } else if (comparing.includes(index)) {
        color = COLORS.comparing;
      } else if (index === pivot) {
        color = COLORS.pivot;
      }

      const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bar.setAttribute('x', String(x));
      bar.setAttribute('y', String(y));
      bar.setAttribute('width', String(barWidth));
      bar.setAttribute('height', String(barHeight));
      bar.setAttribute('fill', color);
      this.svg.appendChild(bar);
    });
  }
}

