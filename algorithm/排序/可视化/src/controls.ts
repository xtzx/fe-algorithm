/**
 * 控制面板
 */

import type { DataDistribution } from './visualizer';

// ============================================================================
// 类型定义
// ============================================================================

export interface ControlsConfig {
  algorithms: { id: string; name: string }[];
  defaultSize: number;
  defaultSpeed: number;
  defaultDistribution: DataDistribution;
}

export interface ControlsCallbacks {
  onAlgorithmChange: (algorithmId: string) => void;
  onSizeChange: (size: number) => void;
  onSpeedChange: (speed: number) => void;
  onDistributionChange: (distribution: DataDistribution) => void;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
}

// ============================================================================
// 控制面板类
// ============================================================================

export class Controls {
  private container: HTMLElement;
  private config: ControlsConfig;
  private callbacks: ControlsCallbacks;

  // UI 元素
  private algorithmSelect!: HTMLSelectElement;
  private sizeSlider!: HTMLInputElement;
  private sizeLabel!: HTMLSpanElement;
  private speedSlider!: HTMLInputElement;
  private speedLabel!: HTMLSpanElement;
  private distributionRadios!: HTMLInputElement[];
  private playButton!: HTMLButtonElement;
  private pauseButton!: HTMLButtonElement;
  private stepButton!: HTMLButtonElement;
  private resetButton!: HTMLButtonElement;

  // 统计显示
  private comparisonsDisplay!: HTMLSpanElement;
  private swapsDisplay!: HTMLSpanElement;
  private timeDisplay!: HTMLSpanElement;

  private isPlaying = false;

  constructor(
    container: HTMLElement,
    config: ControlsConfig,
    callbacks: ControlsCallbacks
  ) {
    this.container = container;
    this.config = config;
    this.callbacks = callbacks;
    this.render();
    this.setupEventListeners();
    this.setupKeyboardShortcuts();
  }

  /**
   * 渲染控制面板
   */
  private render(): void {
    this.container.innerHTML = `
      <div class="controls-panel">
        <!-- 第一行：算法和数据规模 -->
        <div class="controls-row">
          <div class="control-group">
            <label>算法</label>
            <select id="algorithm-select">
              ${this.config.algorithms.map(algo =>
                `<option value="${algo.id}">${algo.name}</option>`
              ).join('')}
            </select>
          </div>

          <div class="control-group">
            <label>数据量: <span id="size-label">${this.config.defaultSize}</span></label>
            <input type="range" id="size-slider"
              min="10" max="200" value="${this.config.defaultSize}" step="10">
          </div>
        </div>

        <!-- 第二行：数据分布 -->
        <div class="controls-row">
          <div class="control-group distribution-group">
            <label>数据分布</label>
            <div class="radio-group">
              <label><input type="radio" name="distribution" value="random" checked> 随机</label>
              <label><input type="radio" name="distribution" value="sorted"> 有序</label>
              <label><input type="radio" name="distribution" value="reversed"> 逆序</label>
              <label><input type="radio" name="distribution" value="nearlySorted"> 近乎有序</label>
              <label><input type="radio" name="distribution" value="duplicates"> 重复多</label>
            </div>
          </div>
        </div>

        <!-- 第三行：速度和播放控制 -->
        <div class="controls-row">
          <div class="control-group">
            <label>速度: <span id="speed-label">${this.config.defaultSpeed}x</span></label>
            <input type="range" id="speed-slider"
              min="0.5" max="4" value="${this.config.defaultSpeed}" step="0.5">
          </div>

          <div class="control-group buttons-group">
            <button id="play-btn" class="btn btn-primary">▶ 播放</button>
            <button id="pause-btn" class="btn btn-secondary" disabled>⏸ 暂停</button>
            <button id="step-btn" class="btn btn-secondary">⏭ 步进</button>
            <button id="reset-btn" class="btn btn-secondary">🔄 重置</button>
          </div>
        </div>

        <!-- 第四行：统计信息 -->
        <div class="controls-row stats-row">
          <div class="stat">
            <span class="stat-label">比较次数:</span>
            <span id="comparisons-display" class="stat-value">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">交换次数:</span>
            <span id="swaps-display" class="stat-value">0</span>
          </div>
          <div class="stat">
            <span class="stat-label">耗时:</span>
            <span id="time-display" class="stat-value">0ms</span>
          </div>
        </div>
      </div>
    `;

    // 获取元素引用
    this.algorithmSelect = this.container.querySelector('#algorithm-select')!;
    this.sizeSlider = this.container.querySelector('#size-slider')!;
    this.sizeLabel = this.container.querySelector('#size-label')!;
    this.speedSlider = this.container.querySelector('#speed-slider')!;
    this.speedLabel = this.container.querySelector('#speed-label')!;
    this.distributionRadios = Array.from(
      this.container.querySelectorAll('input[name="distribution"]')
    );
    this.playButton = this.container.querySelector('#play-btn')!;
    this.pauseButton = this.container.querySelector('#pause-btn')!;
    this.stepButton = this.container.querySelector('#step-btn')!;
    this.resetButton = this.container.querySelector('#reset-btn')!;
    this.comparisonsDisplay = this.container.querySelector('#comparisons-display')!;
    this.swapsDisplay = this.container.querySelector('#swaps-display')!;
    this.timeDisplay = this.container.querySelector('#time-display')!;
  }

  /**
   * 设置事件监听
   */
  private setupEventListeners(): void {
    // 算法选择
    this.algorithmSelect.addEventListener('change', () => {
      this.callbacks.onAlgorithmChange(this.algorithmSelect.value);
    });

    // 数据规模
    this.sizeSlider.addEventListener('input', () => {
      const size = parseInt(this.sizeSlider.value);
      this.sizeLabel.textContent = String(size);
      this.callbacks.onSizeChange(size);
    });

    // 速度
    this.speedSlider.addEventListener('input', () => {
      const speed = parseFloat(this.speedSlider.value);
      this.speedLabel.textContent = `${speed}x`;
      this.callbacks.onSpeedChange(speed);
    });

    // 数据分布
    this.distributionRadios.forEach(radio => {
      radio.addEventListener('change', () => {
        if (radio.checked) {
          this.callbacks.onDistributionChange(radio.value as DataDistribution);
        }
      });
    });

    // 播放控制
    this.playButton.addEventListener('click', () => {
      this.setPlaying(true);
      this.callbacks.onPlay();
    });

    this.pauseButton.addEventListener('click', () => {
      this.setPlaying(false);
      this.callbacks.onPause();
    });

    this.stepButton.addEventListener('click', () => {
      this.callbacks.onStep();
    });

    this.resetButton.addEventListener('click', () => {
      this.setPlaying(false);
      this.callbacks.onReset();
    });
  }

  /**
   * 设置键盘快捷键
   */
  private setupKeyboardShortcuts(): void {
    document.addEventListener('keydown', (e) => {
      // 避免在输入框中触发
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) {
        return;
      }

      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (this.isPlaying) {
            this.pauseButton.click();
          } else {
            this.playButton.click();
          }
          break;

        case 'ArrowRight':
          e.preventDefault();
          this.stepButton.click();
          break;

        case 'r':
        case 'R':
          e.preventDefault();
          this.resetButton.click();
          break;

        case '+':
        case '=':
          e.preventDefault();
          this.speedSlider.value = String(
            Math.min(4, parseFloat(this.speedSlider.value) + 0.5)
          );
          this.speedSlider.dispatchEvent(new Event('input'));
          break;

        case '-':
          e.preventDefault();
          this.speedSlider.value = String(
            Math.max(0.5, parseFloat(this.speedSlider.value) - 0.5)
          );
          this.speedSlider.dispatchEvent(new Event('input'));
          break;
      }
    });
  }

  /**
   * 设置播放状态
   */
  setPlaying(playing: boolean): void {
    this.isPlaying = playing;
    this.playButton.disabled = playing;
    this.pauseButton.disabled = !playing;
    this.stepButton.disabled = playing;
  }

  /**
   * 更新统计信息
   */
  updateStats(comparisons: number, swaps: number, time: number): void {
    this.comparisonsDisplay.textContent = String(comparisons);
    this.swapsDisplay.textContent = String(swaps);
    this.timeDisplay.textContent = `${Math.round(time)}ms`;
  }

  /**
   * 重置统计信息
   */
  resetStats(): void {
    this.updateStats(0, 0, 0);
  }

  /**
   * 获取当前配置
   */
  getCurrentConfig(): {
    algorithm: string;
    size: number;
    speed: number;
    distribution: DataDistribution;
  } {
    const checkedRadio = this.distributionRadios.find(r => r.checked);

    return {
      algorithm: this.algorithmSelect.value,
      size: parseInt(this.sizeSlider.value),
      speed: parseFloat(this.speedSlider.value),
      distribution: (checkedRadio?.value ?? 'random') as DataDistribution,
    };
  }
}

