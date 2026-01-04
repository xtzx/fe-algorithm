/**
 * 排序可视化核心引擎
 */

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 排序步骤类型
 */
export interface SortStep {
  type: 'compare' | 'swap' | 'highlight' | 'sorted' | 'pivot';
  indices: number[];
  values?: number[];
  message?: string;
}

/**
 * 算法生成器类型
 */
export type SortGenerator = Generator<SortStep, void, unknown>;

/**
 * 算法函数类型
 */
export type SortAlgorithm = (arr: number[]) => SortGenerator;

/**
 * 可视化器状态
 */
export interface VisualizerState {
  array: number[];
  comparing: number[];
  swapping: number[];
  sorted: number[];
  pivot: number | null;
  stats: {
    comparisons: number;
    swaps: number;
    startTime: number;
    elapsedTime: number;
  };
}

/**
 * 可视化器配置
 */
export interface VisualizerConfig {
  speed: number;        // 延迟倍率，1 = 100ms
  onStep?: (state: VisualizerState) => void;
  onComplete?: (stats: VisualizerState['stats']) => void;
}

// ============================================================================
// 可视化引擎
// ============================================================================

export class Visualizer {
  private array: number[] = [];
  private generator: SortGenerator | null = null;
  private algorithm: SortAlgorithm | null = null;
  private config: VisualizerConfig;

  private isPlaying = false;
  private isPaused = false;
  private animationId: number | null = null;

  private state: VisualizerState = {
    array: [],
    comparing: [],
    swapping: [],
    sorted: [],
    pivot: null,
    stats: {
      comparisons: 0,
      swaps: 0,
      startTime: 0,
      elapsedTime: 0,
    },
  };

  constructor(config: Partial<VisualizerConfig> = {}) {
    this.config = {
      speed: 1,
      ...config,
    };
  }

  /**
   * 设置算法
   */
  setAlgorithm(algorithm: SortAlgorithm): void {
    this.algorithm = algorithm;
    this.reset();
  }

  /**
   * 设置数据
   */
  setArray(array: number[]): void {
    this.array = [...array];
    this.reset();
  }

  /**
   * 设置速度
   */
  setSpeed(speed: number): void {
    this.config.speed = speed;
  }

  /**
   * 获取当前状态
   */
  getState(): VisualizerState {
    return { ...this.state };
  }

  /**
   * 重置
   */
  reset(): void {
    this.stop();

    this.state = {
      array: [...this.array],
      comparing: [],
      swapping: [],
      sorted: [],
      pivot: null,
      stats: {
        comparisons: 0,
        swaps: 0,
        startTime: 0,
        elapsedTime: 0,
      },
    };

    if (this.algorithm) {
      this.generator = this.algorithm([...this.array]);
    }

    this.config.onStep?.(this.state);
  }

  /**
   * 播放
   */
  async play(): Promise<void> {
    if (!this.generator || !this.algorithm) return;
    if (this.isPlaying && !this.isPaused) return;

    this.isPlaying = true;
    this.isPaused = false;

    if (this.state.stats.startTime === 0) {
      this.state.stats.startTime = performance.now();
    }

    await this.runAnimation();
  }

  /**
   * 暂停
   */
  pause(): void {
    this.isPaused = true;
  }

  /**
   * 停止
   */
  stop(): void {
    this.isPlaying = false;
    this.isPaused = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * 单步执行
   */
  step(): boolean {
    if (!this.generator) return false;

    const result = this.generator.next();

    if (result.done) {
      this.onComplete();
      return false;
    }

    this.processStep(result.value);
    return true;
  }

  /**
   * 处理排序步骤
   */
  private processStep(step: SortStep): void {
    // 清除临时状态
    this.state.comparing = [];
    this.state.swapping = [];

    switch (step.type) {
      case 'compare':
        this.state.comparing = step.indices;
        this.state.stats.comparisons++;
        break;

      case 'swap':
        this.state.swapping = step.indices;
        this.state.stats.swaps++;
        // 执行交换
        const [i, j] = step.indices;
        [this.state.array[i], this.state.array[j]] =
          [this.state.array[j], this.state.array[i]];
        break;

      case 'sorted':
        this.state.sorted = [...this.state.sorted, ...step.indices];
        break;

      case 'pivot':
        this.state.pivot = step.indices[0];
        break;

      case 'highlight':
        this.state.comparing = step.indices;
        break;
    }

    this.state.stats.elapsedTime = performance.now() - this.state.stats.startTime;
    this.config.onStep?.(this.state);
  }

  /**
   * 运行动画循环
   */
  private async runAnimation(): Promise<void> {
    while (this.isPlaying && !this.isPaused) {
      const hasNext = this.step();

      if (!hasNext) {
        break;
      }

      // 根据速度计算延迟
      const delay = 100 / this.config.speed;
      await this.sleep(delay);
    }
  }

  /**
   * 完成回调
   */
  private onComplete(): void {
    this.isPlaying = false;
    this.state.comparing = [];
    this.state.swapping = [];
    this.state.pivot = null;
    // 标记所有元素为已排序
    this.state.sorted = this.state.array.map((_, i) => i);
    this.config.onStep?.(this.state);
    this.config.onComplete?.(this.state.stats);
  }

  /**
   * 延迟
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// 数据生成器
// ============================================================================

export type DataDistribution = 'random' | 'sorted' | 'reversed' | 'duplicates' | 'nearlySorted';

/**
 * 生成测试数据
 */
export function generateArray(size: number, distribution: DataDistribution): number[] {
  const array: number[] = [];

  switch (distribution) {
    case 'random':
      for (let i = 0; i < size; i++) {
        array.push(Math.floor(Math.random() * size) + 1);
      }
      break;

    case 'sorted':
      for (let i = 0; i < size; i++) {
        array.push(i + 1);
      }
      break;

    case 'reversed':
      for (let i = size; i > 0; i--) {
        array.push(i);
      }
      break;

    case 'duplicates':
      const values = [1, 2, 3, 4, 5];
      for (let i = 0; i < size; i++) {
        array.push(values[Math.floor(Math.random() * values.length)] * (size / 5));
      }
      break;

    case 'nearlySorted':
      for (let i = 0; i < size; i++) {
        array.push(i + 1);
      }
      // 随机交换 10% 的元素
      const swaps = Math.floor(size * 0.1);
      for (let i = 0; i < swaps; i++) {
        const a = Math.floor(Math.random() * size);
        const b = Math.floor(Math.random() * size);
        [array[a], array[b]] = [array[b], array[a]];
      }
      break;
  }

  return array;
}

