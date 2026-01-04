/**
 * 增量更新排序
 *
 * 处理流式数据的 TopK 追踪
 */

import type { Comparator } from '../../../算法包/公共库/src/比较器';
import { MinHeap } from './TopK小顶堆';

// ============================================================================
// TopK 追踪器
// ============================================================================

/**
 * TopK 追踪器
 *
 * 用于流式数据的实时 TopK 维护
 */
export class TopKTracker<T> {
  private heap: MinHeap<T>;
  private k: number;
  private cmp: Comparator<T>;
  private totalAdded: number = 0;
  private enteredCount: number = 0;

  constructor(k: number, cmp: Comparator<T>) {
    this.k = k;
    this.cmp = cmp;
    this.heap = new MinHeap(cmp);
  }

  /**
   * 添加新元素
   *
   * @returns 是否进入 TopK
   */
  add(item: T): boolean {
    this.totalAdded++;

    if (this.heap.size() < this.k) {
      this.heap.push(item);
      this.enteredCount++;
      return true;
    }

    if (this.cmp(item, this.heap.peek()!) > 0) {
      this.heap.replaceTop(item);
      this.enteredCount++;
      return true;
    }

    return false;
  }

  /**
   * 批量添加
   *
   * @returns 进入 TopK 的数量
   */
  addBatch(items: T[]): number {
    let count = 0;
    for (const item of items) {
      if (this.add(item)) count++;
    }
    return count;
  }

  /**
   * 获取当前 TopK（已排序，最大在前）
   */
  getTopK(): T[] {
    return this.heap.toSortedArray().reverse();
  }

  /**
   * 获取门槛值（第 K 名的值）
   */
  getThreshold(): T | undefined {
    return this.heap.peek();
  }

  /**
   * 检查元素是否能进入 TopK
   */
  wouldEnter(item: T): boolean {
    if (this.heap.size() < this.k) return true;
    return this.cmp(item, this.heap.peek()!) > 0;
  }

  /**
   * 当前 TopK 的大小
   */
  size(): number {
    return this.heap.size();
  }

  /**
   * 统计信息
   */
  getStats(): {
    k: number;
    currentSize: number;
    totalAdded: number;
    enteredCount: number;
    entryRate: number;
  } {
    return {
      k: this.k,
      currentSize: this.heap.size(),
      totalAdded: this.totalAdded,
      enteredCount: this.enteredCount,
      entryRate: this.totalAdded > 0 ? this.enteredCount / this.totalAdded : 0,
    };
  }

  /**
   * 重置
   */
  reset(): void {
    this.heap = new MinHeap(this.cmp);
    this.totalAdded = 0;
    this.enteredCount = 0;
  }
}

// ============================================================================
// 滑动窗口 TopK
// ============================================================================

/**
 * 滑动窗口 TopK
 *
 * 维护最近 N 个元素中的 TopK
 */
export class SlidingWindowTopK<T> {
  private window: T[] = [];
  private windowSize: number;
  private k: number;
  private cmp: Comparator<T>;

  constructor(windowSize: number, k: number, cmp: Comparator<T>) {
    this.windowSize = windowSize;
    this.k = k;
    this.cmp = cmp;
  }

  /**
   * 添加新元素
   */
  add(item: T): void {
    this.window.push(item);

    // 滑动窗口：移除最旧的元素
    if (this.window.length > this.windowSize) {
      this.window.shift();
    }
  }

  /**
   * 获取当前窗口的 TopK
   */
  getTopK(): T[] {
    const heap = new MinHeap<T>(this.cmp);

    for (const item of this.window) {
      if (heap.size() < this.k) {
        heap.push(item);
      } else if (this.cmp(item, heap.peek()!) > 0) {
        heap.replaceTop(item);
      }
    }

    return heap.toSortedArray().reverse();
  }

  /**
   * 当前窗口大小
   */
  size(): number {
    return this.window.length;
  }
}

// ============================================================================
// 实时排行榜
// ============================================================================

/**
 * 实时排行榜
 *
 * 支持更新已有元素的分数
 */
export class Leaderboard<T> {
  private items: Map<string, T> = new Map();
  private k: number;
  private getKey: (item: T) => string;
  private cmp: Comparator<T>;

  constructor(
    k: number,
    getKey: (item: T) => string,
    cmp: Comparator<T>
  ) {
    this.k = k;
    this.getKey = getKey;
    this.cmp = cmp;
  }

  /**
   * 添加或更新
   */
  upsert(item: T): void {
    const key = this.getKey(item);
    this.items.set(key, item);
  }

  /**
   * 移除
   */
  remove(key: string): boolean {
    return this.items.delete(key);
  }

  /**
   * 获取排行榜
   */
  getLeaderboard(): T[] {
    const all = Array.from(this.items.values());
    return all
      .sort((a, b) => -this.cmp(a, b)) // 降序
      .slice(0, this.k);
  }

  /**
   * 获取排名
   */
  getRank(key: string): number | null {
    const item = this.items.get(key);
    if (!item) return null;

    const all = Array.from(this.items.values());
    const sorted = all.sort((a, b) => -this.cmp(a, b));

    return sorted.findIndex(i => this.getKey(i) === key) + 1;
  }

  /**
   * 元素总数
   */
  size(): number {
    return this.items.size;
  }
}

// ============================================================================
// 事件驱动的 TopK
// ============================================================================

export type TopKEvent<T> =
  | { type: 'entered'; item: T; rank: number }
  | { type: 'dropped'; item: T }
  | { type: 'unchanged' };

/**
 * 带事件通知的 TopK 追踪器
 */
export class TopKTrackerWithEvents<T> {
  private tracker: TopKTracker<T>;
  private listeners: Array<(event: TopKEvent<T>) => void> = [];

  constructor(k: number, cmp: Comparator<T>) {
    this.tracker = new TopKTracker(k, cmp);
  }

  /**
   * 添加事件监听器
   */
  on(listener: (event: TopKEvent<T>) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  /**
   * 添加元素
   */
  add(item: T): void {
    const entered = this.tracker.add(item);

    if (entered) {
      const topK = this.tracker.getTopK();
      const rank = topK.findIndex(i => i === item) + 1;
      this.emit({ type: 'entered', item, rank });
    } else {
      this.emit({ type: 'unchanged' });
    }
  }

  /**
   * 获取 TopK
   */
  getTopK(): T[] {
    return this.tracker.getTopK();
  }

  private emit(event: TopKEvent<T>): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}

// ============================================================================
// 导出
// ============================================================================

export default {
  TopKTracker,
  SlidingWindowTopK,
  Leaderboard,
  TopKTrackerWithEvents,
};

