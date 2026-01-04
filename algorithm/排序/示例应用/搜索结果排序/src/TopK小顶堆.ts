/**
 * TopK 小顶堆实现
 *
 * 用于高效获取最大的 K 个元素
 */

import type { Comparator } from '../../../算法包/公共库/src/比较器';

// ============================================================================
// 最小堆实现
// ============================================================================

/**
 * 最小堆（用于 TopK 大值问题）
 *
 * 维护 K 个最大元素：
 * - 堆顶是 K 个中最小的
 * - 新元素比堆顶大时，替换堆顶
 */
export class MinHeap<T> {
  private heap: T[] = [];
  private cmp: Comparator<T>;

  constructor(cmp: Comparator<T>) {
    this.cmp = cmp;
  }

  /**
   * 堆大小
   */
  size(): number {
    return this.heap.length;
  }

  /**
   * 是否为空
   */
  isEmpty(): boolean {
    return this.heap.length === 0;
  }

  /**
   * 查看堆顶（最小值）
   */
  peek(): T | undefined {
    return this.heap[0];
  }

  /**
   * 入堆
   */
  push(item: T): void {
    this.heap.push(item);
    this.siftUp(this.heap.length - 1);
  }

  /**
   * 出堆（弹出最小值）
   */
  pop(): T | undefined {
    if (this.heap.length === 0) return undefined;

    const top = this.heap[0];
    const last = this.heap.pop()!;

    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.siftDown(0);
    }

    return top;
  }

  /**
   * 替换堆顶（比 pop + push 更高效）
   */
  replaceTop(item: T): T {
    const top = this.heap[0];
    this.heap[0] = item;
    this.siftDown(0);
    return top;
  }

  /**
   * 转为有序数组
   */
  toSortedArray(): T[] {
    return [...this.heap].sort(this.cmp);
  }

  /**
   * 转为数组（无序）
   */
  toArray(): T[] {
    return [...this.heap];
  }

  // 上浮
  private siftUp(i: number): void {
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (this.cmp(this.heap[i], this.heap[parent]) >= 0) break;
      [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
      i = parent;
    }
  }

  // 下沉
  private siftDown(i: number): void {
    const n = this.heap.length;

    while (true) {
      let smallest = i;
      const left = 2 * i + 1;
      const right = 2 * i + 2;

      if (left < n && this.cmp(this.heap[left], this.heap[smallest]) < 0) {
        smallest = left;
      }
      if (right < n && this.cmp(this.heap[right], this.heap[smallest]) < 0) {
        smallest = right;
      }

      if (smallest === i) break;

      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }
}

// ============================================================================
// TopK 函数
// ============================================================================

/**
 * 使用堆获取 TopK（最大的 K 个）
 *
 * @param arr 数据数组
 * @param k 需要的数量
 * @param cmp 比较函数（升序）
 * @returns TopK 结果（已排序，最大的在前）
 */
export function topKByHeap<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  if (k <= 0) return [];
  if (arr.length <= k) return [...arr].sort((a, b) => -cmp(a, b));

  // 小顶堆维护最大的 k 个
  const heap = new MinHeap<T>(cmp);

  for (const item of arr) {
    if (heap.size() < k) {
      heap.push(item);
    } else if (cmp(item, heap.peek()!) > 0) {
      // 新元素比堆顶大，替换
      heap.replaceTop(item);
    }
  }

  // 返回降序排列
  return heap.toSortedArray().reverse();
}

/**
 * 使用堆获取 BottomK（最小的 K 个）
 */
export function bottomKByHeap<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  // 反转比较器
  return topKByHeap(arr, k, (a, b) => -cmp(a, b)).reverse();
}

/**
 * 全量排序获取 TopK（对比用）
 */
export function topKBySort<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  return [...arr].sort((a, b) => -cmp(a, b)).slice(0, k);
}

// ============================================================================
// 带过滤的 TopK
// ============================================================================

/**
 * 带过滤条件的 TopK
 */
export function topKWithFilter<T>(
  arr: readonly T[],
  k: number,
  filter: (item: T) => boolean,
  cmp: Comparator<T>
): T[] {
  if (k <= 0) return [];

  const heap = new MinHeap<T>(cmp);

  for (const item of arr) {
    if (!filter(item)) continue;

    if (heap.size() < k) {
      heap.push(item);
    } else if (cmp(item, heap.peek()!) > 0) {
      heap.replaceTop(item);
    }
  }

  return heap.toSortedArray().reverse();
}

/**
 * 多维度 TopK
 *
 * 支持多个排序维度
 */
export function topKMultiDimension<T>(
  arr: readonly T[],
  k: number,
  dimensions: Array<{
    getValue: (item: T) => number;
    weight: number;
    order: 'asc' | 'desc';
  }>
): T[] {
  // 计算综合分数
  const getScore = (item: T): number => {
    return dimensions.reduce((score, dim) => {
      const value = dim.getValue(item);
      const normalizedValue = dim.order === 'desc' ? value : -value;
      return score + normalizedValue * dim.weight;
    }, 0);
  };

  const cmp: Comparator<T> = (a, b) => getScore(a) - getScore(b);

  return topKByHeap(arr, k, cmp);
}

// ============================================================================
// 导出
// ============================================================================

export default {
  MinHeap,
  topKByHeap,
  bottomKByHeap,
  topKBySort,
  topKWithFilter,
  topKMultiDimension,
};

