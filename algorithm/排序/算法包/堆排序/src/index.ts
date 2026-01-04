/**
 * 堆排序 (Heap Sort)
 *
 * 核心思想：利用堆的性质，反复取出堆顶放到末尾
 *
 * 时间复杂度：O(n log n) 所有情况
 * 空间复杂度：O(1) 原地
 * 稳定性：❌ 不稳定
 *
 * 优势：时间稳定，空间最优，不会退化
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 堆排序（不修改原数组）
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @returns 排序后的新数组
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  sortInPlace(result, cmp);
  return result;
}

/**
 * 堆排序（原地排序）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  const n = arr.length;
  if (n <= 1) return arr;

  // 1. 建立最大堆（O(n)）
  buildMaxHeap(arr, cmp);

  // 2. 排序：交换堆顶与末尾，缩小堆，堆化
  for (let i = n - 1; i > 0; i--) {
    // 堆顶（最大值）与当前末尾交换
    [arr[0], arr[i]] = [arr[i], arr[0]];
    // 堆大小减 1，对新堆顶进行堆化
    heapifyDown(arr, i, 0, cmp);
  }

  return arr;
}

// ============================================================================
// 堆操作
// ============================================================================

/**
 * 建立最大堆（自底向上，O(n)）
 *
 * 从最后一个非叶子节点开始，逐个向下堆化
 */
export function buildMaxHeap<T>(arr: T[], cmp: Comparator<T>): void {
  const n = arr.length;
  // 最后一个非叶子节点的索引
  const lastNonLeaf = (n >> 1) - 1;

  for (let i = lastNonLeaf; i >= 0; i--) {
    heapifyDown(arr, n, i, cmp);
  }
}

/**
 * 建立最大堆（自顶向下，O(n log n)）
 *
 * 模拟逐个插入元素
 */
export function buildMaxHeapTopDown<T>(arr: T[], cmp: Comparator<T>): void {
  const n = arr.length;
  for (let i = 1; i < n; i++) {
    heapifyUp(arr, i, cmp);
  }
}

/**
 * 向下堆化（sift down）
 *
 * 将节点 i 下沉到正确位置，维护最大堆性质
 *
 * @param arr 堆数组
 * @param heapSize 堆的大小
 * @param i 要堆化的节点索引
 * @param cmp 比较函数
 */
export function heapifyDown<T>(
  arr: T[],
  heapSize: number,
  i: number,
  cmp: Comparator<T>
): void {
  while (true) {
    let largest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;

    // 比较左子节点
    if (left < heapSize && cmp(arr[left], arr[largest]) > 0) {
      largest = left;
    }

    // 比较右子节点
    if (right < heapSize && cmp(arr[right], arr[largest]) > 0) {
      largest = right;
    }

    // 如果最大的是当前节点，堆化完成
    if (largest === i) break;

    // 交换并继续下沉
    [arr[i], arr[largest]] = [arr[largest], arr[i]];
    i = largest;
  }
}

/**
 * 向上堆化（sift up）
 *
 * 将节点 i 上浮到正确位置
 */
export function heapifyUp<T>(arr: T[], i: number, cmp: Comparator<T>): void {
  while (i > 0) {
    const parent = (i - 1) >> 1;

    if (cmp(arr[i], arr[parent]) <= 0) break;

    [arr[i], arr[parent]] = [arr[parent], arr[i]];
    i = parent;
  }
}

// ============================================================================
// TopK 相关
// ============================================================================

/**
 * 找出最大的 K 个元素
 *
 * 使用最小堆，时间 O(n log k)，空间 O(k)
 */
export function findTopKLargest<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  if (k <= 0) return [];
  if (k >= arr.length) return sort(arr, cmp).reverse();

  // 最小堆比较器（反转）
  const minCmp: Comparator<T> = (a, b) => cmp(b, a);

  // 维护大小为 k 的最小堆
  const heap = arr.slice(0, k);
  buildMinHeap(heap, cmp);

  for (let i = k; i < arr.length; i++) {
    // 如果当前元素比堆顶大，替换堆顶
    if (cmp(arr[i], heap[0]) > 0) {
      heap[0] = arr[i];
      heapifyDownMin(heap, k, 0, cmp);
    }
  }

  return heap;
}

/**
 * 找出最小的 K 个元素
 */
export function findTopKSmallest<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  // 反转比较器
  const reverseCmp: Comparator<T> = (a, b) => cmp(b, a);
  return findTopKLargest(arr, k, reverseCmp);
}

function buildMinHeap<T>(arr: T[], cmp: Comparator<T>): void {
  const n = arr.length;
  for (let i = (n >> 1) - 1; i >= 0; i--) {
    heapifyDownMin(arr, n, i, cmp);
  }
}

function heapifyDownMin<T>(
  arr: T[],
  heapSize: number,
  i: number,
  cmp: Comparator<T>
): void {
  while (true) {
    let smallest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;

    if (left < heapSize && cmp(arr[left], arr[smallest]) < 0) {
      smallest = left;
    }
    if (right < heapSize && cmp(arr[right], arr[smallest]) < 0) {
      smallest = right;
    }

    if (smallest === i) break;

    [arr[i], arr[smallest]] = [arr[smallest], arr[i]];
    i = smallest;
  }
}

// ============================================================================
// 优先队列实现
// ============================================================================

/**
 * 最大优先队列
 */
export class MaxPriorityQueue<T> {
  private heap: T[] = [];

  constructor(private cmp: Comparator<T>) {}

  get size(): number {
    return this.heap.length;
  }

  isEmpty(): boolean {
    return this.heap.length === 0;
  }

  peek(): T | undefined {
    return this.heap[0];
  }

  push(item: T): void {
    this.heap.push(item);
    heapifyUp(this.heap, this.heap.length - 1, this.cmp);
  }

  pop(): T | undefined {
    if (this.heap.length === 0) return undefined;

    const top = this.heap[0];
    const last = this.heap.pop()!;

    if (this.heap.length > 0) {
      this.heap[0] = last;
      heapifyDown(this.heap, this.heap.length, 0, this.cmp);
    }

    return top;
  }

  toArray(): T[] {
    return [...this.heap];
  }
}

/**
 * 最小优先队列
 */
export class MinPriorityQueue<T> {
  private maxQueue: MaxPriorityQueue<T>;

  constructor(cmp: Comparator<T>) {
    // 反转比较器
    this.maxQueue = new MaxPriorityQueue((a, b) => cmp(b, a));
  }

  get size(): number {
    return this.maxQueue.size;
  }

  isEmpty(): boolean {
    return this.maxQueue.isEmpty();
  }

  peek(): T | undefined {
    return this.maxQueue.peek();
  }

  push(item: T): void {
    this.maxQueue.push(item);
  }

  pop(): T | undefined {
    return this.maxQueue.pop();
  }

  toArray(): T[] {
    return this.maxQueue.toArray();
  }
}

// ============================================================================
// 带统计版本
// ============================================================================

export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): {
  result: T[];
  comparisons: number;
  swaps: number;
} {
  const result = [...arr];
  const n = result.length;
  let comparisons = 0;
  let swaps = 0;

  // 建堆
  for (let i = (n >> 1) - 1; i >= 0; i--) {
    heapifyWithStats(result, n, i);
  }

  // 排序
  for (let i = n - 1; i > 0; i--) {
    [result[0], result[i]] = [result[i], result[0]];
    swaps++;
    heapifyWithStats(result, i, 0);
  }

  function heapifyWithStats(arr: T[], size: number, idx: number): void {
    let i = idx;
    while (true) {
      let largest = i;
      const left = 2 * i + 1;
      const right = 2 * i + 2;

      if (left < size) {
        comparisons++;
        if (cmp(arr[left], arr[largest]) > 0) {
          largest = left;
        }
      }

      if (right < size) {
        comparisons++;
        if (cmp(arr[right], arr[largest]) > 0) {
          largest = right;
        }
      }

      if (largest === i) break;

      [arr[i], arr[largest]] = [arr[largest], arr[i]];
      swaps++;
      i = largest;
    }
  }

  return { result, comparisons, swaps };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '堆排序',
  englishName: 'Heap Sort',
  stable: false,
  inPlace: true,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n log n)',
    worst: 'O(n log n)', // ⭐ 不会退化
  },
  spaceComplexity: 'O(1)',
  适用场景: ['内存受限', '最坏情况敏感', 'TopK 问题', '优先队列'],
  不适用场景: ['需要稳定排序', '追求最快平均速度', '缓存敏感场景'],
  特点: ['时间稳定', '空间最优', '缓存不友好'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortWithStats,
  buildMaxHeap,
  buildMaxHeapTopDown,
  heapifyDown,
  heapifyUp,
  findTopKLargest,
  findTopKSmallest,
  MaxPriorityQueue,
  MinPriorityQueue,
  meta,
};

