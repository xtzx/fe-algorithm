/**
 * 外部归并排序 (External Merge Sort)
 *
 * 用于处理无法一次装入内存的大数据集。
 * 这是内存模拟版本，演示核心思想。
 *
 * 核心步骤：
 * 1. 分块：将数据分成能装入内存的小块
 * 2. 内部排序：对每个小块进行排序
 * 3. K 路归并：使用最小堆合并所有有序块
 *
 * 时间复杂度：O(n log n)
 * 空间复杂度：O(n)（辅助数组）
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 类型定义
// ============================================================================

/** 外部排序配置 */
export interface ExternalSortOptions<T> {
  /** 每块大小 */
  chunkSize: number;
  /** 块内排序函数 */
  sortChunk: (chunk: T[], cmp: Comparator<T>) => T[];
  /** 多路归并函数 */
  mergeChunks: (chunks: T[][], cmp: Comparator<T>) => T[];
}

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 外部归并排序（内存模拟版）
 *
 * @param arr 待排序数组
 * @param chunkSize 每块大小（模拟内存限制）
 * @param cmp 比较函数
 * @returns 排序后的新数组
 *
 * @invariant 排序前后元素相同（置换性）
 * @invariant 返回数组按升序排列
 */
export function externalMergeSort<T>(
  arr: readonly T[],
  chunkSize: number,
  cmp: Comparator<T>
): T[] {
  const n = arr.length;

  // 能一次装入内存，直接排序
  if (n <= chunkSize) {
    return [...arr].sort(cmp);
  }

  // 1. 分块并排序
  const sortedChunks: T[][] = [];
  for (let i = 0; i < n; i += chunkSize) {
    const chunk = arr.slice(i, Math.min(i + chunkSize, n));
    sortedChunks.push([...chunk].sort(cmp));
  }

  // 2. K 路归并
  return kWayMerge(sortedChunks, cmp);
}

/**
 * K 路归并（使用最小堆）
 *
 * @param chunks 多个已排序的数组
 * @param cmp 比较函数
 * @returns 合并后的有序数组
 */
export function kWayMerge<T>(
  chunks: readonly (readonly T[])[],
  cmp: Comparator<T>
): T[] {
  const k = chunks.length;
  if (k === 0) return [];
  if (k === 1) return [...chunks[0]];

  // 堆元素类型
  interface HeapItem {
    value: T;
    chunkIdx: number;
    itemIdx: number;
  }

  // 初始化堆（取每个块的第一个元素）
  const heap: HeapItem[] = [];

  for (let i = 0; i < k; i++) {
    if (chunks[i].length > 0) {
      heap.push({
        value: chunks[i][0],
        chunkIdx: i,
        itemIdx: 0,
      });
    }
  }

  const heapCmp: Comparator<HeapItem> = (a, b) => cmp(a.value, b.value);

  // 建最小堆
  buildMinHeap(heap, heapCmp);

  const result: T[] = [];

  while (heap.length > 0) {
    // 取出堆顶（最小值）
    const min = heap[0];
    result.push(min.value);

    // 从同一块取下一个元素
    const nextIdx = min.itemIdx + 1;
    if (nextIdx < chunks[min.chunkIdx].length) {
      // 替换堆顶
      heap[0] = {
        value: chunks[min.chunkIdx][nextIdx],
        chunkIdx: min.chunkIdx,
        itemIdx: nextIdx,
      };
      heapifyDown(heap, 0, heapCmp);
    } else {
      // 该块已空，用最后一个元素替换堆顶
      heap[0] = heap[heap.length - 1];
      heap.pop();
      if (heap.length > 0) {
        heapifyDown(heap, 0, heapCmp);
      }
    }
  }

  return result;
}

// ============================================================================
// 最小堆辅助函数
// ============================================================================

function buildMinHeap<T>(heap: T[], cmp: Comparator<T>): void {
  for (let i = Math.floor(heap.length / 2) - 1; i >= 0; i--) {
    heapifyDown(heap, i, cmp);
  }
}

function heapifyDown<T>(heap: T[], i: number, cmp: Comparator<T>): void {
  const n = heap.length;

  while (true) {
    let smallest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;

    if (left < n && cmp(heap[left], heap[smallest]) < 0) {
      smallest = left;
    }
    if (right < n && cmp(heap[right], heap[smallest]) < 0) {
      smallest = right;
    }

    if (smallest === i) break;

    [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
    i = smallest;
  }
}

// ============================================================================
// 可配置版本
// ============================================================================

/**
 * 创建可配置的外部排序函数
 */
export function createExternalSorter<T>(
  options: ExternalSortOptions<T>
): (arr: readonly T[], cmp: Comparator<T>) => T[] {
  const { chunkSize, sortChunk, mergeChunks } = options;

  return (arr: readonly T[], cmp: Comparator<T>): T[] => {
    const n = arr.length;

    if (n <= chunkSize) {
      return sortChunk([...arr], cmp);
    }

    // 分块
    const chunks: T[][] = [];
    for (let i = 0; i < n; i += chunkSize) {
      chunks.push(arr.slice(i, Math.min(i + chunkSize, n)));
    }

    // 排序每个块
    const sortedChunks = chunks.map(chunk => sortChunk([...chunk], cmp));

    // 归并
    return mergeChunks(sortedChunks, cmp);
  };
}

// ============================================================================
// 流式处理器
// ============================================================================

/**
 * 流式排序器
 *
 * 用于处理流式到达的数据
 */
export class StreamSorter<T> {
  private chunks: T[][] = [];
  private buffer: T[] = [];
  private chunkSize: number;
  private cmp: Comparator<T>;

  constructor(chunkSize: number, cmp: Comparator<T>) {
    this.chunkSize = chunkSize;
    this.cmp = cmp;
  }

  /**
   * 添加单个元素
   */
  add(item: T): void {
    this.buffer.push(item);
    if (this.buffer.length >= this.chunkSize) {
      this.flush();
    }
  }

  /**
   * 批量添加元素
   */
  addBatch(items: T[]): void {
    for (const item of items) {
      this.add(item);
    }
  }

  /**
   * 将缓冲区刷入已排序的块
   */
  private flush(): void {
    if (this.buffer.length > 0) {
      const sorted = [...this.buffer].sort(this.cmp);
      this.chunks.push(sorted);
      this.buffer = [];
    }
  }

  /**
   * 获取最终排序结果
   */
  getResult(): T[] {
    this.flush();
    return kWayMerge(this.chunks, this.cmp);
  }

  /**
   * 重置状态
   */
  reset(): void {
    this.chunks = [];
    this.buffer = [];
  }

  /**
   * 获取当前块数量
   */
  getChunkCount(): number {
    return this.chunks.length + (this.buffer.length > 0 ? 1 : 0);
  }
}

// ============================================================================
// 两路归并（作为参考）
// ============================================================================

/**
 * 两路归并
 *
 * 合并两个有序数组
 */
export function twoWayMerge<T>(
  arr1: readonly T[],
  arr2: readonly T[],
  cmp: Comparator<T>
): T[] {
  const result: T[] = [];
  let i = 0;
  let j = 0;

  while (i < arr1.length && j < arr2.length) {
    if (cmp(arr1[i], arr2[j]) <= 0) {
      result.push(arr1[i]);
      i++;
    } else {
      result.push(arr2[j]);
      j++;
    }
  }

  // 处理剩余元素
  while (i < arr1.length) {
    result.push(arr1[i]);
    i++;
  }
  while (j < arr2.length) {
    result.push(arr2[j]);
    j++;
  }

  return result;
}

/**
 * 自底向上的迭代归并
 *
 * 使用两路归并逐层合并所有块
 */
export function iterativeMerge<T>(
  chunks: T[][],
  cmp: Comparator<T>
): T[] {
  if (chunks.length === 0) return [];

  let current = chunks;

  while (current.length > 1) {
    const next: T[][] = [];

    for (let i = 0; i < current.length; i += 2) {
      if (i + 1 < current.length) {
        next.push(twoWayMerge(current[i], current[i + 1], cmp));
      } else {
        next.push(current[i]);
      }
    }

    current = next;
  }

  return current[0];
}

// ============================================================================
// 元数据
// ============================================================================

export const meta = {
  name: '外部归并排序',
  stable: true, // 取决于子排序和归并实现
  inPlace: false,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n log n)',
    worst: 'O(n log n)',
  },
  spaceComplexity: 'O(n)',
  适用场景: [
    '数据太大无法一次装入内存',
    '流式/分页数据排序',
    '合并多个有序数据源',
    '分布式排序',
  ],
  不适用场景: [
    '小数据量（直接排序更简单）',
    '内存充足的场景',
  ],
  特点: [
    '可插拔的块内排序算法',
    '可插拔的归并策略',
    '支持流式处理',
  ],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  externalMergeSort,
  kWayMerge,
  twoWayMerge,
  iterativeMerge,
  createExternalSorter,
  StreamSorter,
  meta,
};

