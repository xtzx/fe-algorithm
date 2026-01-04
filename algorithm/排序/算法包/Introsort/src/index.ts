/**
 * Introsort (内省排序)
 *
 * 核心思想：快排 + 堆排 + 插入排序的混合
 * - 默认使用快排（平均最快）
 * - 递归深度超过阈值时切换堆排（避免 O(n²)）
 * - 小数组使用插入排序（减少递归开销）
 *
 * 时间复杂度：O(n log n) 所有情况
 * 空间复杂度：O(log n) 递归栈
 * 稳定性：❌ 不稳定
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 常量
// ============================================================================

const INSERTION_THRESHOLD = 16;

// ============================================================================
// 核心实现
// ============================================================================

/**
 * Introsort（不修改原数组）
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  sortInPlace(result, cmp);
  return result;
}

/**
 * Introsort（原地排序）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  if (arr.length <= 1) return arr;

  // 计算最大递归深度：2 * floor(log2(n))
  const maxDepth = 2 * Math.floor(Math.log2(arr.length));
  introsortLoop(arr, 0, arr.length - 1, maxDepth, cmp);

  return arr;
}

/**
 * Introsort 核心循环
 */
function introsortLoop<T>(
  arr: T[],
  left: number,
  right: number,
  depthLimit: number,
  cmp: Comparator<T>
): void {
  const size = right - left + 1;

  // 1. 小数组：插入排序
  if (size <= INSERTION_THRESHOLD) {
    insertionSortRange(arr, left, right, cmp);
    return;
  }

  // 2. 深度超限：堆排序兜底
  if (depthLimit === 0) {
    heapSortRange(arr, left, right, cmp);
    return;
  }

  // 3. 正常情况：快排
  const pivotIndex = partitionMedianOfThree(arr, left, right, cmp);

  // 递归处理两边，深度减 1
  introsortLoop(arr, left, pivotIndex - 1, depthLimit - 1, cmp);
  introsortLoop(arr, pivotIndex + 1, right, depthLimit - 1, cmp);
}

// ============================================================================
// 辅助算法
// ============================================================================

/**
 * 插入排序（范围版）
 */
function insertionSortRange<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): void {
  for (let i = left + 1; i <= right; i++) {
    const current = arr[i];
    let j = i - 1;

    while (j >= left && cmp(arr[j], current) > 0) {
      arr[j + 1] = arr[j];
      j--;
    }

    arr[j + 1] = current;
  }
}

/**
 * 堆排序（范围版）
 */
function heapSortRange<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): void {
  const n = right - left + 1;

  // 建堆
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    heapifyRange(arr, left, n, i, cmp);
  }

  // 排序
  for (let i = n - 1; i > 0; i--) {
    [arr[left], arr[left + i]] = [arr[left + i], arr[left]];
    heapifyRange(arr, left, i, 0, cmp);
  }
}

function heapifyRange<T>(
  arr: T[],
  offset: number,
  heapSize: number,
  i: number,
  cmp: Comparator<T>
): void {
  while (true) {
    let largest = i;
    const leftChild = 2 * i + 1;
    const rightChild = 2 * i + 2;

    if (leftChild < heapSize && cmp(arr[offset + leftChild], arr[offset + largest]) > 0) {
      largest = leftChild;
    }

    if (rightChild < heapSize && cmp(arr[offset + rightChild], arr[offset + largest]) > 0) {
      largest = rightChild;
    }

    if (largest === i) break;

    [arr[offset + i], arr[offset + largest]] = [arr[offset + largest], arr[offset + i]];
    i = largest;
  }
}

/**
 * 三数取中分区
 */
function partitionMedianOfThree<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): number {
  const mid = (left + right) >> 1;

  // 排序 left, mid, right
  if (cmp(arr[left], arr[mid]) > 0) {
    [arr[left], arr[mid]] = [arr[mid], arr[left]];
  }
  if (cmp(arr[left], arr[right]) > 0) {
    [arr[left], arr[right]] = [arr[right], arr[left]];
  }
  if (cmp(arr[mid], arr[right]) > 0) {
    [arr[mid], arr[right]] = [arr[right], arr[mid]];
  }

  // 将中值移到 right-1 位置作为 pivot
  [arr[mid], arr[right - 1]] = [arr[right - 1], arr[mid]];

  const pivot = arr[right - 1];
  let i = left;
  let j = right - 1;

  while (true) {
    while (cmp(arr[++i], pivot) < 0) {}
    while (j > left && cmp(arr[--j], pivot) > 0) {}

    if (i >= j) break;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }

  // 将 pivot 放到最终位置
  [arr[i], arr[right - 1]] = [arr[right - 1], arr[i]];

  return i;
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
  insertionCalls: number;
  heapCalls: number;
  quickCalls: number;
} {
  const result = [...arr];
  const stats = {
    comparisons: 0,
    swaps: 0,
    insertionCalls: 0,
    heapCalls: 0,
    quickCalls: 0,
  };

  const maxDepth = 2 * Math.floor(Math.log2(result.length));

  function sortHelper(left: number, right: number, depth: number): void {
    const size = right - left + 1;

    if (size <= INSERTION_THRESHOLD) {
      stats.insertionCalls++;
      // 简化：直接用插入排序
      for (let i = left + 1; i <= right; i++) {
        const current = result[i];
        let j = i - 1;
        while (j >= left) {
          stats.comparisons++;
          if (cmp(result[j], current) > 0) {
            result[j + 1] = result[j];
            stats.swaps++;
            j--;
          } else {
            break;
          }
        }
        result[j + 1] = current;
      }
      return;
    }

    if (depth === 0) {
      stats.heapCalls++;
      heapSortRange(result, left, right, cmp);
      return;
    }

    stats.quickCalls++;
    const pivotIndex = partitionMedianOfThree(result, left, right, cmp);
    sortHelper(left, pivotIndex - 1, depth - 1);
    sortHelper(pivotIndex + 1, right, depth - 1);
  }

  if (result.length > 1) {
    sortHelper(0, result.length - 1, maxDepth);
  }

  return { result, ...stats };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: 'Introsort',
  englishName: 'Introspective Sort',
  stable: false,
  inPlace: true,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n log n)',
    worst: 'O(n log n)', // ⭐ 堆排兜底
  },
  spaceComplexity: 'O(log n)',
  组成: ['快速排序', '堆排序', '插入排序'],
  阈值: {
    insertionThreshold: INSERTION_THRESHOLD,
    depthLimit: '2 * floor(log₂n)',
  },
  适用场景: ['通用排序', '不可信输入', '需要保证最坏情况'],
  不适用场景: ['需要稳定排序'],
  使用者: ['C++ std::sort', '.NET Array.Sort'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortWithStats,
  meta,
};

