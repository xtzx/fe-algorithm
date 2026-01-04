/**
 * 快速排序 (Quick Sort)
 *
 * 核心思想：分区征服 - 选 pivot，分区，递归
 *
 * 时间复杂度：O(n log n) 平均，O(n²) 最坏
 * 空间复杂度：O(log n) 栈空间
 * 稳定性：❌ 不稳定
 *
 * 优势：平均性能最优，原地排序，缓存友好
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现 - 基础版
// ============================================================================

/**
 * 快速排序（不修改原数组）
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @returns 排序后的新数组
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  quickSortHelper(result, 0, result.length - 1, cmp);
  return result;
}

/**
 * 快速排序（原地排序）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  quickSortHelper(arr, 0, arr.length - 1, cmp);
  return arr;
}

function quickSortHelper<T>(
  arr: T[],
  low: number,
  high: number,
  cmp: Comparator<T>
): void {
  if (low >= high) return;

  const pivotIndex = partition(arr, low, high, cmp);
  quickSortHelper(arr, low, pivotIndex - 1, cmp);
  quickSortHelper(arr, pivotIndex + 1, high, cmp);
}

// ============================================================================
// 分区实现
// ============================================================================

/**
 * Lomuto 分区方案
 *
 * 选最后一个元素作为 pivot
 * 返回 pivot 的最终位置
 */
export function partition<T>(
  arr: T[],
  low: number,
  high: number,
  cmp: Comparator<T>
): number {
  const pivot = arr[high];
  let i = low - 1;

  for (let j = low; j < high; j++) {
    if (cmp(arr[j], pivot) < 0) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
  return i + 1;
}

/**
 * 随机 pivot 分区
 */
export function partitionRandom<T>(
  arr: T[],
  low: number,
  high: number,
  cmp: Comparator<T>
): number {
  const randomIndex = low + Math.floor(Math.random() * (high - low + 1));
  [arr[randomIndex], arr[high]] = [arr[high], arr[randomIndex]];
  return partition(arr, low, high, cmp);
}

/**
 * 三数取中分区
 *
 * 从 low, mid, high 三个位置选中值作为 pivot
 * 有效避免有序/逆序数据的最坏情况
 */
export function partitionMedianOfThree<T>(
  arr: T[],
  low: number,
  high: number,
  cmp: Comparator<T>
): number {
  const mid = (low + high) >> 1;

  // 排序 low, mid, high
  if (cmp(arr[low], arr[mid]) > 0) {
    [arr[low], arr[mid]] = [arr[mid], arr[low]];
  }
  if (cmp(arr[low], arr[high]) > 0) {
    [arr[low], arr[high]] = [arr[high], arr[low]];
  }
  if (cmp(arr[mid], arr[high]) > 0) {
    [arr[mid], arr[high]] = [arr[high], arr[mid]];
  }

  // 将中值移到 high-1 位置
  [arr[mid], arr[high - 1]] = [arr[high - 1], arr[mid]];

  // 以 high-1 为 pivot 进行分区
  const pivot = arr[high - 1];
  let i = low;
  let j = high - 1;

  while (true) {
    while (cmp(arr[++i], pivot) < 0) {}
    while (cmp(arr[--j], pivot) > 0) {}

    if (i >= j) break;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }

  [arr[i], arr[high - 1]] = [arr[high - 1], arr[i]];
  return i;
}

/**
 * Hoare 分区方案
 *
 * 双指针从两端向中间扫描
 * 交换次数比 Lomuto 少
 */
export function partitionHoare<T>(
  arr: T[],
  low: number,
  high: number,
  cmp: Comparator<T>
): number {
  const pivot = arr[(low + high) >> 1];
  let i = low - 1;
  let j = high + 1;

  while (true) {
    do {
      i++;
    } while (cmp(arr[i], pivot) < 0);

    do {
      j--;
    } while (cmp(arr[j], pivot) > 0);

    if (i >= j) return j;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ============================================================================
// 优化版本
// ============================================================================

/**
 * 健壮版快速排序
 *
 * 优化点：
 * 1. 三数取中选 pivot
 * 2. 尾递归优化（先递归较小分区）
 * 3. 小数组使用插入排序
 */
export function sortRobust<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  robustQuickSort(result, 0, result.length - 1, cmp);
  return result;
}

const INSERTION_THRESHOLD = 16;

function robustQuickSort<T>(
  arr: T[],
  low: number,
  high: number,
  cmp: Comparator<T>
): void {
  while (low < high) {
    // 优化 1：小数组使用插入排序
    if (high - low < INSERTION_THRESHOLD) {
      insertionSortRange(arr, low, high, cmp);
      return;
    }

    // 优化 2：三数取中
    const pivotIndex = partitionMedianOfThree(arr, low, high, cmp);

    // 优化 3：尾递归优化 - 先递归较小的分区
    if (pivotIndex - low < high - pivotIndex) {
      robustQuickSort(arr, low, pivotIndex - 1, cmp);
      low = pivotIndex + 1; // 循环处理较大分区
    } else {
      robustQuickSort(arr, pivotIndex + 1, high, cmp);
      high = pivotIndex - 1;
    }
  }
}

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

// ============================================================================
// 带统计版本
// ============================================================================

/**
 * 带统计的快速排序
 */
export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): {
  result: T[];
  comparisons: number;
  swaps: number;
  recursionDepth: number;
} {
  const result = [...arr];
  const stats = { comparisons: 0, swaps: 0, recursionDepth: 0, maxDepth: 0 };

  function sortHelper(low: number, high: number, depth: number): void {
    stats.maxDepth = Math.max(stats.maxDepth, depth);

    if (low >= high) return;

    // 分区
    const pivot = result[high];
    let i = low - 1;

    for (let j = low; j < high; j++) {
      stats.comparisons++;
      if (cmp(result[j], pivot) < 0) {
        i++;
        [result[i], result[j]] = [result[j], result[i]];
        stats.swaps++;
      }
    }

    [result[i + 1], result[high]] = [result[high], result[i + 1]];
    stats.swaps++;

    const pivotIndex = i + 1;
    sortHelper(low, pivotIndex - 1, depth + 1);
    sortHelper(pivotIndex + 1, high, depth + 1);
  }

  sortHelper(0, result.length - 1, 1);

  return {
    result,
    comparisons: stats.comparisons,
    swaps: stats.swaps,
    recursionDepth: stats.maxDepth,
  };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '快速排序',
  englishName: 'Quick Sort',
  stable: false,
  inPlace: true,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n log n)',
    worst: 'O(n²)', // ⚠️
  },
  spaceComplexity: 'O(log n)', // 递归栈
  适用场景: ['通用内部排序', '内存受限', '缓存敏感场景'],
  不适用场景: ['需要稳定排序', '数据已有序/逆序', '大量重复元素'],
  pivot策略: ['首/尾元素', '随机选择', '三数取中'],
  优化技巧: ['尾递归优化', '小数组插入排序', '三路快排（重复多）'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortRobust,
  sortWithStats,
  partition,
  partitionRandom,
  partitionMedianOfThree,
  partitionHoare,
  meta,
};

