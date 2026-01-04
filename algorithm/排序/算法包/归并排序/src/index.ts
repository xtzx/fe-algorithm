/**
 * 归并排序 (Merge Sort)
 *
 * 核心思想：分治 - 递归拆分，逐层合并
 *
 * 时间复杂度：O(n log n) 所有情况
 * 空间复杂度：O(n) 辅助数组
 * 稳定性：✅ 稳定
 *
 * 优势：性能稳定，适合链表排序和外部排序
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现 - 递归版（自顶向下）
// ============================================================================

/**
 * 归并排序（不修改原数组）
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @returns 排序后的新数组
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  if (arr.length <= 1) return [...arr];
  return mergeSort(arr as T[], 0, arr.length - 1, cmp);
}

/**
 * 归并排序（原地版本 - 使用辅助数组）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  if (arr.length <= 1) return arr;

  const aux = new Array(arr.length);
  mergeSortInPlace(arr, aux, 0, arr.length - 1, cmp);
  return arr;
}

function mergeSort<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): T[] {
  if (left >= right) {
    return [arr[left]];
  }

  const mid = (left + right) >> 1;
  const leftPart = mergeSort(arr, left, mid, cmp);
  const rightPart = mergeSort(arr, mid + 1, right, cmp);

  return merge(leftPart, rightPart, cmp);
}

function mergeSortInPlace<T>(
  arr: T[],
  aux: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): void {
  if (left >= right) return;

  const mid = (left + right) >> 1;
  mergeSortInPlace(arr, aux, left, mid, cmp);
  mergeSortInPlace(arr, aux, mid + 1, right, cmp);
  mergeInPlace(arr, aux, left, mid, right, cmp);
}

// ============================================================================
// 核心实现 - 迭代版（自底向上）
// ============================================================================

/**
 * 归并排序 - 迭代版（自底向上）
 *
 * 优势：无递归栈，适合超大数组
 */
export function sortIterative<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  const result = [...arr];
  const aux = new Array(n);

  // 从 size=1 开始，每次翻倍
  for (let size = 1; size < n; size *= 2) {
    // 遍历所有相邻的 size 块进行合并
    for (let left = 0; left < n - size; left += 2 * size) {
      const mid = left + size - 1;
      const right = Math.min(left + 2 * size - 1, n - 1);
      mergeInPlace(result, aux, left, mid, right, cmp);
    }
  }

  return result;
}

// ============================================================================
// 合并函数
// ============================================================================

/**
 * 合并两个有序数组
 *
 * @param left 左数组（已排序）
 * @param right 右数组（已排序）
 * @param cmp 比较函数
 * @returns 合并后的有序数组
 */
export function merge<T>(
  left: readonly T[],
  right: readonly T[],
  cmp: Comparator<T>
): T[] {
  const result: T[] = [];
  let i = 0;
  let j = 0;

  while (i < left.length && j < right.length) {
    // ⚠️ <= 保证稳定性：相等时取左边元素
    if (cmp(left[i], right[j]) <= 0) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }

  // 合并剩余元素
  while (i < left.length) result.push(left[i++]);
  while (j < right.length) result.push(right[j++]);

  return result;
}

/**
 * 原地合并（使用辅助数组）
 */
function mergeInPlace<T>(
  arr: T[],
  aux: T[],
  left: number,
  mid: number,
  right: number,
  cmp: Comparator<T>
): void {
  // 复制到辅助数组
  for (let k = left; k <= right; k++) {
    aux[k] = arr[k];
  }

  let i = left;
  let j = mid + 1;

  for (let k = left; k <= right; k++) {
    if (i > mid) {
      // 左边用完，取右边
      arr[k] = aux[j++];
    } else if (j > right) {
      // 右边用完，取左边
      arr[k] = aux[i++];
    } else if (cmp(aux[i], aux[j]) <= 0) {
      // ⚠️ <= 保证稳定性
      arr[k] = aux[i++];
    } else {
      arr[k] = aux[j++];
    }
  }
}

// ============================================================================
// 优化版本
// ============================================================================

/**
 * 优化版归并排序
 *
 * 优化点：
 * 1. 小数组使用插入排序
 * 2. 已有序时跳过合并
 */
export function sortOptimized<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const INSERTION_THRESHOLD = 16;
  const n = arr.length;

  if (n <= 1) return [...arr];

  const result = [...arr];
  const aux = new Array(n);

  optimizedMergeSort(result, aux, 0, n - 1, cmp, INSERTION_THRESHOLD);
  return result;
}

function optimizedMergeSort<T>(
  arr: T[],
  aux: T[],
  left: number,
  right: number,
  cmp: Comparator<T>,
  threshold: number
): void {
  // 优化 1：小数组用插入排序
  if (right - left < threshold) {
    insertionSortRange(arr, left, right, cmp);
    return;
  }

  const mid = (left + right) >> 1;
  optimizedMergeSort(arr, aux, left, mid, cmp, threshold);
  optimizedMergeSort(arr, aux, mid + 1, right, cmp, threshold);

  // 优化 2：如果已有序，跳过合并
  if (cmp(arr[mid], arr[mid + 1]) <= 0) {
    return;
  }

  mergeInPlace(arr, aux, left, mid, right, cmp);
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
// 工具函数
// ============================================================================

/**
 * 合并 K 个有序数组
 *
 * 使用两两合并策略，时间复杂度 O(N log K)
 * 其中 N 是所有元素总数，K 是数组个数
 */
export function mergeKSorted<T>(
  arrays: readonly T[][],
  cmp: Comparator<T>
): T[] {
  if (arrays.length === 0) return [];
  if (arrays.length === 1) return [...arrays[0]];

  let result = [...arrays];

  while (result.length > 1) {
    const merged: T[][] = [];
    for (let i = 0; i < result.length; i += 2) {
      if (i + 1 < result.length) {
        merged.push(merge(result[i], result[i + 1], cmp));
      } else {
        merged.push(result[i]);
      }
    }
    result = merged;
  }

  return result[0];
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '归并排序',
  englishName: 'Merge Sort',
  stable: true,
  inPlace: false,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n log n)',
    worst: 'O(n log n)',
  },
  spaceComplexity: 'O(n)',
  适用场景: ['稳定排序需求', '链表排序', '外部排序', '合并多个有序数据源'],
  不适用场景: ['内存受限', '小规模数据'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortIterative,
  sortOptimized,
  merge,
  mergeKSorted,
  meta,
};

