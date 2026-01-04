/**
 * 选择排序 (Selection Sort)
 *
 * 核心思想：每轮从未排序部分选择最小元素，放到已排序部分末尾
 *
 * 时间复杂度：O(n²)（任何情况）
 * 空间复杂度：O(1)
 * 稳定性：❌ 不稳定
 *
 * 优势：交换次数最少，只有 O(n) 次
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 选择排序（不修改原数组）
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @returns 排序后的新数组
 *
 * @example
 * sort([5, 3, 8], (a, b) => a - b)  // [3, 5, 8]
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  return sortInPlace(result, cmp);
}

/**
 * 选择排序（原地排序）
 *
 * @param arr 待排序数组（会被修改）
 * @param cmp 比较函数
 * @returns 排序后的数组（同一引用）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  const n = arr.length;

  for (let i = 0; i < n - 1; i++) {
    // 找到 [i, n-1] 中的最小元素
    let minIndex = i;
    for (let j = i + 1; j < n; j++) {
      if (cmp(arr[j], arr[minIndex]) < 0) {
        minIndex = j;
      }
    }

    // 交换到位置 i
    if (minIndex !== i) {
      [arr[i], arr[minIndex]] = [arr[minIndex], arr[i]];
    }
  }

  return arr;
}

// ============================================================================
// 变种实现
// ============================================================================

/**
 * 双向选择排序
 *
 * 每轮同时找最小和最大，分别放到两端
 * 比较次数减半，但仍是 O(n²)
 */
export function sortBidirectional<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  const n = result.length;

  let left = 0;
  let right = n - 1;

  while (left < right) {
    let minIndex = left;
    let maxIndex = left;

    // 同时找最小和最大
    for (let i = left; i <= right; i++) {
      if (cmp(result[i], result[minIndex]) < 0) {
        minIndex = i;
      }
      if (cmp(result[i], result[maxIndex]) > 0) {
        maxIndex = i;
      }
    }

    // 先交换最小到左边
    if (minIndex !== left) {
      [result[left], result[minIndex]] = [result[minIndex], result[left]];
    }

    // 如果最大值在 left 位置，被交换走了，更新索引
    if (maxIndex === left) {
      maxIndex = minIndex;
    }

    // 再交换最大到右边
    if (maxIndex !== right) {
      [result[right], result[maxIndex]] = [result[maxIndex], result[right]];
    }

    left++;
    right--;
  }

  return result;
}

/**
 * 带计数的选择排序（返回交换次数）
 *
 * 用于演示选择排序交换次数少的优势
 */
export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): { result: T[]; swaps: number; comparisons: number } {
  const result = [...arr];
  const n = result.length;
  let swaps = 0;
  let comparisons = 0;

  for (let i = 0; i < n - 1; i++) {
    let minIndex = i;
    for (let j = i + 1; j < n; j++) {
      comparisons++;
      if (cmp(result[j], result[minIndex]) < 0) {
        minIndex = j;
      }
    }

    if (minIndex !== i) {
      [result[i], result[minIndex]] = [result[minIndex], result[i]];
      swaps++;
    }
  }

  return { result, swaps, comparisons };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '选择排序',
  englishName: 'Selection Sort',
  stable: false,  // ⚠️ 不稳定
  inPlace: true,
  timeComplexity: {
    best: 'O(n²)',
    average: 'O(n²)',
    worst: 'O(n²)',
  },
  spaceComplexity: 'O(1)',
  适用场景: ['交换成本高', '小规模数据', '教学演示'],
  不适用场景: ['需要稳定排序', '大规模数据'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortBidirectional,
  sortWithStats,
  meta,
};
