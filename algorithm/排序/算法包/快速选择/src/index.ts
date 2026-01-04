/**
 * 快速选择 (Quick Select)
 *
 * 核心思想：利用快排的 partition，但只递归处理包含目标的那一边
 *
 * 时间复杂度：O(n) 平均，O(n²) 最坏
 * 空间复杂度：O(1)
 *
 * 优势：平均 O(n) 找到第 K 大/小元素
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 快速选择：找到第 k 小的元素（k 从 0 开始）
 *
 * ⚠️ 注意：会修改原数组！
 *
 * @param arr 输入数组（会被修改）
 * @param k 目标位置（0-indexed）
 * @param cmp 比较函数
 * @returns 第 k 小的元素
 *
 * @example
 * const arr = [3, 1, 4, 1, 5];
 * quickSelect(arr, 0, numberAsc); // 1（最小）
 * quickSelect(arr, 2, numberAsc); // 3（第 3 小）
 */
export function quickSelect<T>(
  arr: T[],
  k: number,
  cmp: Comparator<T>
): T {
  if (k < 0 || k >= arr.length) {
    throw new Error(`k=${k} 超出范围 [0, ${arr.length - 1}]`);
  }

  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    const pivotIndex = partitionRandom(arr, left, right, cmp);

    if (pivotIndex === k) {
      return arr[k];
    } else if (pivotIndex < k) {
      left = pivotIndex + 1;
    } else {
      right = pivotIndex - 1;
    }
  }

  return arr[k];
}

/**
 * 不修改原数组的版本
 */
export function quickSelectCopy<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T {
  return quickSelect([...arr], k, cmp);
}

// ============================================================================
// 分区实现
// ============================================================================

/**
 * 随机 pivot 分区（Lomuto 方案）
 */
function partitionRandom<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): number {
  // 随机选择 pivot 避免最坏情况
  const randomIndex = left + Math.floor(Math.random() * (right - left + 1));
  [arr[randomIndex], arr[right]] = [arr[right], arr[randomIndex]];

  const pivot = arr[right];
  let i = left - 1;

  for (let j = left; j < right; j++) {
    if (cmp(arr[j], pivot) < 0) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  [arr[i + 1], arr[right]] = [arr[right], arr[i + 1]];
  return i + 1;
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
  if (right - left < 3) {
    return partitionRandom(arr, left, right, cmp);
  }

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

  // 将中值移到 right-1
  [arr[mid], arr[right - 1]] = [arr[right - 1], arr[mid]];

  return partitionRandom(arr, left, right, cmp);
}

// ============================================================================
// TopK 实现
// ============================================================================

/**
 * 返回最小的 k 个元素（无序）
 *
 * 时间复杂度：O(n) 平均
 *
 * @param arr 输入数组（不会被修改）
 * @param k 要返回的元素个数
 * @param cmp 比较函数
 */
export function topKSmallest<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  if (k <= 0) return [];
  if (k >= arr.length) return [...arr];

  const copy = [...arr];

  // quickSelect 后，前 k 个元素就是最小的 k 个（无序）
  quickSelect(copy, k - 1, cmp);

  return copy.slice(0, k);
}

/**
 * 返回最大的 k 个元素（无序）
 */
export function topKLargest<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  // 反转比较器
  const reverseCmp: Comparator<T> = (a, b) => cmp(b, a);
  return topKSmallest(arr, k, reverseCmp);
}

/**
 * 返回最小的 k 个元素（有序）
 */
export function topKSmallestSorted<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  return topKSmallest(arr, k, cmp).sort(cmp);
}

/**
 * 返回最大的 k 个元素（有序，降序）
 */
export function topKLargestSorted<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T[] {
  const result = topKLargest(arr, k, cmp);
  return result.sort((a, b) => cmp(b, a));
}

// ============================================================================
// 统计函数
// ============================================================================

/**
 * 计算中位数
 */
export function median(arr: readonly number[]): number {
  if (arr.length === 0) {
    throw new Error('数组不能为空');
  }

  const copy = [...arr];
  const n = copy.length;
  const cmp = (a: number, b: number) => a - b;

  if (n % 2 === 1) {
    // 奇数：中间那个
    return quickSelect(copy, n >> 1, cmp);
  } else {
    // 偶数：中间两个的平均
    const mid1 = quickSelect(copy, (n >> 1) - 1, cmp);
    // 此时 copy 已被部分排序，mid1 左边都小于等于它
    // 需要重新 quickSelect 找第二个中位数
    const copy2 = [...arr];
    const mid2 = quickSelect(copy2, n >> 1, cmp);
    return (mid1 + mid2) / 2;
  }
}

/**
 * 计算百分位数
 *
 * @param arr 输入数组
 * @param p 百分位（0-1 之间）
 */
export function percentile(arr: readonly number[], p: number): number {
  if (arr.length === 0) {
    throw new Error('数组不能为空');
  }
  if (p < 0 || p > 1) {
    throw new Error('p 应在 [0, 1] 范围内');
  }

  const copy = [...arr];
  const index = Math.floor((copy.length - 1) * p);
  return quickSelect(copy, index, (a, b) => a - b);
}

/**
 * 计算多个百分位数
 *
 * 优化：从大到小计算，复用之前的 partition 结果
 */
export function percentiles(
  arr: readonly number[],
  ps: number[]
): Map<number, number> {
  const result = new Map<number, number>();

  for (const p of ps) {
    result.set(p, percentile(arr, p));
  }

  return result;
}

// ============================================================================
// 第 K 大/小（LeetCode 风格，1-indexed）
// ============================================================================

/**
 * 第 K 小的元素（1-indexed，LeetCode 风格）
 */
export function kthSmallest<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T {
  if (k < 1 || k > arr.length) {
    throw new Error(`k=${k} 超出范围 [1, ${arr.length}]`);
  }
  return quickSelectCopy(arr, k - 1, cmp);
}

/**
 * 第 K 大的元素（1-indexed，LeetCode 风格）
 */
export function kthLargest<T>(
  arr: readonly T[],
  k: number,
  cmp: Comparator<T>
): T {
  if (k < 1 || k > arr.length) {
    throw new Error(`k=${k} 超出范围 [1, ${arr.length}]`);
  }
  // 第 k 大 = 第 (n - k + 1) 小
  return quickSelectCopy(arr, arr.length - k, cmp);
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '快速选择',
  englishName: 'Quick Select',
  timeComplexity: {
    average: 'O(n)',
    worst: 'O(n²)', // ⚠️
  },
  spaceComplexity: 'O(1)',
  适用场景: ['第 K 大/小元素', 'TopK 问题（无需有序）', '中位数', '百分位数'],
  不适用场景: ['在线/流式数据', '需要 TopK 有序'],
  与快排关系: '快排每次处理两边，快选只处理一边',
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  quickSelect,
  quickSelectCopy,
  topKSmallest,
  topKLargest,
  topKSmallestSorted,
  topKLargestSorted,
  kthSmallest,
  kthLargest,
  median,
  percentile,
  percentiles,
  meta,
};

