/**
 * 希尔排序 (Shell Sort)
 *
 * 核心思想：分组进行插入排序，逐步缩小间隔
 *
 * 时间复杂度：取决于间隔序列，平均 O(n^1.3) ~ O(n^1.5)
 * 空间复杂度：O(1)
 * 稳定性：❌ 不稳定
 *
 * 优势：比 O(n²) 快，实现简单，原地排序
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 间隔序列生成
// ============================================================================

/**
 * 生成 Knuth 序列 (推荐)
 * 序列: 1, 4, 13, 40, 121, ...
 * 公式: h = h * 3 + 1
 */
export function getKnuthGaps(n: number): number[] {
  const gaps: number[] = [];
  let gap = 1;

  // 找到最大的 gap < n/3
  while (gap < Math.floor(n / 3)) {
    gap = gap * 3 + 1;
  }

  // 从大到小生成序列
  while (gap >= 1) {
    gaps.push(gap);
    gap = Math.floor(gap / 3);
  }

  return gaps;
}

/**
 * 生成 Shell 原始序列 (n/2, n/4, ..., 1)
 */
export function getShellGaps(n: number): number[] {
  const gaps: number[] = [];
  let gap = Math.floor(n / 2);

  while (gap > 0) {
    gaps.push(gap);
    gap = Math.floor(gap / 2);
  }

  return gaps;
}

/**
 * 生成 Hibbard 序列 (1, 3, 7, 15, ...)
 * 公式: 2^k - 1
 */
export function getHibbardGaps(n: number): number[] {
  const gaps: number[] = [];
  let k = 1;

  while ((1 << k) - 1 < n) {
    gaps.push((1 << k) - 1);
    k++;
  }

  return gaps.reverse();
}

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 希尔排序（不修改原数组）
 *
 * 默认使用 Knuth 序列
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
 * 希尔排序（原地排序）
 *
 * @param arr 待排序数组（会被修改）
 * @param cmp 比较函数
 * @returns 排序后的数组（同一引用）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  const gaps = getKnuthGaps(arr.length);
  return sortWithGapsInPlace(arr, cmp, gaps);
}

/**
 * 使用自定义间隔序列的希尔排序
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @param gaps 间隔序列（从大到小）
 * @returns 排序后的新数组
 */
export function sortWithGaps<T>(
  arr: readonly T[],
  cmp: Comparator<T>,
  gaps: number[]
): T[] {
  const result = [...arr];
  return sortWithGapsInPlace(result, cmp, gaps);
}

/**
 * 使用自定义间隔序列的原地希尔排序
 */
export function sortWithGapsInPlace<T>(
  arr: T[],
  cmp: Comparator<T>,
  gaps: number[]
): T[] {
  const n = arr.length;
  if (n <= 1) return arr;

  for (const gap of gaps) {
    // 对每个间隔进行插入排序
    for (let i = gap; i < n; i++) {
      const current = arr[i];
      let j = i;

      // 插入排序：与前面间隔 gap 的元素比较
      while (j >= gap && cmp(arr[j - gap], current) > 0) {
        arr[j] = arr[j - gap];
        j -= gap;
      }

      arr[j] = current;
    }
  }

  return arr;
}

// ============================================================================
// 变种实现
// ============================================================================

/**
 * 使用 Shell 原始序列的排序
 */
export function sortShell<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  const gaps = getShellGaps(result.length);
  return sortWithGapsInPlace(result, cmp, gaps);
}

/**
 * 使用 Hibbard 序列的排序
 */
export function sortHibbard<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  const gaps = getHibbardGaps(result.length);
  return sortWithGapsInPlace(result, cmp, gaps);
}

/**
 * 带统计的希尔排序
 */
export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): { result: T[]; comparisons: number; moves: number; gaps: number[] } {
  const result = [...arr];
  const n = result.length;
  const gaps = getKnuthGaps(n);

  let comparisons = 0;
  let moves = 0;

  for (const gap of gaps) {
    for (let i = gap; i < n; i++) {
      const current = result[i];
      let j = i;

      while (j >= gap) {
        comparisons++;
        if (cmp(result[j - gap], current) > 0) {
          result[j] = result[j - gap];
          moves++;
          j -= gap;
        } else {
          break;
        }
      }

      if (j !== i) {
        result[j] = current;
        moves++;
      }
    }
  }

  return { result, comparisons, moves, gaps };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '希尔排序',
  englishName: 'Shell Sort',
  stable: false,  // ⚠️ 不稳定
  inPlace: true,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n^1.3)',  // Knuth 序列
    worst: 'O(n²)',       // Shell 原始序列
  },
  spaceComplexity: 'O(1)',
  适用场景: ['中等规模数据', '不需要稳定性', '内存受限环境'],
  不适用场景: ['需要稳定排序', '追求极致性能'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortWithGaps,
  sortWithGapsInPlace,
  sortShell,
  sortHibbard,
  sortWithStats,
  getKnuthGaps,
  getShellGaps,
  getHibbardGaps,
  meta,
};
