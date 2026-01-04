/**
 * 冒泡排序 (Bubble Sort)
 *
 * 核心思想：相邻元素比较交换，让大元素"冒泡"到末尾
 *
 * 时间复杂度：O(n²) 平均/最坏，O(n) 最好（已排序+优化）
 * 空间复杂度：O(1)
 * 稳定性：✅ 稳定
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 冒泡排序（不修改原数组）
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
 * 冒泡排序（原地排序）
 *
 * 实现优化：
 * 1. 提前终止：某轮无交换时结束
 * 2. 边界记录：只比较到上次最后交换位置
 *
 * @param arr 待排序数组（会被修改）
 * @param cmp 比较函数
 * @returns 排序后的数组（同一引用）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  const n = arr.length;
  if (n <= 1) return arr;

  let boundary = n - 1;  // 有序边界

  while (boundary > 0) {
    let lastSwapIndex = 0;  // 记录最后交换位置

    for (let j = 0; j < boundary; j++) {
      if (cmp(arr[j], arr[j + 1]) > 0) {
        // 交换
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        lastSwapIndex = j;
      }
    }

    // 更新边界：下次只需比较到上次最后交换位置
    boundary = lastSwapIndex;
  }

  return arr;
}

// ============================================================================
// 变种实现
// ============================================================================

/**
 * 基础版冒泡排序（无优化，用于教学）
 */
export function sortBasic<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  const n = result.length;

  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - 1 - i; j++) {
      if (cmp(result[j], result[j + 1]) > 0) {
        [result[j], result[j + 1]] = [result[j + 1], result[j]];
      }
    }
  }

  return result;
}

/**
 * 双向冒泡排序（鸡尾酒排序）
 *
 * 交替从左到右、从右到左冒泡
 * 对于"乌龟"元素（小元素在右边）更有效
 */
export function sortCocktail<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  const n = result.length;

  let left = 0;
  let right = n - 1;

  while (left < right) {
    // 从左到右，把最大的冒泡到右边
    let lastSwap = left;
    for (let i = left; i < right; i++) {
      if (cmp(result[i], result[i + 1]) > 0) {
        [result[i], result[i + 1]] = [result[i + 1], result[i]];
        lastSwap = i;
      }
    }
    right = lastSwap;

    // 从右到左，把最小的冒泡到左边
    lastSwap = right;
    for (let i = right; i > left; i--) {
      if (cmp(result[i - 1], result[i]) > 0) {
        [result[i - 1], result[i]] = [result[i], result[i - 1]];
        lastSwap = i;
      }
    }
    left = lastSwap;
  }

  return result;
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '冒泡排序',
  englishName: 'Bubble Sort',
  stable: true,
  inPlace: true,
  timeComplexity: {
    best: 'O(n)',
    average: 'O(n²)',
    worst: 'O(n²)',
  },
  spaceComplexity: 'O(1)',
  适用场景: ['教学演示', '小规模数据', '检测数组是否有序'],
  不适用场景: ['大规模数据', '性能敏感场景'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortBasic,
  sortCocktail,
  meta,
};
