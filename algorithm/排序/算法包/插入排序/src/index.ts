/**
 * 插入排序 (Insertion Sort)
 *
 * 核心思想：将每个元素插入到已排序部分的正确位置
 *
 * 时间复杂度：O(n) 最好（已排序），O(n²) 平均/最坏
 * 空间复杂度：O(1)
 * 稳定性：✅ 稳定
 *
 * 优势：近乎有序数据极快，常数因子小
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 插入排序（不修改原数组）
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
 * 插入排序（原地排序）
 *
 * @param arr 待排序数组（会被修改）
 * @param cmp 比较函数
 * @returns 排序后的数组（同一引用）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  const n = arr.length;

  for (let i = 1; i < n; i++) {
    const current = arr[i];
    let j = i - 1;

    // 从后往前找插入位置，同时移动元素
    while (j >= 0 && cmp(arr[j], current) > 0) {
      arr[j + 1] = arr[j];
      j--;
    }

    arr[j + 1] = current;
  }

  return arr;
}

/**
 * 插入排序（指定范围）
 *
 * 对 arr[left..right] 进行插入排序
 * 用于其他算法的子程序（如 TimSort、Introsort）
 *
 * @param arr 待排序数组
 * @param left 起始索引（包含）
 * @param right 结束索引（包含）
 * @param cmp 比较函数
 */
export function sortRange<T>(
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
// 变种实现
// ============================================================================

/**
 * 二分插入排序
 *
 * 使用二分查找找到插入位置，减少比较次数
 * 比较次数：O(n log n)，移动次数：仍是 O(n²)
 */
export function sortBinary<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  const n = result.length;

  for (let i = 1; i < n; i++) {
    const current = result[i];

    // 二分查找插入位置
    let left = 0;
    let right = i;

    while (left < right) {
      const mid = (left + right) >>> 1;
      if (cmp(result[mid], current) > 0) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    // 移动元素
    for (let j = i; j > left; j--) {
      result[j] = result[j - 1];
    }

    result[left] = current;
  }

  return result;
}

/**
 * 向已排序数组插入单个元素
 *
 * 用于在线排序场景
 * 时间复杂度：O(n)
 *
 * @param arr 已排序数组（会被修改）
 * @param item 要插入的元素
 * @param cmp 比较函数
 * @returns 插入位置索引
 */
export function insertSorted<T>(
  arr: T[],
  item: T,
  cmp: Comparator<T>
): number {
  // 二分查找插入位置
  let left = 0;
  let right = arr.length;

  while (left < right) {
    const mid = (left + right) >>> 1;
    if (cmp(arr[mid], item) > 0) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }

  // 插入元素
  arr.splice(left, 0, item);

  return left;
}

/**
 * 带计数的插入排序（返回比较和移动次数）
 */
export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): { result: T[]; comparisons: number; moves: number } {
  const result = [...arr];
  const n = result.length;
  let comparisons = 0;
  let moves = 0;

  for (let i = 1; i < n; i++) {
    const current = result[i];
    let j = i - 1;

    while (j >= 0) {
      comparisons++;
      if (cmp(result[j], current) > 0) {
        result[j + 1] = result[j];
        moves++;
        j--;
      } else {
        break;
      }
    }

    result[j + 1] = current;
    if (j + 1 !== i) moves++;  // 最后的赋值也算一次移动
  }

  return { result, comparisons, moves };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '插入排序',
  englishName: 'Insertion Sort',
  stable: true,
  inPlace: true,
  timeComplexity: {
    best: 'O(n)',      // ⭐ 近乎有序时极快
    average: 'O(n²)',
    worst: 'O(n²)',
  },
  spaceComplexity: 'O(1)',
  适用场景: ['小规模数据', '近乎有序数据', '在线排序', '作为其他算法的子程序'],
  不适用场景: ['大规模随机数据'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortRange,
  sortBinary,
  insertSorted,
  sortWithStats,
  meta,
};
