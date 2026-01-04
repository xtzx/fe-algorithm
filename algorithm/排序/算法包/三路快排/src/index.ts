/**
 * 三路快速排序 (3-Way Quick Sort)
 *
 * 核心思想：Dijkstra 三路分区，将数组分为 <, =, > pivot 三部分
 *
 * 时间复杂度：O(n log n) 平均，O(n) 全相同元素
 * 空间复杂度：O(log n) 递归栈
 * 稳定性：❌ 不稳定
 *
 * 优势：大量重复元素时远优于标准快排
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 三路快速排序（不修改原数组）
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @returns 排序后的新数组
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  threeWayQuickSort(result, 0, result.length - 1, cmp);
  return result;
}

/**
 * 三路快速排序（原地排序）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  threeWayQuickSort(arr, 0, arr.length - 1, cmp);
  return arr;
}

/**
 * 三路快排核心递归
 */
function threeWayQuickSort<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): void {
  if (left >= right) return;

  // 随机选择 pivot
  const randomIndex = left + Math.floor(Math.random() * (right - left + 1));
  [arr[randomIndex], arr[left]] = [arr[left], arr[randomIndex]];

  const pivot = arr[left];

  // Dijkstra 三路分区
  // arr[left..lt-1] < pivot
  // arr[lt..i-1] = pivot
  // arr[gt+1..right] > pivot
  // arr[i..gt] 未检查

  let lt = left;
  let gt = right;
  let i = left + 1;

  while (i <= gt) {
    const c = cmp(arr[i], pivot);

    if (c < 0) {
      // arr[i] < pivot：交换到左边
      [arr[lt], arr[i]] = [arr[i], arr[lt]];
      lt++;
      i++;
    } else if (c > 0) {
      // arr[i] > pivot：交换到右边
      [arr[i], arr[gt]] = [arr[gt], arr[i]];
      gt--;
      // 注意：i 不递增，因为交换来的元素还未检查
    } else {
      // arr[i] = pivot：保持在中间
      i++;
    }
  }

  // 递归处理 < pivot 和 > pivot 的部分
  // = pivot 的部分已经在正确位置
  threeWayQuickSort(arr, left, lt - 1, cmp);
  threeWayQuickSort(arr, gt + 1, right, cmp);
}

// ============================================================================
// 带优化的版本
// ============================================================================

const INSERTION_THRESHOLD = 16;

/**
 * 优化版三路快排
 *
 * - 小数组使用插入排序
 * - 尾递归优化
 */
export function sortOptimized<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  optimizedThreeWay(result, 0, result.length - 1, cmp);
  return result;
}

function optimizedThreeWay<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): void {
  while (left < right) {
    // 小数组用插入排序
    if (right - left < INSERTION_THRESHOLD) {
      insertionSortRange(arr, left, right, cmp);
      return;
    }

    // 三路分区
    const [lt, gt] = threeWayPartition(arr, left, right, cmp);

    // 尾递归优化：先递归较小的分区
    if (lt - left < right - gt) {
      optimizedThreeWay(arr, left, lt - 1, cmp);
      left = gt + 1; // 尾递归转循环
    } else {
      optimizedThreeWay(arr, gt + 1, right, cmp);
      right = lt - 1;
    }
  }
}

/**
 * 三路分区
 *
 * @returns [lt, gt] - 相等区间的边界
 */
function threeWayPartition<T>(
  arr: T[],
  left: number,
  right: number,
  cmp: Comparator<T>
): [number, number] {
  // 随机 pivot
  const randomIndex = left + Math.floor(Math.random() * (right - left + 1));
  [arr[randomIndex], arr[left]] = [arr[left], arr[randomIndex]];

  const pivot = arr[left];
  let lt = left;
  let gt = right;
  let i = left + 1;

  while (i <= gt) {
    const c = cmp(arr[i], pivot);
    if (c < 0) {
      [arr[lt], arr[i]] = [arr[i], arr[lt]];
      lt++;
      i++;
    } else if (c > 0) {
      [arr[i], arr[gt]] = [arr[gt], arr[i]];
      gt--;
    } else {
      i++;
    }
  }

  return [lt, gt];
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

export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): {
  result: T[];
  comparisons: number;
  swaps: number;
  recursions: number;
} {
  const result = [...arr];
  const stats = { comparisons: 0, swaps: 0, recursions: 0 };

  function sortHelper(left: number, right: number): void {
    if (left >= right) return;
    stats.recursions++;

    const randomIndex = left + Math.floor(Math.random() * (right - left + 1));
    [result[randomIndex], result[left]] = [result[left], result[randomIndex]];
    stats.swaps++;

    const pivot = result[left];
    let lt = left;
    let gt = right;
    let i = left + 1;

    while (i <= gt) {
      stats.comparisons++;
      const c = cmp(result[i], pivot);

      if (c < 0) {
        [result[lt], result[i]] = [result[i], result[lt]];
        stats.swaps++;
        lt++;
        i++;
      } else if (c > 0) {
        [result[i], result[gt]] = [result[gt], result[i]];
        stats.swaps++;
        gt--;
      } else {
        i++;
      }
    }

    sortHelper(left, lt - 1);
    sortHelper(gt + 1, right);
  }

  sortHelper(0, result.length - 1);
  return { result, ...stats };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '三路快速排序',
  englishName: '3-Way Quick Sort',
  stable: false,
  inPlace: true,
  timeComplexity: {
    best: 'O(n log n)',
    average: 'O(n log n)',
    worst: 'O(n²)', // 但全相同时是 O(n)
  },
  spaceComplexity: 'O(log n)',
  特点: '全相同元素时 O(n)',
  适用场景: ['大量重复元素', '枚举值排序', '状态码排序'],
  不适用场景: ['几乎无重复的数据', '需要稳定排序'],
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  sort,
  sortInPlace,
  sortOptimized,
  sortWithStats,
  meta,
};

