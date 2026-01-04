/**
 * TimSort (教学简化版)
 *
 * 核心思想：利用现实数据中的有序片段（run），结合插入排序和归并排序
 *
 * 特点：
 * - 检测自然有序的 run
 * - 短 run 用插入排序扩展到 minrun
 * - 按规则合并 run
 *
 * 时间复杂度：O(n) 最好（完全有序），O(n log n) 平均/最坏
 * 空间复杂度：O(n) 合并需要
 * 稳定性：✅ 稳定
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 常量
// ============================================================================

const MIN_MERGE = 32;

// ============================================================================
// 核心实现
// ============================================================================

/**
 * TimSort（不修改原数组）
 */
export function sort<T>(arr: readonly T[], cmp: Comparator<T>): T[] {
  const result = [...arr];
  sortInPlace(result, cmp);
  return result;
}

/**
 * TimSort（原地排序）
 */
export function sortInPlace<T>(arr: T[], cmp: Comparator<T>): T[] {
  const n = arr.length;

  if (n < 2) return arr;

  // 小数组直接用插入排序
  if (n < MIN_MERGE) {
    insertionSort(arr, 0, n - 1, cmp);
    return arr;
  }

  const minrun = computeMinrun(n);
  const runs: Run[] = [];

  let lo = 0;

  while (lo < n) {
    // 1. 检测 run 并确保升序
    let runLen = countRunAndMakeAscending(arr, lo, n - 1, cmp);

    // 2. 如果 run 太短，用插入排序扩展到 minrun
    if (runLen < minrun) {
      const force = Math.min(n - lo, minrun);
      insertionSort(arr, lo, lo + force - 1, cmp);
      runLen = force;
    }

    // 3. 将 run 压入栈
    runs.push({ start: lo, length: runLen });

    // 4. 检查并执行合并
    mergeCollapse(arr, runs, cmp);

    lo += runLen;
  }

  // 5. 合并所有剩余的 run
  mergeForceCollapse(arr, runs, cmp);

  return arr;
}

// ============================================================================
// 类型
// ============================================================================

interface Run {
  start: number;
  length: number;
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 计算 minrun
 *
 * 保证 n/minrun 是 2 的幂或略小，这样合并时效率最高
 */
function computeMinrun(n: number): number {
  let r = 0;
  while (n >= MIN_MERGE) {
    r |= n & 1;
    n >>= 1;
  }
  return n + r;
}

/**
 * 检测 run 并确保升序
 *
 * 返回 run 的长度
 * 如果是降序 run，会反转为升序
 */
function countRunAndMakeAscending<T>(
  arr: T[],
  lo: number,
  hi: number,
  cmp: Comparator<T>
): number {
  let runHi = lo + 1;

  if (runHi > hi) return 1;

  // 检查是升序还是降序
  if (cmp(arr[runHi], arr[lo]) < 0) {
    // 严格降序
    while (runHi <= hi && cmp(arr[runHi], arr[runHi - 1]) < 0) {
      runHi++;
    }
    // 反转为升序
    reverseRange(arr, lo, runHi - 1);
  } else {
    // 非严格升序（包括相等）
    while (runHi <= hi && cmp(arr[runHi], arr[runHi - 1]) >= 0) {
      runHi++;
    }
  }

  return runHi - lo;
}

/**
 * 反转数组区间
 */
function reverseRange<T>(arr: T[], lo: number, hi: number): void {
  while (lo < hi) {
    [arr[lo], arr[hi]] = [arr[hi], arr[lo]];
    lo++;
    hi--;
  }
}

/**
 * 插入排序
 */
function insertionSort<T>(
  arr: T[],
  lo: number,
  hi: number,
  cmp: Comparator<T>
): void {
  for (let i = lo + 1; i <= hi; i++) {
    const current = arr[i];
    let j = i - 1;

    while (j >= lo && cmp(arr[j], current) > 0) {
      arr[j + 1] = arr[j];
      j--;
    }

    arr[j + 1] = current;
  }
}

// ============================================================================
// 合并相关
// ============================================================================

/**
 * 检查并执行合并
 *
 * 维护不变式：
 * - runLen[n-2] > runLen[n-1] + runLen[n]
 * - runLen[n-1] > runLen[n]
 */
function mergeCollapse<T>(arr: T[], runs: Run[], cmp: Comparator<T>): void {
  while (runs.length > 1) {
    let n = runs.length - 2;

    if (
      n > 0 &&
      runs[n - 1].length <= runs[n].length + runs[n + 1].length
    ) {
      // 违反第一个不变式
      if (runs[n - 1].length < runs[n + 1].length) {
        n--;
      }
      mergeAt(arr, runs, n, cmp);
    } else if (runs[n].length <= runs[n + 1].length) {
      // 违反第二个不变式
      mergeAt(arr, runs, n, cmp);
    } else {
      // 满足不变式，停止
      break;
    }
  }
}

/**
 * 强制合并所有剩余的 run
 */
function mergeForceCollapse<T>(arr: T[], runs: Run[], cmp: Comparator<T>): void {
  while (runs.length > 1) {
    let n = runs.length - 2;

    if (n > 0 && runs[n - 1].length < runs[n + 1].length) {
      n--;
    }

    mergeAt(arr, runs, n, cmp);
  }
}

/**
 * 合并 runs[n] 和 runs[n+1]
 */
function mergeAt<T>(arr: T[], runs: Run[], n: number, cmp: Comparator<T>): void {
  const run1 = runs[n];
  const run2 = runs[n + 1];

  // 稳定合并
  stableMerge(
    arr,
    run1.start,
    run1.start + run1.length - 1,
    run2.start + run2.length - 1,
    cmp
  );

  // 更新栈
  runs[n] = { start: run1.start, length: run1.length + run2.length };
  runs.splice(n + 1, 1);
}

/**
 * 稳定合并 arr[lo..mid] 和 arr[mid+1..hi]
 */
function stableMerge<T>(
  arr: T[],
  lo: number,
  mid: number,
  hi: number,
  cmp: Comparator<T>
): void {
  // 复制左半部分
  const leftLen = mid - lo + 1;
  const left = arr.slice(lo, mid + 1);

  let i = 0; // left 指针
  let j = mid + 1; // 右半部分指针
  let k = lo; // 输出指针

  while (i < leftLen && j <= hi) {
    // <= 保证稳定性：相等时取左边
    if (cmp(left[i], arr[j]) <= 0) {
      arr[k++] = left[i++];
    } else {
      arr[k++] = arr[j++];
    }
  }

  // 复制剩余的左半部分
  while (i < leftLen) {
    arr[k++] = left[i++];
  }

  // 右半部分已经在正确位置
}

// ============================================================================
// 带统计版本
// ============================================================================

export function sortWithStats<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): {
  result: T[];
  runs: number;
  merges: number;
  comparisons: number;
} {
  const result = [...arr];
  const n = result.length;
  const stats = { runs: 0, merges: 0, comparisons: 0 };

  if (n < 2) return { result, ...stats };

  if (n < MIN_MERGE) {
    insertionSort(result, 0, n - 1, cmp);
    return { result, ...stats };
  }

  const minrun = computeMinrun(n);
  const runs: Run[] = [];

  let lo = 0;

  while (lo < n) {
    let runLen = countRunAndMakeAscending(result, lo, n - 1, cmp);
    stats.runs++;

    if (runLen < minrun) {
      const force = Math.min(n - lo, minrun);
      insertionSort(result, lo, lo + force - 1, cmp);
      runLen = force;
    }

    runs.push({ start: lo, length: runLen });

    // 简化的合并计数
    const prevLen = runs.length;
    mergeCollapse(result, runs, cmp);
    stats.merges += prevLen - runs.length;

    lo += runLen;
  }

  const prevLen = runs.length;
  mergeForceCollapse(result, runs, cmp);
  stats.merges += prevLen - runs.length;

  return { result, ...stats };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: 'TimSort',
  stable: true,
  inPlace: false,
  timeComplexity: {
    best: 'O(n)', // ⭐ 完全有序时
    average: 'O(n log n)',
    worst: 'O(n log n)',
  },
  spaceComplexity: 'O(n)',
  特点: '近乎有序数据最优',
  适用场景: ['近乎有序数据', '表格多列排序', '需要稳定排序'],
  不适用场景: ['完全随机数据', '内存受限'],
  使用者: ['Python list.sort()', 'Java Arrays.sort() (对象)', 'Rust stable_sort'],
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

