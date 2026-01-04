/**
 * 桶排序 (Bucket Sort)
 *
 * 非比较排序（桶分配阶段），将元素分散到多个桶中，
 * 每个桶内部排序，最后合并。
 *
 * 时间复杂度：O(n + k)（均匀分布），O(n²)（最坏）
 * 空间复杂度：O(n + k)
 * 稳定性：取决于桶内排序算法
 */

import type { Comparator } from '../../公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 桶排序（适用于 [0, 1) 范围的浮点数）
 *
 * @param arr 待排序数组，值在 [0, 1) 范围内
 * @returns 排序后的新数组
 *
 * @invariant 排序前后元素相同（置换性）
 * @invariant 返回数组按升序排列
 */
export function bucketSort(arr: readonly number[]): number[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  // 输入校验
  for (const num of arr) {
    if (num < 0 || num >= 1) {
      throw new Error(`值 ${num} 不在 [0, 1) 范围内`);
    }
  }

  // 创建 n 个桶
  const buckets: number[][] = Array.from({ length: n }, () => []);

  // 分配到桶：桶索引 = floor(value × n)
  for (const num of arr) {
    const bucketIdx = Math.min(n - 1, Math.floor(num * n));
    buckets[bucketIdx].push(num);
  }

  // 桶内排序（使用插入排序，因为桶内元素少）
  for (const bucket of buckets) {
    insertionSortNumbers(bucket);
  }

  // 合并所有桶
  const result: number[] = [];
  for (const bucket of buckets) {
    result.push(...bucket);
  }

  return result;
}

/**
 * 通用桶排序
 *
 * @param arr 待排序数组
 * @param bucketCount 桶数量
 * @param getBucketIndex 元素到桶索引的映射函数（返回 [0, bucketCount)）
 * @param cmp 比较函数（用于桶内排序）
 * @returns 排序后的新数组
 */
export function bucketSortGeneric<T>(
  arr: readonly T[],
  bucketCount: number,
  getBucketIndex: (item: T) => number,
  cmp: Comparator<T>
): T[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  if (bucketCount <= 0) {
    throw new Error(`桶数量必须为正数，收到：${bucketCount}`);
  }

  // 创建桶
  const buckets: T[][] = Array.from({ length: bucketCount }, () => []);

  // 分配到桶
  for (const item of arr) {
    const idx = getBucketIndex(item);
    if (idx < 0 || idx >= bucketCount) {
      throw new Error(`桶索引 ${idx} 超出范围 [0, ${bucketCount})`);
    }
    buckets[idx].push(item);
  }

  // 桶内排序（使用原生 sort，不保证稳定）
  for (const bucket of buckets) {
    bucket.sort(cmp);
  }

  // 合并
  const result: T[] = [];
  for (const bucket of buckets) {
    result.push(...bucket);
  }

  return result;
}

/**
 * 稳定的通用桶排序
 *
 * 使用插入排序保证桶内排序的稳定性
 */
export function bucketSortStable<T>(
  arr: readonly T[],
  bucketCount: number,
  getBucketIndex: (item: T) => number,
  cmp: Comparator<T>
): T[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  if (bucketCount <= 0) {
    throw new Error(`桶数量必须为正数，收到：${bucketCount}`);
  }

  // 创建桶
  const buckets: T[][] = Array.from({ length: bucketCount }, () => []);

  // 分配到桶（按原顺序，保证稳定）
  for (const item of arr) {
    const idx = getBucketIndex(item);
    if (idx < 0 || idx >= bucketCount) {
      throw new Error(`桶索引 ${idx} 超出范围 [0, ${bucketCount})`);
    }
    buckets[idx].push(item);
  }

  // 桶内使用插入排序（稳定）
  for (const bucket of buckets) {
    insertionSort(bucket, cmp);
  }

  // 合并
  const result: T[] = [];
  for (const bucket of buckets) {
    result.push(...bucket);
  }

  return result;
}

// ============================================================================
// 工厂函数
// ============================================================================

/**
 * 创建按数值范围分桶的桶排序函数
 *
 * @param min 值域最小值
 * @param max 值域最大值
 * @param bucketCount 桶数量
 */
export function createRangeBucketSort<T>(
  min: number,
  max: number,
  bucketCount: number,
  keyFn: (item: T) => number,
  cmp: Comparator<T>
): (arr: readonly T[]) => T[] {
  const range = max - min;
  const bucketSize = range / bucketCount;

  const getBucketIndex = (item: T): number => {
    const key = keyFn(item);
    if (key < min || key > max) {
      throw new Error(`值 ${key} 超出范围 [${min}, ${max}]`);
    }
    return Math.min(bucketCount - 1, Math.floor((key - min) / bucketSize));
  };

  return (arr: readonly T[]) =>
    bucketSortStable(arr, bucketCount, getBucketIndex, cmp);
}

// ============================================================================
// 辅助函数
// ============================================================================

/** 插入排序（数字专用） */
function insertionSortNumbers(arr: number[]): void {
  for (let i = 1; i < arr.length; i++) {
    const current = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > current) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = current;
  }
}

/** 通用插入排序（稳定） */
function insertionSort<T>(arr: T[], cmp: Comparator<T>): void {
  for (let i = 1; i < arr.length; i++) {
    const current = arr[i];
    let j = i - 1;
    while (j >= 0 && cmp(arr[j], current) > 0) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = current;
  }
}

// ============================================================================
// 元数据
// ============================================================================

export const meta = {
  name: '桶排序',
  stable: false, // 取决于桶内排序
  stableVariant: 'bucketSortStable',
  inPlace: false,
  timeComplexity: {
    best: 'O(n)',
    average: 'O(n + k)',
    worst: 'O(n²)',
  },
  spaceComplexity: 'O(n + k)',
  适用场景: [
    '均匀分布的数据',
    '可预知分布的数据',
    '分布式排序场景',
  ],
  不适用场景: [
    '分布极不均匀的数据',
    '无法确定映射函数的数据',
    '小规模数据（直接比较排序更简单）',
  ],
  输入限制: '需要定义映射函数',
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  bucketSort,
  bucketSortGeneric,
  bucketSortStable,
  createRangeBucketSort,
  meta,
};

