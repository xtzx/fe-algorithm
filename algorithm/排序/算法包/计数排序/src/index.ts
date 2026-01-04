/**
 * 计数排序 (Counting Sort)
 *
 * 非比较排序，通过统计元素出现次数来排序。
 * 适用于小范围整数。
 *
 * 时间复杂度：O(n + k)，k 为值域大小
 * 空间复杂度：O(n + k)
 * 稳定性：✅ 稳定（从后往前回填）
 */

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 计数排序（指定范围）
 *
 * @param arr 待排序数组（非负整数）
 * @param min 值域最小值
 * @param max 值域最大值
 * @returns 排序后的新数组
 *
 * @invariant 排序前后元素相同（置换性）
 * @invariant 相同值的元素保持原始相对顺序（稳定性）
 * @invariant 返回数组按升序排列
 */
export function countingSort(
  arr: readonly number[],
  min: number,
  max: number
): number[] {
  // 输入校验
  if (min > max) {
    throw new Error(`无效范围：min(${min}) > max(${max})`);
  }

  const n = arr.length;
  if (n <= 1) return [...arr];

  const range = max - min + 1;
  const count = new Array<number>(range).fill(0);
  const output = new Array<number>(n);

  // 1. 统计每个值的出现次数
  for (const num of arr) {
    if (!Number.isInteger(num)) {
      throw new Error(`计数排序仅支持整数，收到：${num}`);
    }
    if (num < min || num > max) {
      throw new Error(`值 ${num} 超出范围 [${min}, ${max}]`);
    }
    count[num - min]++;
  }

  // 2. 计算前缀和（累积计数）
  // count[i] 表示 ≤(min+i) 的元素个数
  for (let i = 1; i < range; i++) {
    count[i] += count[i - 1];
  }

  // 3. 从后往前回填（保证稳定性）
  // 后面的元素放在后面的位置
  for (let i = n - 1; i >= 0; i--) {
    const idx = arr[i] - min;
    output[count[idx] - 1] = arr[i];
    count[idx]--;
  }

  return output;
}

/**
 * 计数排序（自动检测范围）
 *
 * @param arr 待排序数组
 * @returns 排序后的新数组
 */
export function countingSortAuto(arr: readonly number[]): number[] {
  if (arr.length === 0) return [];

  let min = arr[0];
  let max = arr[0];

  for (const num of arr) {
    if (!Number.isInteger(num)) {
      throw new Error(`计数排序仅支持整数，收到：${num}`);
    }
    if (num < min) min = num;
    if (num > max) max = num;
  }

  return countingSort(arr, min, max);
}

// ============================================================================
// 对象排序版本
// ============================================================================

/**
 * 按整数字段进行计数排序
 *
 * @param arr 待排序对象数组
 * @param keyFn 提取整数 key 的函数
 * @param min key 的最小值
 * @param max key 的最大值
 * @returns 排序后的新数组
 */
export function countingSortBy<T>(
  arr: readonly T[],
  keyFn: (item: T) => number,
  min: number,
  max: number
): T[] {
  if (min > max) {
    throw new Error(`无效范围：min(${min}) > max(${max})`);
  }

  const n = arr.length;
  if (n <= 1) return [...arr];

  const range = max - min + 1;
  const count = new Array<number>(range).fill(0);
  const output = new Array<T>(n);

  // 1. 按 key 计数
  for (const item of arr) {
    const key = keyFn(item);
    if (!Number.isInteger(key)) {
      throw new Error(`key 必须是整数，收到：${key}`);
    }
    if (key < min || key > max) {
      throw new Error(`key ${key} 超出范围 [${min}, ${max}]`);
    }
    count[key - min]++;
  }

  // 2. 前缀和
  for (let i = 1; i < range; i++) {
    count[i] += count[i - 1];
  }

  // 3. 从后往前回填（稳定）
  for (let i = n - 1; i >= 0; i--) {
    const key = keyFn(arr[i]) - min;
    output[count[key] - 1] = arr[i];
    count[key]--;
  }

  return output;
}

/**
 * 按整数字段进行计数排序（自动检测范围）
 *
 * @param arr 待排序对象数组
 * @param keyFn 提取整数 key 的函数
 * @returns 排序后的新数组
 */
export function countingSortByAuto<T>(
  arr: readonly T[],
  keyFn: (item: T) => number
): T[] {
  if (arr.length === 0) return [];

  let min = keyFn(arr[0]);
  let max = keyFn(arr[0]);

  for (const item of arr) {
    const key = keyFn(item);
    if (!Number.isInteger(key)) {
      throw new Error(`key 必须是整数，收到：${key}`);
    }
    if (key < min) min = key;
    if (key > max) max = key;
  }

  return countingSortBy(arr, keyFn, min, max);
}

// ============================================================================
// 元数据
// ============================================================================

export const meta = {
  name: '计数排序',
  stable: true,
  inPlace: false,
  timeComplexity: {
    best: 'O(n + k)',
    average: 'O(n + k)',
    worst: 'O(n + k)',
  },
  spaceComplexity: 'O(n + k)',
  适用场景: [
    '小范围整数排序',
    '分数/年龄/状态码等有限值域数据',
    '需要稳定排序的整数数据',
  ],
  不适用场景: [
    '浮点数排序',
    '值域极大（如 0~10^9）',
    '内存受限场景',
  ],
  输入限制: '整数，需指定值域范围',
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  countingSort,
  countingSortAuto,
  countingSortBy,
  countingSortByAuto,
  meta,
};

