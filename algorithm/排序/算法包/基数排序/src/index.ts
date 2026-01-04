/**
 * 基数排序 (Radix Sort)
 *
 * 非比较排序，按每一位进行稳定子排序。
 * 适用于整数或定长字符串。
 *
 * 时间复杂度：O(d · (n + k))，d 是位数，k 是基数
 * 空间复杂度：O(n + k)
 * 稳定性：✅ 稳定（每轮使用计数排序）
 */

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 基数排序（LSD，非负整数）
 *
 * @param arr 待排序数组（非负整数）
 * @param radix 基数，默认 10
 * @returns 排序后的新数组
 *
 * @invariant 排序前后元素相同（置换性）
 * @invariant 相同值的元素保持原始相对顺序（稳定性）
 * @invariant 返回数组按升序排列
 */
export function radixSort(
  arr: readonly number[],
  radix: number = 10
): number[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  // 输入校验
  for (const num of arr) {
    if (!Number.isInteger(num)) {
      throw new Error(`基数排序仅支持整数，收到：${num}`);
    }
    if (num < 0) {
      throw new Error(`基数排序默认不支持负数，收到：${num}。请使用 radixSortWithNegative`);
    }
  }

  // 找最大值确定位数
  let max = 0;
  for (const num of arr) {
    if (num > max) max = num;
  }

  let result = [...arr];

  // 按每一位进行计数排序
  // exp 表示当前处理的位的权重（1, 10, 100, ...）
  for (let exp = 1; max / exp >= 1; exp *= radix) {
    result = countingSortByDigit(result, exp, radix);
  }

  return result;
}

/**
 * 按某一位进行计数排序（稳定）
 */
function countingSortByDigit(
  arr: number[],
  exp: number,
  radix: number
): number[] {
  const n = arr.length;
  const output = new Array<number>(n);
  const count = new Array<number>(radix).fill(0);

  // 1. 统计该位的计数
  for (const num of arr) {
    const digit = Math.floor(num / exp) % radix;
    count[digit]++;
  }

  // 2. 前缀和
  for (let i = 1; i < radix; i++) {
    count[i] += count[i - 1];
  }

  // 3. 从后往前回填（保证稳定性）
  for (let i = n - 1; i >= 0; i--) {
    const digit = Math.floor(arr[i] / exp) % radix;
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }

  return output;
}

// ============================================================================
// 对象排序版本
// ============================================================================

/**
 * 按整数字段进行基数排序
 *
 * @param arr 待排序对象数组
 * @param keyFn 提取整数 key 的函数（必须返回非负整数）
 * @param radix 基数，默认 10
 * @returns 排序后的新数组
 */
export function radixSortBy<T>(
  arr: readonly T[],
  keyFn: (item: T) => number,
  radix: number = 10
): T[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  // 找最大 key 确定位数
  let max = 0;
  for (const item of arr) {
    const key = keyFn(item);
    if (!Number.isInteger(key)) {
      throw new Error(`key 必须是整数，收到：${key}`);
    }
    if (key < 0) {
      throw new Error(`key 不能为负数，收到：${key}`);
    }
    if (key > max) max = key;
  }

  let result = [...arr];

  for (let exp = 1; max / exp >= 1; exp *= radix) {
    result = countingSortByDigitGeneric(result, keyFn, exp, radix);
  }

  return result;
}

/**
 * 按某一位对对象进行计数排序（稳定）
 */
function countingSortByDigitGeneric<T>(
  arr: T[],
  keyFn: (item: T) => number,
  exp: number,
  radix: number
): T[] {
  const n = arr.length;
  const output = new Array<T>(n);
  const count = new Array<number>(radix).fill(0);

  // 1. 计数
  for (const item of arr) {
    const digit = Math.floor(keyFn(item) / exp) % radix;
    count[digit]++;
  }

  // 2. 前缀和
  for (let i = 1; i < radix; i++) {
    count[i] += count[i - 1];
  }

  // 3. 从后往前回填（稳定）
  for (let i = n - 1; i >= 0; i--) {
    const digit = Math.floor(keyFn(arr[i]) / exp) % radix;
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }

  return output;
}

// ============================================================================
// 支持负数版本
// ============================================================================

/**
 * 基数排序（支持负数）
 *
 * @param arr 待排序数组（可包含负数）
 * @param radix 基数，默认 10
 * @returns 排序后的新数组
 */
export function radixSortWithNegative(
  arr: readonly number[],
  radix: number = 10
): number[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  // 分离正数和负数
  const negative: number[] = [];
  const nonNegative: number[] = [];

  for (const num of arr) {
    if (!Number.isInteger(num)) {
      throw new Error(`基数排序仅支持整数，收到：${num}`);
    }
    if (num < 0) {
      negative.push(-num); // 取绝对值
    } else {
      nonNegative.push(num);
    }
  }

  // 分别排序
  // 负数取绝对值排序后反转，再变回负数
  const sortedNegative = negative.length > 0
    ? radixSort(negative, radix).reverse().map(x => -x)
    : [];

  const sortedNonNegative = nonNegative.length > 0
    ? radixSort(nonNegative, radix)
    : [];

  // 合并：负数在前，非负数在后
  return [...sortedNegative, ...sortedNonNegative];
}

// ============================================================================
// 支持负数的对象版本
// ============================================================================

/**
 * 按整数字段进行基数排序（支持负数 key）
 */
export function radixSortByWithNegative<T>(
  arr: readonly T[],
  keyFn: (item: T) => number,
  radix: number = 10
): T[] {
  const n = arr.length;
  if (n <= 1) return [...arr];

  // 分离正负
  const negative: T[] = [];
  const nonNegative: T[] = [];

  for (const item of arr) {
    const key = keyFn(item);
    if (!Number.isInteger(key)) {
      throw new Error(`key 必须是整数，收到：${key}`);
    }
    if (key < 0) {
      negative.push(item);
    } else {
      nonNegative.push(item);
    }
  }

  // 负数按绝对值排序后反转
  const sortedNegative = negative.length > 0
    ? radixSortBy(negative, item => -keyFn(item), radix).reverse()
    : [];

  const sortedNonNegative = nonNegative.length > 0
    ? radixSortBy(nonNegative, keyFn, radix)
    : [];

  return [...sortedNegative, ...sortedNonNegative];
}

// ============================================================================
// 元数据
// ============================================================================

export const meta = {
  name: '基数排序',
  stable: true,
  inPlace: false,
  timeComplexity: {
    best: 'O(d·(n+k))',
    average: 'O(d·(n+k))',
    worst: 'O(d·(n+k))',
  },
  spaceComplexity: 'O(n + k)',
  适用场景: [
    '固定位数的整数排序',
    '手机号、订单号等数字 ID',
    '定长字符串排序',
    '需要稳定排序的整数数据',
  ],
  不适用场景: [
    '浮点数排序',
    '位数差异极大的数据',
    '混合正负数（需额外处理）',
  ],
  输入限制: '默认仅支持非负整数',
};

// ============================================================================
// 默认导出
// ============================================================================

export default {
  radixSort,
  radixSortBy,
  radixSortWithNegative,
  radixSortByWithNegative,
  meta,
};

