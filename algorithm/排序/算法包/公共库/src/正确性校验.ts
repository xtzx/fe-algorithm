/**
 * 正确性校验模块
 *
 * 验证排序算法的正确性：有序性、置换性、稳定性。
 */

import type { Comparator } from './比较器';

// ============================================================================
// 类型定义
// ============================================================================

export interface VerificationResult {
  passed: boolean;
  error?: string;
  details?: unknown;
}

// ============================================================================
// 有序性校验
// ============================================================================

/**
 * 验证数组是否有序
 *
 * @param arr 排序后的数组
 * @param cmp 比较函数
 */
export function verifySorted<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): VerificationResult {
  for (let i = 1; i < arr.length; i++) {
    if (cmp(arr[i - 1], arr[i]) > 0) {
      return {
        passed: false,
        error: `无序：arr[${i - 1}] > arr[${i}]`,
        details: { index: i - 1, left: arr[i - 1], right: arr[i] },
      };
    }
  }
  return { passed: true };
}

/**
 * 断言数组有序，失败时抛出异常
 */
export function assertSorted<T>(
  arr: readonly T[],
  cmp: Comparator<T>,
  message?: string
): void {
  const result = verifySorted(arr, cmp);
  if (!result.passed) {
    throw new Error(message || result.error);
  }
}

// ============================================================================
// 置换性校验
// ============================================================================

/**
 * 验证 sorted 是 original 的置换（元素相同，可能顺序不同）
 */
export function verifyPermutation<T>(
  original: readonly T[],
  sorted: readonly T[]
): VerificationResult {
  if (original.length !== sorted.length) {
    return {
      passed: false,
      error: `长度不同：原 ${original.length}，现 ${sorted.length}`,
    };
  }

  // 简单实现：排序后比较（O(n log n)）
  // 注意：这里假设元素可以用 JSON.stringify 序列化
  const countMap = new Map<string, number>();

  for (const item of original) {
    const key = JSON.stringify(item);
    countMap.set(key, (countMap.get(key) || 0) + 1);
  }

  for (const item of sorted) {
    const key = JSON.stringify(item);
    const count = countMap.get(key);
    if (!count) {
      return {
        passed: false,
        error: `新增元素：${key}`,
        details: { item },
      };
    }
    countMap.set(key, count - 1);
  }

  for (const [key, count] of countMap) {
    if (count !== 0) {
      return {
        passed: false,
        error: `元素计数不匹配：${key} 差 ${count}`,
      };
    }
  }

  return { passed: true };
}

/**
 * 断言置换正确
 */
export function assertPermutation<T>(
  original: readonly T[],
  sorted: readonly T[],
  message?: string
): void {
  const result = verifyPermutation(original, sorted);
  if (!result.passed) {
    throw new Error(message || result.error);
  }
}

// ============================================================================
// 稳定性校验
// ============================================================================

/**
 * 验证排序的稳定性
 *
 * 对于"相等"的元素，检查它们是否保持原始相对顺序。
 *
 * @param original 原始数组
 * @param sorted 排序后的数组
 * @param cmp 比较函数
 */
export function verifyStable<T>(
  original: readonly T[],
  sorted: readonly T[],
  cmp: Comparator<T>
): VerificationResult {
  // 给原始元素附加索引
  const indexed = original.map((value, originalIndex) => ({ value, originalIndex }));

  // 找到 sorted 中每个元素对应的原始索引
  const usedIndices = new Set<number>();
  const sortedOriginalIndices: number[] = [];

  for (const sortedItem of sorted) {
    let found = false;
    for (const { value, originalIndex } of indexed) {
      if (!usedIndices.has(originalIndex) && cmp(value, sortedItem) === 0) {
        // 进一步检查是否是同一个对象（对于引用类型）
        if (value === sortedItem || JSON.stringify(value) === JSON.stringify(sortedItem)) {
          sortedOriginalIndices.push(originalIndex);
          usedIndices.add(originalIndex);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      return {
        passed: false,
        error: '无法匹配元素',
        details: { sortedItem },
      };
    }
  }

  // 检查相等元素的原始索引是否保持递增
  for (let i = 1; i < sorted.length; i++) {
    if (cmp(sorted[i - 1], sorted[i]) === 0) {
      if (sortedOriginalIndices[i - 1] > sortedOriginalIndices[i]) {
        return {
          passed: false,
          error: `稳定性违反：相等元素 [${i - 1}] 和 [${i}] 顺序颠倒`,
          details: {
            index: i,
            originalIndex1: sortedOriginalIndices[i - 1],
            originalIndex2: sortedOriginalIndices[i],
          },
        };
      }
    }
  }

  return { passed: true };
}

/**
 * 断言排序稳定
 */
export function assertStable<T>(
  original: readonly T[],
  sorted: readonly T[],
  cmp: Comparator<T>,
  message?: string
): void {
  const result = verifyStable(original, sorted, cmp);
  if (!result.passed) {
    throw new Error(message || result.error);
  }
}

// ============================================================================
// 综合校验
// ============================================================================

/**
 * 综合验证排序结果
 *
 * @param original 原始数组
 * @param sorted 排序后数组
 * @param cmp 比较函数
 * @param checkStability 是否检查稳定性
 */
export function verifySort<T>(
  original: readonly T[],
  sorted: readonly T[],
  cmp: Comparator<T>,
  checkStability: boolean = false
): VerificationResult {
  // 1. 检查有序性
  const sortedResult = verifySorted(sorted, cmp);
  if (!sortedResult.passed) {
    return { ...sortedResult, error: `有序性: ${sortedResult.error}` };
  }

  // 2. 检查置换性
  const permResult = verifyPermutation(original, sorted);
  if (!permResult.passed) {
    return { ...permResult, error: `置换性: ${permResult.error}` };
  }

  // 3. 可选：检查稳定性
  if (checkStability) {
    const stableResult = verifyStable(original, sorted, cmp);
    if (!stableResult.passed) {
      return { ...stableResult, error: `稳定性: ${stableResult.error}` };
    }
  }

  return { passed: true };
}

/**
 * 综合断言
 */
export function assertSort<T>(
  original: readonly T[],
  sorted: readonly T[],
  cmp: Comparator<T>,
  checkStability: boolean = false,
  message?: string
): void {
  const result = verifySort(original, sorted, cmp, checkStability);
  if (!result.passed) {
    throw new Error(message || result.error);
  }
}

// ============================================================================
// 导出
// ============================================================================

export default {
  verifySorted,
  assertSorted,
  verifyPermutation,
  assertPermutation,
  verifyStable,
  assertStable,
  verifySort,
  assertSort,
};

