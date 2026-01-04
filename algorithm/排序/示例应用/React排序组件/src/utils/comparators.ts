/**
 * comparators - 比较器工具函数
 *
 * 提供常用的比较器和比较器组合工具
 */

// ============================================================================
// 类型定义
// ============================================================================

export type Comparator<T> = (a: T, b: T) => number;

export type SortOrder = 'asc' | 'desc';

// ============================================================================
// 基础比较器
// ============================================================================

/**
 * 数值比较器
 */
export const numberComparator: Comparator<number> = (a, b) => a - b;

/**
 * 字符串比较器（区分大小写）
 */
export const stringComparator: Comparator<string> = (a, b) => a.localeCompare(b);

/**
 * 字符串比较器（不区分大小写）
 */
export const stringCaseInsensitiveComparator: Comparator<string> = (a, b) =>
  a.toLowerCase().localeCompare(b.toLowerCase());

/**
 * 日期比较器
 */
export const dateComparator: Comparator<Date> = (a, b) => a.getTime() - b.getTime();

/**
 * 布尔比较器（true 在前）
 */
export const booleanComparator: Comparator<boolean> = (a, b) =>
  a === b ? 0 : a ? -1 : 1;

// ============================================================================
// 比较器工厂
// ============================================================================

/**
 * 创建对象字段比较器
 */
export function byKey<T, K extends keyof T>(
  key: K,
  comparator?: Comparator<T[K]>
): Comparator<T> {
  return (a, b) => {
    const aVal = a[key];
    const bVal = b[key];

    if (comparator) {
      return comparator(aVal, bVal);
    }

    // 自动推断类型
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return aVal - bVal;
    }
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return aVal.localeCompare(bVal);
    }
    if (aVal instanceof Date && bVal instanceof Date) {
      return aVal.getTime() - bVal.getTime();
    }

    // 回退到字符串比较
    return String(aVal).localeCompare(String(bVal));
  };
}

/**
 * 反转比较器（降序）
 */
export function reverse<T>(comparator: Comparator<T>): Comparator<T> {
  return (a, b) => -comparator(a, b);
}

/**
 * 创建带方向的比较器
 */
export function withOrder<T>(
  comparator: Comparator<T>,
  order: SortOrder
): Comparator<T> {
  return order === 'asc' ? comparator : reverse(comparator);
}

/**
 * 组合多个比较器（多列排序）
 */
export function compose<T>(...comparators: Comparator<T>[]): Comparator<T> {
  return (a, b) => {
    for (const comparator of comparators) {
      const result = comparator(a, b);
      if (result !== 0) {
        return result;
      }
    }
    return 0;
  };
}

/**
 * 处理 null/undefined 的比较器包装
 */
export function nullsLast<T>(comparator: Comparator<T>): Comparator<T | null | undefined> {
  return (a, b) => {
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    return comparator(a, b);
  };
}

export function nullsFirst<T>(comparator: Comparator<T>): Comparator<T | null | undefined> {
  return (a, b) => {
    if (a == null && b == null) return 0;
    if (a == null) return -1;
    if (b == null) return 1;
    return comparator(a, b);
  };
}

// ============================================================================
// 特殊比较器
// ============================================================================

/**
 * 自然排序比较器（数字字符串按数值排序）
 *
 * 例如: ['item1', 'item2', 'item10'] 按 1, 2, 10 排序
 */
export const naturalComparator: Comparator<string> = (a, b) => {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
};

/**
 * 版本号比较器
 *
 * 例如: '1.2.3' vs '1.10.1'
 */
export const versionComparator: Comparator<string> = (a, b) => {
  const aParts = a.split('.').map(Number);
  const bParts = b.split('.').map(Number);
  const maxLength = Math.max(aParts.length, bParts.length);

  for (let i = 0; i < maxLength; i++) {
    const aPart = aParts[i] ?? 0;
    const bPart = bParts[i] ?? 0;
    if (aPart !== bPart) {
      return aPart - bPart;
    }
  }

  return 0;
};

// ============================================================================
// 工具函数
// ============================================================================

/**
 * 创建排序配置的比较器
 */
export interface SortConfig<T> {
  key: keyof T;
  order: SortOrder;
}

export function createComparator<T extends Record<string, unknown>>(
  config: SortConfig<T> | SortConfig<T>[]
): Comparator<T> {
  const configs = Array.isArray(config) ? config : [config];

  const comparators = configs.map(({ key, order }) => {
    const baseComparator = byKey<T, keyof T>(key);
    return withOrder(baseComparator, order);
  });

  return compose(...comparators);
}

/**
 * 验证比较器的一致性
 *
 * 比较器必须满足以下属性：
 * 1. 反对称性: compare(a, b) = -compare(b, a)
 * 2. 传递性: compare(a, b) > 0 && compare(b, c) > 0 => compare(a, c) > 0
 */
export function validateComparator<T>(
  comparator: Comparator<T>,
  samples: T[]
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // 检查反对称性
  for (let i = 0; i < samples.length; i++) {
    for (let j = i + 1; j < samples.length; j++) {
      const ab = comparator(samples[i], samples[j]);
      const ba = comparator(samples[j], samples[i]);

      if (Math.sign(ab) !== -Math.sign(ba) && ab !== 0 && ba !== 0) {
        errors.push(
          `Antisymmetry violation: compare(${i}, ${j}) = ${ab}, compare(${j}, ${i}) = ${ba}`
        );
      }
    }
  }

  // 检查传递性（简化检测）
  for (let i = 0; i < samples.length; i++) {
    for (let j = 0; j < samples.length; j++) {
      for (let k = 0; k < samples.length; k++) {
        const ab = comparator(samples[i], samples[j]);
        const bc = comparator(samples[j], samples[k]);
        const ac = comparator(samples[i], samples[k]);

        if (ab > 0 && bc > 0 && ac <= 0) {
          errors.push(
            `Transitivity violation at indices ${i}, ${j}, ${k}`
          );
        }
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

