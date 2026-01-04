/**
 * 比较器模块
 *
 * 提供符合 Array.prototype.sort() 规范的比较函数。
 * 所有比较器满足：自反性、反对称性、传递性。
 */

// ============================================================================
// 类型定义
// ============================================================================

/** 比较函数类型 */
export type Comparator<T> = (a: T, b: T) => number;

// ============================================================================
// 基础比较器
// ============================================================================

/** 数字升序 */
export const numberAsc: Comparator<number> = (a, b) => {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
};

/** 数字降序 */
export const numberDesc: Comparator<number> = (a, b) => {
  if (a < b) return 1;
  if (a > b) return -1;
  return 0;
};

/** 字符串升序（ASCII 比较） */
export const stringAsc: Comparator<string> = (a, b) => {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
};

/** 字符串降序 */
export const stringDesc: Comparator<string> = (a, b) => {
  if (a < b) return 1;
  if (a > b) return -1;
  return 0;
};

/** 字符串升序（本地化比较，支持中文等） */
export const stringLocaleAsc: Comparator<string> = (a, b) =>
  a.localeCompare(b);

/** 字符串降序（本地化比较） */
export const stringLocaleDesc: Comparator<string> = (a, b) =>
  b.localeCompare(a);

// ============================================================================
// 对象字段比较器
// ============================================================================

/**
 * 按对象字段排序
 *
 * @example
 * users.sort(byField('age', numberAsc));
 */
export function byField<T, K extends keyof T>(
  field: K,
  cmp: Comparator<T[K]>
): Comparator<T> {
  return (a, b) => cmp(a[field], b[field]);
}

/**
 * 按嵌套字段排序
 *
 * @example
 * items.sort(byPath(['user', 'profile', 'age'], numberAsc));
 */
export function byPath<T>(
  path: string[],
  cmp: Comparator<unknown>
): Comparator<T> {
  return (a, b) => {
    let valA: unknown = a;
    let valB: unknown = b;
    for (const key of path) {
      valA = (valA as Record<string, unknown>)?.[key];
      valB = (valB as Record<string, unknown>)?.[key];
    }
    return cmp(valA, valB);
  };
}

// ============================================================================
// 比较器组合
// ============================================================================

/**
 * 组合多个比较器（用于多列排序）
 *
 * 按顺序应用比较器，前一个返回 0 时使用下一个。
 *
 * @example
 * // 先按年龄升序，同龄按姓名升序
 * users.sort(compose(
 *   byField('age', numberAsc),
 *   byField('name', stringAsc)
 * ));
 */
export function compose<T>(...comparators: Comparator<T>[]): Comparator<T> {
  return (a, b) => {
    for (const cmp of comparators) {
      const result = cmp(a, b);
      if (result !== 0) return result;
    }
    return 0;
  };
}

/**
 * 反转比较器
 *
 * @example
 * const desc = reverse(numberAsc);
 */
export function reverse<T>(cmp: Comparator<T>): Comparator<T> {
  return (a, b) => cmp(b, a);
}

/**
 * 处理 null/undefined 的比较器
 *
 * @param cmp 原始比较器
 * @param nullsFirst true 时 null 排在前面
 */
export function nullSafe<T>(
  cmp: Comparator<T>,
  nullsFirst: boolean = true
): Comparator<T | null | undefined> {
  return (a, b) => {
    const aNull = a == null;
    const bNull = b == null;

    if (aNull && bNull) return 0;
    if (aNull) return nullsFirst ? -1 : 1;
    if (bNull) return nullsFirst ? 1 : -1;

    return cmp(a, b);
  };
}

// ============================================================================
// 验证工具
// ============================================================================

/**
 * 验证比较器是否满足数学性质
 *
 * @param arr 测试数组
 * @param cmp 比较器
 * @returns 是否合法
 */
export function validateComparator<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): { valid: boolean; error?: string } {
  // 自反性
  for (const item of arr) {
    if (cmp(item, item) !== 0) {
      return { valid: false, error: '违反自反性' };
    }
  }

  // 反对称性
  for (let i = 0; i < arr.length; i++) {
    for (let j = i + 1; j < arr.length; j++) {
      const ab = cmp(arr[i], arr[j]);
      const ba = cmp(arr[j], arr[i]);
      if (Math.sign(ab) !== -Math.sign(ba)) {
        return { valid: false, error: '违反反对称性' };
      }
    }
  }

  // 传递性检查（采样，完整检查 O(n³) 太慢）
  const sample = arr.slice(0, Math.min(arr.length, 50));
  for (let i = 0; i < sample.length; i++) {
    for (let j = 0; j < sample.length; j++) {
      for (let k = 0; k < sample.length; k++) {
        const ij = cmp(sample[i], sample[j]);
        const jk = cmp(sample[j], sample[k]);
        const ik = cmp(sample[i], sample[k]);

        if (ij < 0 && jk < 0 && ik >= 0) {
          return { valid: false, error: '违反传递性' };
        }
      }
    }
  }

  return { valid: true };
}

// ============================================================================
// 类型导出
// ============================================================================

export default {
  numberAsc,
  numberDesc,
  stringAsc,
  stringDesc,
  stringLocaleAsc,
  stringLocaleDesc,
  byField,
  byPath,
  compose,
  reverse,
  nullSafe,
  validateComparator,
};

