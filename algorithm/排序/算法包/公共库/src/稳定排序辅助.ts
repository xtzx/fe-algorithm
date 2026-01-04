/**
 * 稳定排序辅助模块
 *
 * 提供 Schwartzian Transform（装饰-排序-还原）实现，
 * 无论底层排序算法是否稳定，都能保证输出稳定。
 */

import type { Comparator } from './比较器';

// ============================================================================
// 类型定义
// ============================================================================

/** 装饰后的元素 */
interface DecoratedItem<T> {
  value: T;
  originalIndex: number;
}

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 稳定排序（Schwartzian Transform）
 *
 * 实现原理：
 * 1. 装饰：给每个元素附加原始索引
 * 2. 排序：相等时比较原始索引
 * 3. 还原：去掉装饰，返回原始元素
 *
 * @param arr 原始数组（不修改）
 * @param cmp 比较函数
 * @returns 稳定排序后的新数组
 *
 * @example
 * const users = [
 *   { name: 'Alice', age: 30 },
 *   { name: 'Bob', age: 25 },
 *   { name: 'Charlie', age: 30 },
 * ];
 *
 * const sorted = stableSort(users, (a, b) => a.age - b.age);
 * // 保证：Alice 仍在 Charlie 前面
 */
export function stableSort<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): T[] {
  // 1. 装饰：附加原始索引
  const decorated: DecoratedItem<T>[] = arr.map((value, originalIndex) => ({
    value,
    originalIndex,
  }));

  // 2. 排序：相等时比较原始索引
  decorated.sort((a, b) => {
    const result = cmp(a.value, b.value);
    if (result !== 0) return result;
    return a.originalIndex - b.originalIndex;
  });

  // 3. 还原：提取原始值
  return decorated.map(item => item.value);
}

/**
 * 稳定原地排序
 *
 * 注意：由于需要额外空间存储索引，实际并非真正原地。
 * 但会修改原数组。
 */
export function stableSortInPlace<T>(
  arr: T[],
  cmp: Comparator<T>
): T[] {
  const sorted = stableSort(arr, cmp);
  for (let i = 0; i < arr.length; i++) {
    arr[i] = sorted[i];
  }
  return arr;
}

// ============================================================================
// 多键排序
// ============================================================================

/**
 * 稳定多键排序
 *
 * 从最低优先级键到最高优先级键依次排序。
 * 利用稳定性保证多键排序的正确性。
 *
 * @param arr 原始数组
 * @param comparators 比较器数组（从低优先级到高优先级）
 *
 * @example
 * // 按 score 降序，同分按 name 升序
 * const sorted = stableMultiSort(users, [
 *   (a, b) => a.name.localeCompare(b.name),  // 低优先级
 *   (a, b) => b.score - a.score,             // 高优先级
 * ]);
 */
export function stableMultiSort<T>(
  arr: readonly T[],
  comparators: Comparator<T>[]
): T[] {
  let result = [...arr];

  // 从低优先级到高优先级依次排序
  for (const cmp of comparators) {
    result = stableSort(result, cmp);
  }

  return result;
}

// ============================================================================
// 缓存键排序（性能优化）
// ============================================================================

/**
 * 带键缓存的稳定排序
 *
 * 当比较函数涉及昂贵计算时，可以缓存键值。
 *
 * @param arr 原始数组
 * @param keyFn 键提取函数
 * @param cmp 键比较函数
 *
 * @example
 * // 按字符串长度排序，避免重复计算
 * const sorted = stableSortByKey(
 *   strings,
 *   s => s.length,
 *   (a, b) => a - b
 * );
 */
export function stableSortByKey<T, K>(
  arr: readonly T[],
  keyFn: (item: T) => K,
  cmp: Comparator<K>
): T[] {
  // 装饰：附加键和原始索引
  const decorated = arr.map((value, originalIndex) => ({
    value,
    key: keyFn(value),
    originalIndex,
  }));

  // 排序：先比较键，相等时比较索引
  decorated.sort((a, b) => {
    const result = cmp(a.key, b.key);
    if (result !== 0) return result;
    return a.originalIndex - b.originalIndex;
  });

  // 还原
  return decorated.map(item => item.value);
}

// ============================================================================
// 表格多列排序（实用工具）
// ============================================================================

/** 列排序配置 */
export interface ColumnSort<T> {
  field: keyof T;
  order: 'asc' | 'desc';
}

/**
 * 表格多列稳定排序
 *
 * @param data 表格数据
 * @param columns 列排序配置（从高优先级到低优先级）
 *
 * @example
 * const sorted = tableSort(users, [
 *   { field: 'score', order: 'desc' },  // 高优先级
 *   { field: 'name', order: 'asc' },    // 低优先级
 * ]);
 */
export function tableSort<T extends Record<string, unknown>>(
  data: readonly T[],
  columns: ColumnSort<T>[]
): T[] {
  // 反转顺序，因为 stableMultiSort 期望从低到高
  const comparators = columns.reverse().map(({ field, order }) => {
    return (a: T, b: T): number => {
      const valA = a[field];
      const valB = b[field];

      let result: number;
      if (typeof valA === 'number' && typeof valB === 'number') {
        result = valA - valB;
      } else if (typeof valA === 'string' && typeof valB === 'string') {
        result = valA.localeCompare(valB);
      } else {
        result = String(valA).localeCompare(String(valB));
      }

      return order === 'asc' ? result : -result;
    };
  });

  return stableMultiSort(data, comparators);
}

// ============================================================================
// 导出
// ============================================================================

export default {
  stableSort,
  stableSortInPlace,
  stableMultiSort,
  stableSortByKey,
  tableSort,
};

