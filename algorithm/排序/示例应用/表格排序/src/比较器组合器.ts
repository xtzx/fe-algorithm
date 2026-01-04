/**
 * 比较器组合器实现
 *
 * 方式 B：使用 compose 组合多个比较器，一次排序完成
 */

import type { SortColumn, SortOrder, FieldType } from './数据模型';
import {
  compose,
  byField,
  reverse,
  nullSafe,
  numberAsc,
  stringAsc,
  type Comparator,
} from '../../../算法包/公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 创建表格多列比较器
 *
 * @param columns 排序列配置（按优先级排列，第一个最重要）
 * @returns 组合后的比较器
 */
export function createTableComparator<T extends Record<string, unknown>>(
  columns: SortColumn<T>[]
): Comparator<T> {
  if (columns.length === 0) {
    return () => 0; // 无排序
  }

  const comparators = columns.map(({ field, order, type = 'string' }) => {
    const fieldCmp = createFieldComparator<T>(field, type);
    return order === 'desc' ? reverse(fieldCmp) : fieldCmp;
  });

  return compose(...comparators);
}

/**
 * 创建单字段比较器
 */
export function createFieldComparator<T extends Record<string, unknown>>(
  field: keyof T,
  type: FieldType = 'string'
): Comparator<T> {
  switch (type) {
    case 'number':
      return byField(field, nullSafe(numberAsc as Comparator<T[keyof T]>));

    case 'date':
      return (a, b) => {
        const dateA = new Date(a[field] as string).getTime();
        const dateB = new Date(b[field] as string).getTime();
        if (isNaN(dateA) && isNaN(dateB)) return 0;
        if (isNaN(dateA)) return 1;
        if (isNaN(dateB)) return -1;
        return numberAsc(dateA, dateB);
      };

    case 'string':
    default:
      return byField(field, nullSafe((a, b) => stringAsc(String(a), String(b))));
  }
}

// ============================================================================
// 动态比较器构建器
// ============================================================================

/**
 * 表格比较器构建器
 *
 * 支持链式调用，动态添加排序条件
 */
export class TableComparatorBuilder<T extends Record<string, unknown>> {
  private columns: SortColumn<T>[] = [];

  /**
   * 添加排序列
   */
  addColumn(field: keyof T, order: SortOrder = 'asc', type: FieldType = 'string'): this {
    this.columns.push({ field, order, type });
    return this;
  }

  /**
   * 移除排序列
   */
  removeColumn(field: keyof T): this {
    this.columns = this.columns.filter(c => c.field !== field);
    return this;
  }

  /**
   * 切换排序方向
   */
  toggleOrder(field: keyof T): this {
    const column = this.columns.find(c => c.field === field);
    if (column) {
      column.order = column.order === 'asc' ? 'desc' : 'asc';
    }
    return this;
  }

  /**
   * 清空所有排序列
   */
  clear(): this {
    this.columns = [];
    return this;
  }

  /**
   * 获取当前排序列配置
   */
  getColumns(): SortColumn<T>[] {
    return [...this.columns];
  }

  /**
   * 构建比较器
   */
  build(): Comparator<T> {
    return createTableComparator(this.columns);
  }

  /**
   * 排序数据
   */
  sort(data: readonly T[]): T[] {
    return [...data].sort(this.build());
  }
}

// ============================================================================
// 预设比较器
// ============================================================================

/** 数字升序 */
export const numAsc = <T extends Record<string, unknown>>(field: keyof T) =>
  createFieldComparator<T>(field, 'number');

/** 数字降序 */
export const numDesc = <T extends Record<string, unknown>>(field: keyof T) =>
  reverse(createFieldComparator<T>(field, 'number'));

/** 字符串升序 */
export const strAsc = <T extends Record<string, unknown>>(field: keyof T) =>
  createFieldComparator<T>(field, 'string');

/** 字符串降序 */
export const strDesc = <T extends Record<string, unknown>>(field: keyof T) =>
  reverse(createFieldComparator<T>(field, 'string'));

/** 日期升序 */
export const dateAsc = <T extends Record<string, unknown>>(field: keyof T) =>
  createFieldComparator<T>(field, 'date');

/** 日期降序 */
export const dateDesc = <T extends Record<string, unknown>>(field: keyof T) =>
  reverse(createFieldComparator<T>(field, 'date'));

// ============================================================================
// 便捷排序函数
// ============================================================================

/**
 * 使用比较器组合进行多列排序
 */
export function sortByComparator<T extends Record<string, unknown>>(
  data: readonly T[],
  columns: SortColumn<T>[]
): T[] {
  return [...data].sort(createTableComparator(columns));
}

/**
 * 快速创建排序函数
 */
export function createSortFunction<T extends Record<string, unknown>>(
  columns: SortColumn<T>[]
): (data: readonly T[]) => T[] {
  const comparator = createTableComparator(columns);
  return (data) => [...data].sort(comparator);
}

// ============================================================================
// 导出
// ============================================================================

export default {
  createTableComparator,
  createFieldComparator,
  TableComparatorBuilder,
  sortByComparator,
  createSortFunction,
  numAsc,
  numDesc,
  strAsc,
  strDesc,
  dateAsc,
  dateDesc,
};

