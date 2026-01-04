/**
 * 多列稳定排序实现
 *
 * 方式 A：从次关键字到主关键字依次进行稳定排序
 */

import type { TableRow, SortColumn, SortOrder } from './数据模型';
import { stableSortBy } from '../../../算法包/公共库/src/稳定排序辅助';
import { numberAsc, stringAsc, type Comparator } from '../../../算法包/公共库/src/比较器';

// ============================================================================
// 核心实现
// ============================================================================

/**
 * 多列稳定排序
 *
 * 从最后一列（次要）到第一列（主要）依次排序。
 * 由于使用稳定排序，主关键字相同的行会保持次关键字的排序结果。
 *
 * @param data 待排序数据
 * @param columns 排序列配置（第一个是主关键字）
 * @returns 排序后的新数组
 */
export function sortByMultipleColumns<T extends Record<string, unknown>>(
  data: readonly T[],
  columns: SortColumn<T>[]
): T[] {
  if (columns.length === 0) return [...data];

  let result = [...data];

  // 从后往前排序（最后的是次关键字，先排）
  for (let i = columns.length - 1; i >= 0; i--) {
    const { field, order, type = 'string' } = columns[i];
    const cmp = getComparator(type, order);

    result = stableSortBy(
      result,
      (row) => row[field] as string | number,
      cmp
    );
  }

  return result;
}

/**
 * 单列稳定排序
 */
export function sortByColumn<T extends Record<string, unknown>>(
  data: readonly T[],
  column: SortColumn<T>
): T[] {
  const { field, order, type = 'string' } = column;
  const cmp = getComparator(type, order);

  return stableSortBy(
    [...data],
    (row) => row[field] as string | number,
    cmp
  );
}

// ============================================================================
// 表格状态管理
// ============================================================================

/**
 * 处理列点击（单列排序模式）
 */
export function handleSingleColumnSort<T extends Record<string, unknown>>(
  data: readonly T[],
  currentColumn: SortColumn<T> | null,
  clickedField: keyof T,
  type: SortColumn<T>['type'] = 'string'
): { data: T[]; column: SortColumn<T> | null } {
  let newColumn: SortColumn<T> | null;

  if (currentColumn?.field === clickedField) {
    // 同一列：切换排序方向
    if (currentColumn.order === 'asc') {
      newColumn = { ...currentColumn, order: 'desc' };
    } else {
      // 取消排序
      newColumn = null;
    }
  } else {
    // 不同列：新建升序
    newColumn = { field: clickedField, order: 'asc', type };
  }

  const sortedData = newColumn
    ? sortByColumn(data, newColumn)
    : [...data];

  return { data: sortedData, column: newColumn };
}

/**
 * 处理列点击（多列排序模式，Shift+点击添加）
 */
export function handleMultiColumnSort<T extends Record<string, unknown>>(
  data: readonly T[],
  currentColumns: SortColumn<T>[],
  clickedField: keyof T,
  type: SortColumn<T>['type'] = 'string',
  addToExisting: boolean = false
): { data: T[]; columns: SortColumn<T>[] } {
  const existingIndex = currentColumns.findIndex(c => c.field === clickedField);
  let newColumns: SortColumn<T>[];

  if (existingIndex >= 0) {
    // 已存在的列
    const existing = currentColumns[existingIndex];
    if (existing.order === 'asc') {
      // 切换为降序
      newColumns = currentColumns.map((c, i) =>
        i === existingIndex ? { ...c, order: 'desc' as const } : c
      );
    } else {
      // 移除该列
      newColumns = currentColumns.filter((_, i) => i !== existingIndex);
    }
  } else {
    // 新列
    if (addToExisting) {
      // 添加到现有排序
      newColumns = [...currentColumns, { field: clickedField, order: 'asc', type }];
    } else {
      // 替换现有排序
      newColumns = [{ field: clickedField, order: 'asc', type }];
    }
  }

  const sortedData = sortByMultipleColumns(data, newColumns);
  return { data: sortedData, columns: newColumns };
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 根据类型和方向获取比较器
 */
function getComparator(
  type: SortColumn['type'],
  order: SortOrder
): Comparator<unknown> {
  let baseCmp: Comparator<unknown>;

  switch (type) {
    case 'number':
      baseCmp = (a, b) => numberAsc(a as number, b as number);
      break;
    case 'date':
      baseCmp = (a, b) => {
        const dateA = new Date(a as string).getTime();
        const dateB = new Date(b as string).getTime();
        return numberAsc(dateA, dateB);
      };
      break;
    case 'string':
    default:
      baseCmp = (a, b) => stringAsc(String(a), String(b));
  }

  return order === 'desc'
    ? (a, b) => -baseCmp(a, b)
    : baseCmp;
}

// ============================================================================
// 导出
// ============================================================================

export default {
  sortByMultipleColumns,
  sortByColumn,
  handleSingleColumnSort,
  handleMultiColumnSort,
};

