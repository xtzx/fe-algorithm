/**
 * 分页与游标实现
 */

import type { PageResult, CursorPageResult } from './数据模型';

// ============================================================================
// 偏移量分页
// ============================================================================

/**
 * 偏移量分页（传统分页）
 *
 * @param data 有序数据
 * @param page 页码（从 1 开始）
 * @param pageSize 每页大小
 */
export function paginateByOffset<T>(
  data: readonly T[],
  page: number,
  pageSize: number
): PageResult<T> {
  const total = data.length;
  const totalPages = Math.ceil(total / pageSize);
  const validPage = Math.max(1, Math.min(page, totalPages || 1));

  const start = (validPage - 1) * pageSize;
  const end = Math.min(start + pageSize, total);

  return {
    items: data.slice(start, end),
    total,
    page: validPage,
    pageSize,
    hasNext: end < total,
    hasPrev: validPage > 1,
  };
}

/**
 * 计算分页信息
 */
export function calculatePagination(
  total: number,
  page: number,
  pageSize: number
): {
  totalPages: number;
  currentPage: number;
  start: number;
  end: number;
  hasNext: boolean;
  hasPrev: boolean;
} {
  const totalPages = Math.ceil(total / pageSize);
  const currentPage = Math.max(1, Math.min(page, totalPages || 1));
  const start = (currentPage - 1) * pageSize;
  const end = Math.min(start + pageSize, total);

  return {
    totalPages,
    currentPage,
    start,
    end,
    hasNext: end < total,
    hasPrev: currentPage > 1,
  };
}

// ============================================================================
// 游标分页
// ============================================================================

/**
 * 游标分页（解决数据变化时的稳定性问题）
 *
 * @param data 有序数据
 * @param cursor 游标（上一页最后一条的 key）
 * @param pageSize 每页大小
 * @param getKey 获取元素的唯一键
 */
export function paginateWithCursor<T>(
  data: readonly T[],
  cursor: string | null,
  pageSize: number,
  getKey: (item: T) => string
): CursorPageResult<T> {
  let startIndex = 0;

  if (cursor) {
    const cursorIndex = data.findIndex(item => getKey(item) === cursor);
    if (cursorIndex >= 0) {
      startIndex = cursorIndex + 1;
    }
    // 游标失效时从头开始
  }

  const items = data.slice(startIndex, startIndex + pageSize);
  const hasMore = startIndex + pageSize < data.length;

  return {
    items,
    nextCursor: items.length > 0 ? getKey(items[items.length - 1]) : null,
    prevCursor: startIndex > 0 ? getKey(data[startIndex - 1]) : null,
    hasMore,
  };
}

/**
 * 反向游标分页（向前翻页）
 */
export function paginateWithCursorBackward<T>(
  data: readonly T[],
  cursor: string | null,
  pageSize: number,
  getKey: (item: T) => string
): CursorPageResult<T> {
  let endIndex = data.length;

  if (cursor) {
    const cursorIndex = data.findIndex(item => getKey(item) === cursor);
    if (cursorIndex >= 0) {
      endIndex = cursorIndex;
    }
  }

  const startIndex = Math.max(0, endIndex - pageSize);
  const items = data.slice(startIndex, endIndex);
  const hasPrev = startIndex > 0;

  return {
    items,
    nextCursor: endIndex < data.length ? getKey(data[endIndex]) : null,
    prevCursor: hasPrev ? getKey(data[startIndex]) : null,
    hasMore: hasPrev,
  };
}

// ============================================================================
// 键集分页（Keyset Pagination）
// ============================================================================

/**
 * 键集分页
 *
 * 使用排序键作为游标，更高效
 */
export function paginateByKeyset<T, K>(
  data: readonly T[],
  afterValue: K | null,
  pageSize: number,
  getValue: (item: T) => K,
  compare: (a: K, b: K) => number
): {
  items: T[];
  nextValue: K | null;
  hasMore: boolean;
} {
  let startIndex = 0;

  if (afterValue !== null) {
    // 找到第一个大于 afterValue 的元素
    startIndex = data.findIndex(item => compare(getValue(item), afterValue) > 0);
    if (startIndex === -1) {
      startIndex = data.length;
    }
  }

  const items = data.slice(startIndex, startIndex + pageSize);
  const hasMore = startIndex + pageSize < data.length;

  return {
    items,
    nextValue: items.length > 0 ? getValue(items[items.length - 1]) : null,
    hasMore,
  };
}

// ============================================================================
// 分页状态管理
// ============================================================================

/**
 * 分页状态管理器
 */
export class PaginationManager<T> {
  private data: T[] = [];
  private pageSize: number;
  private currentPage: number = 1;

  constructor(pageSize: number = 20) {
    this.pageSize = pageSize;
  }

  /**
   * 设置数据
   */
  setData(data: T[]): void {
    this.data = data;
    this.currentPage = 1;
  }

  /**
   * 获取当前页
   */
  getCurrentPage(): PageResult<T> {
    return paginateByOffset(this.data, this.currentPage, this.pageSize);
  }

  /**
   * 下一页
   */
  nextPage(): PageResult<T> | null {
    const result = this.getCurrentPage();
    if (!result.hasNext) return null;

    this.currentPage++;
    return this.getCurrentPage();
  }

  /**
   * 上一页
   */
  prevPage(): PageResult<T> | null {
    if (this.currentPage <= 1) return null;

    this.currentPage--;
    return this.getCurrentPage();
  }

  /**
   * 跳转到指定页
   */
  goToPage(page: number): PageResult<T> {
    const totalPages = Math.ceil(this.data.length / this.pageSize);
    this.currentPage = Math.max(1, Math.min(page, totalPages || 1));
    return this.getCurrentPage();
  }

  /**
   * 修改每页大小
   */
  setPageSize(pageSize: number): void {
    this.pageSize = pageSize;
    this.currentPage = 1;
  }
}

// ============================================================================
// 虚拟滚动支持
// ============================================================================

/**
 * 计算虚拟滚动的可见范围
 */
export function calculateVisibleRange(
  scrollTop: number,
  containerHeight: number,
  itemHeight: number,
  totalItems: number,
  overscan: number = 5
): {
  start: number;
  end: number;
  offsetY: number;
  totalHeight: number;
} {
  const totalHeight = totalItems * itemHeight;
  const visibleStart = Math.floor(scrollTop / itemHeight);
  const visibleEnd = Math.ceil((scrollTop + containerHeight) / itemHeight);

  const start = Math.max(0, visibleStart - overscan);
  const end = Math.min(totalItems, visibleEnd + overscan);
  const offsetY = start * itemHeight;

  return { start, end, offsetY, totalHeight };
}

// ============================================================================
// 导出
// ============================================================================

export default {
  paginateByOffset,
  calculatePagination,
  paginateWithCursor,
  paginateWithCursorBackward,
  paginateByKeyset,
  PaginationManager,
  calculateVisibleRange,
};

