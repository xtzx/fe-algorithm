/**
 * SortableTable - 可排序表格组件
 *
 * 功能：
 * - 点击表头排序
 * - 支持 Shift+点击 多列排序
 * - 显示排序方向指示器
 * - 加载状态显示
 */

import React, { useState, useCallback, useMemo } from 'react';
import { useSortedData, type SortConfig } from '../hooks/useSortedData';

// ============================================================================
// 类型定义
// ============================================================================

export interface Column<T> {
  /** 数据字段名 */
  key: keyof T;
  /** 表头标题 */
  title: string;
  /** 是否可排序 */
  sortable?: boolean;
  /** 列宽度 */
  width?: number | string;
  /** 自定义渲染 */
  render?: (value: T[keyof T], record: T, index: number) => React.ReactNode;
}

export interface SortableTableProps<T extends Record<string, unknown>> {
  /** 数据源 */
  data: T[];
  /** 列定义 */
  columns: Column<T>[];
  /** 行唯一标识字段 */
  rowKey: keyof T;
  /** 是否使用 Web Worker */
  useWorker?: boolean;
  /** 自定义类名 */
  className?: string;
  /** 空数据提示 */
  emptyText?: string;
  /** 加载中 */
  loading?: boolean;
}

// ============================================================================
// 排序指示器组件
// ============================================================================

interface SortIndicatorProps {
  order: 'asc' | 'desc' | null;
  priority?: number;
}

const SortIndicator: React.FC<SortIndicatorProps> = ({ order, priority }) => {
  const baseStyle: React.CSSProperties = {
    marginLeft: '4px',
    fontSize: '12px',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '2px',
  };

  const inactiveStyle: React.CSSProperties = {
    ...baseStyle,
    color: '#999',
  };

  const activeStyle: React.CSSProperties = {
    ...baseStyle,
    color: '#1890ff',
    fontWeight: 'bold',
  };

  if (!order) {
    return <span style={inactiveStyle}>⇅</span>;
  }

  return (
    <span style={activeStyle}>
      {order === 'asc' ? '↑' : '↓'}
      {priority !== undefined && priority > 0 && (
        <sup style={{ fontSize: '10px', marginLeft: '1px' }}>
          {priority + 1}
        </sup>
      )}
    </span>
  );
};

// ============================================================================
// 主组件
// ============================================================================

export function SortableTable<T extends Record<string, unknown>>({
  data,
  columns,
  rowKey,
  useWorker = false,
  className = '',
  emptyText = '暂无数据',
  loading = false,
}: SortableTableProps<T>): React.ReactElement {
  // 排序状态（支持多列）
  const [sortState, setSortState] = useState<SortConfig<T>[]>([]);

  // 当前主排序配置
  const primarySortConfig = sortState[0] ?? null;

  // 使用排序 Hook
  const { sortedData, isLoading } = useSortedData(data, primarySortConfig, {
    useWorker,
    workerThreshold: 10000,
  });

  // 如果有多列排序，在 useMemo 中处理
  const finalData = useMemo(() => {
    if (sortState.length <= 1) {
      return sortedData;
    }

    // 多列排序
    return [...sortedData].sort((a, b) => {
      for (const { key, order } of sortState) {
        const aVal = a[key];
        const bVal = b[key];

        let result: number;
        if (typeof aVal === 'string' && typeof bVal === 'string') {
          result = aVal.localeCompare(bVal);
        } else if (aVal == null) {
          result = 1;
        } else if (bVal == null) {
          result = -1;
        } else {
          result = (aVal as number) < (bVal as number) ? -1 :
                   (aVal as number) > (bVal as number) ? 1 : 0;
        }

        if (result !== 0) {
          return order === 'asc' ? result : -result;
        }
      }
      return 0;
    });
  }, [sortedData, sortState]);

  // 处理表头点击
  const handleHeaderClick = useCallback(
    (key: keyof T, isMultiSort: boolean) => {
      setSortState(prev => {
        const existingIndex = prev.findIndex(s => s.key === key);

        if (isMultiSort) {
          // Shift + 点击：多列排序
          if (existingIndex >= 0) {
            const existing = prev[existingIndex];
            if (existing.order === 'asc') {
              // asc -> desc
              return prev.map((s, i) =>
                i === existingIndex ? { ...s, order: 'desc' as const } : s
              );
            } else {
              // desc -> 移除
              return prev.filter((_, i) => i !== existingIndex);
            }
          } else {
            // 添加新的排序列
            return [...prev, { key, order: 'asc' as const }];
          }
        } else {
          // 普通点击：单列排序
          if (existingIndex === 0 && prev.length === 1) {
            const existing = prev[0];
            if (existing.order === 'asc') {
              return [{ key, order: 'desc' as const }];
            } else {
              return []; // 取消排序
            }
          }
          return [{ key, order: 'asc' as const }];
        }
      });
    },
    []
  );

  // 获取列的排序状态
  const getSortInfo = useCallback(
    (key: keyof T) => {
      const index = sortState.findIndex(s => s.key === key);
      if (index < 0) return { order: null, priority: undefined };
      return {
        order: sortState[index].order,
        priority: sortState.length > 1 ? index : undefined,
      };
    },
    [sortState]
  );

  // 样式
  const tableStyle: React.CSSProperties = {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '14px',
  };

  const thStyle: React.CSSProperties = {
    padding: '12px 16px',
    textAlign: 'left',
    backgroundColor: '#fafafa',
    borderBottom: '1px solid #e8e8e8',
    fontWeight: 500,
  };

  const thSortableStyle: React.CSSProperties = {
    ...thStyle,
    cursor: 'pointer',
    userSelect: 'none',
    transition: 'background-color 0.2s',
  };

  const tdStyle: React.CSSProperties = {
    padding: '12px 16px',
    borderBottom: '1px solid #e8e8e8',
  };

  const loadingOverlayStyle: React.CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '14px',
    color: '#666',
  };

  const emptyStyle: React.CSSProperties = {
    padding: '40px',
    textAlign: 'center',
    color: '#999',
  };

  const isTableLoading = loading || isLoading;

  return (
    <div style={{ position: 'relative' }} className={className}>
      {isTableLoading && (
        <div style={loadingOverlayStyle}>
          排序中...
        </div>
      )}

      <table style={tableStyle}>
        <thead>
          <tr>
            {columns.map(col => {
              const { order, priority } = getSortInfo(col.key);
              const isSortable = col.sortable !== false;

              return (
                <th
                  key={String(col.key)}
                  style={{
                    ...(isSortable ? thSortableStyle : thStyle),
                    width: col.width,
                  }}
                  onClick={(e) => {
                    if (isSortable) {
                      handleHeaderClick(col.key, e.shiftKey);
                    }
                  }}
                  title={isSortable ? '点击排序，Shift+点击添加排序' : undefined}
                >
                  {col.title}
                  {isSortable && (
                    <SortIndicator order={order} priority={priority} />
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {finalData.length === 0 ? (
            <tr>
              <td colSpan={columns.length} style={emptyStyle}>
                {emptyText}
              </td>
            </tr>
          ) : (
            finalData.map((record, rowIndex) => (
              <tr
                key={String(record[rowKey])}
                style={{
                  backgroundColor: rowIndex % 2 === 0 ? '#fff' : '#fafafa',
                }}
              >
                {columns.map(col => (
                  <td key={String(col.key)} style={tdStyle}>
                    {col.render
                      ? col.render(record[col.key], record, rowIndex)
                      : String(record[col.key] ?? '')}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>

      {sortState.length > 0 && (
        <div style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
          排序: {sortState.map((s, i) => (
            <span key={String(s.key)}>
              {i > 0 && ' → '}
              {columns.find(c => c.key === s.key)?.title ?? String(s.key)}
              ({s.order === 'asc' ? '升序' : '降序'})
            </span>
          ))}
          <button
            onClick={() => setSortState([])}
            style={{
              marginLeft: '8px',
              padding: '2px 8px',
              fontSize: '12px',
              cursor: 'pointer',
              border: '1px solid #d9d9d9',
              borderRadius: '4px',
              backgroundColor: '#fff',
            }}
          >
            清除排序
          </button>
        </div>
      )}
    </div>
  );
}

export default SortableTable;

