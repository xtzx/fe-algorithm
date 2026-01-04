/**
 * VirtualSortedList - 虚拟滚动排序列表
 *
 * 功能：
 * - 虚拟滚动，只渲染可见元素
 * - 支持排序
 * - 支持大数据量
 */

import React, { useMemo } from 'react';
import { useVirtualList, type UseVirtualListOptions } from '../hooks/useVirtualList';
import { useSortedData, type SortConfig } from '../hooks/useSortedData';

// ============================================================================
// 类型定义
// ============================================================================

export interface VirtualSortedListProps<T extends Record<string, unknown>> {
  /** 数据源 */
  data: T[];
  /** 每项高度 */
  itemHeight: number;
  /** 容器高度 */
  containerHeight: number;
  /** 排序配置 */
  sortConfig?: SortConfig<T> | null;
  /** 渲染每一项 */
  renderItem: (item: T, index: number) => React.ReactNode;
  /** 是否使用 Web Worker */
  useWorker?: boolean;
  /** 自定义类名 */
  className?: string;
  /** 空数据提示 */
  emptyText?: string;
}

// ============================================================================
// 主组件
// ============================================================================

export function VirtualSortedList<T extends Record<string, unknown>>({
  data,
  itemHeight,
  containerHeight,
  sortConfig = null,
  renderItem,
  useWorker = false,
  className = '',
  emptyText = '暂无数据',
}: VirtualSortedListProps<T>): React.ReactElement {
  // 1. 先排序
  const { sortedData, isLoading } = useSortedData(data, sortConfig, {
    useWorker,
    workerThreshold: 10000,
  });

  // 2. 再虚拟化
  const {
    visibleItems,
    containerProps,
    wrapperProps,
    totalHeight,
    scrollTop,
    scrollToIndex,
  } = useVirtualList({
    items: sortedData,
    itemHeight,
    containerHeight,
    overscan: 5,
  });

  // 样式
  const containerStyle: React.CSSProperties = {
    ...containerProps.style,
    border: '1px solid #e8e8e8',
    borderRadius: '4px',
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
    zIndex: 1,
  };

  const emptyStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: containerHeight,
    color: '#999',
    fontSize: '14px',
  };

  const itemStyle: React.CSSProperties = {
    boxSizing: 'border-box',
    borderBottom: '1px solid #f0f0f0',
  };

  if (sortedData.length === 0) {
    return (
      <div className={className} style={emptyStyle}>
        {emptyText}
      </div>
    );
  }

  return (
    <div className={className} style={{ position: 'relative' }}>
      {isLoading && (
        <div style={loadingOverlayStyle}>
          排序中...
        </div>
      )}

      <div {...containerProps} style={containerStyle}>
        <div {...wrapperProps}>
          {visibleItems.map(({ item, index, style }) => (
            <div
              key={index}
              style={{
                ...style,
                ...itemStyle,
              }}
            >
              {renderItem(item, index)}
            </div>
          ))}
        </div>
      </div>

      {/* 调试信息（生产环境可移除） */}
      <div style={{ marginTop: '8px', fontSize: '12px', color: '#999' }}>
        总数据: {sortedData.length} |
        渲染数: {visibleItems.length} |
        滚动位置: {Math.round(scrollTop)}px
      </div>
    </div>
  );
}

// ============================================================================
// 带排序控制的增强版本
// ============================================================================

export interface VirtualSortedListWithControlsProps<T extends Record<string, unknown>>
  extends Omit<VirtualSortedListProps<T>, 'sortConfig'> {
  /** 可排序的字段 */
  sortableKeys: Array<{
    key: keyof T;
    label: string;
  }>;
}

export function VirtualSortedListWithControls<T extends Record<string, unknown>>({
  sortableKeys,
  ...props
}: VirtualSortedListWithControlsProps<T>): React.ReactElement {
  const [sortConfig, setSortConfig] = React.useState<SortConfig<T> | null>(null);

  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (!value) {
      setSortConfig(null);
      return;
    }

    const [key, order] = value.split(':') as [keyof T, 'asc' | 'desc'];
    setSortConfig({ key, order });
  };

  const selectValue = sortConfig
    ? `${String(sortConfig.key)}:${sortConfig.order}`
    : '';

  const controlStyle: React.CSSProperties = {
    marginBottom: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const selectStyle: React.CSSProperties = {
    padding: '6px 12px',
    borderRadius: '4px',
    border: '1px solid #d9d9d9',
    fontSize: '14px',
  };

  return (
    <div>
      <div style={controlStyle}>
        <span>排序:</span>
        <select
          value={selectValue}
          onChange={handleSortChange}
          style={selectStyle}
        >
          <option value="">不排序</option>
          {sortableKeys.map(({ key, label }) => (
            <React.Fragment key={String(key)}>
              <option value={`${String(key)}:asc`}>{label} ↑</option>
              <option value={`${String(key)}:desc`}>{label} ↓</option>
            </React.Fragment>
          ))}
        </select>
      </div>

      <VirtualSortedList
        {...props}
        sortConfig={sortConfig}
      />
    </div>
  );
}

export default VirtualSortedList;

