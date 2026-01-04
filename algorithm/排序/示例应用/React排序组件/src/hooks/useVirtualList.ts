/**
 * useVirtualList - 虚拟列表 Hook
 *
 * 功能：
 * - 只渲染可见区域的元素
 * - 支持固定高度和动态高度
 * - 流畅滚动
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';

// ============================================================================
// 类型定义
// ============================================================================

export interface UseVirtualListOptions<T> {
  /** 数据项列表 */
  items: T[];
  /** 每项高度（固定高度模式） */
  itemHeight: number;
  /** 容器高度 */
  containerHeight: number;
  /** 预渲染数量（上下各多渲染几个） */
  overscan?: number;
}

export interface VirtualItem<T> {
  /** 原始数据项 */
  item: T;
  /** 原始索引 */
  index: number;
  /** 定位样式 */
  style: React.CSSProperties;
}

export interface UseVirtualListResult<T> {
  /** 可见项列表 */
  visibleItems: VirtualItem<T>[];
  /** 容器属性 */
  containerProps: {
    ref: React.RefObject<HTMLDivElement>;
    style: React.CSSProperties;
    onScroll: (e: React.UIEvent<HTMLDivElement>) => void;
  };
  /** 内部包装器属性 */
  wrapperProps: {
    style: React.CSSProperties;
  };
  /** 总高度 */
  totalHeight: number;
  /** 当前滚动位置 */
  scrollTop: number;
  /** 滚动到指定索引 */
  scrollToIndex: (index: number, align?: 'start' | 'center' | 'end') => void;
}

// ============================================================================
// Hook 实现
// ============================================================================

export function useVirtualList<T>(
  options: UseVirtualListOptions<T>
): UseVirtualListResult<T> {
  const { items, itemHeight, containerHeight, overscan = 3 } = options;

  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);

  // 计算总高度
  const totalHeight = items.length * itemHeight;

  // 计算可见范围
  const { startIndex, endIndex, visibleItems } = useMemo(() => {
    // 可见区域的起始索引
    const start = Math.floor(scrollTop / itemHeight);
    // 可见数量
    const visibleCount = Math.ceil(containerHeight / itemHeight);
    // 结束索引
    const end = Math.min(start + visibleCount, items.length - 1);

    // 加上 overscan
    const startWithOverscan = Math.max(0, start - overscan);
    const endWithOverscan = Math.min(items.length - 1, end + overscan);

    // 生成可见项
    const visible: VirtualItem<T>[] = [];
    for (let i = startWithOverscan; i <= endWithOverscan; i++) {
      visible.push({
        item: items[i],
        index: i,
        style: {
          position: 'absolute',
          top: i * itemHeight,
          left: 0,
          right: 0,
          height: itemHeight,
        },
      });
    }

    return {
      startIndex: startWithOverscan,
      endIndex: endWithOverscan,
      visibleItems: visible,
    };
  }, [items, itemHeight, containerHeight, scrollTop, overscan]);

  // 滚动处理
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    setScrollTop(target.scrollTop);
  }, []);

  // 滚动到指定索引
  const scrollToIndex = useCallback(
    (index: number, align: 'start' | 'center' | 'end' = 'start') => {
      if (!containerRef.current) return;

      let targetScrollTop: number;

      switch (align) {
        case 'start':
          targetScrollTop = index * itemHeight;
          break;
        case 'center':
          targetScrollTop = index * itemHeight - containerHeight / 2 + itemHeight / 2;
          break;
        case 'end':
          targetScrollTop = index * itemHeight - containerHeight + itemHeight;
          break;
      }

      // 限制范围
      targetScrollTop = Math.max(0, Math.min(targetScrollTop, totalHeight - containerHeight));

      containerRef.current.scrollTop = targetScrollTop;
    },
    [itemHeight, containerHeight, totalHeight]
  );

  return {
    visibleItems,
    containerProps: {
      ref: containerRef,
      style: {
        height: containerHeight,
        overflow: 'auto',
        position: 'relative',
      },
      onScroll: handleScroll,
    },
    wrapperProps: {
      style: {
        height: totalHeight,
        position: 'relative',
      },
    },
    totalHeight,
    scrollTop,
    scrollToIndex,
  };
}

// ============================================================================
// 动态高度版本（进阶）
// ============================================================================

export interface UseDynamicVirtualListOptions<T> {
  items: T[];
  estimatedItemHeight: number;
  containerHeight: number;
  overscan?: number;
  getItemHeight?: (item: T, index: number) => number;
}

export function useDynamicVirtualList<T>(
  options: UseDynamicVirtualListOptions<T>
): UseVirtualListResult<T> {
  const {
    items,
    estimatedItemHeight,
    containerHeight,
    overscan = 3,
    getItemHeight,
  } = options;

  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);

  // 缓存每项的位置信息
  const measureCache = useRef<Map<number, { offset: number; height: number }>>(new Map());

  // 计算所有项的位置
  const { totalHeight, itemPositions } = useMemo(() => {
    const positions: { offset: number; height: number }[] = [];
    let offset = 0;

    for (let i = 0; i < items.length; i++) {
      const cached = measureCache.current.get(i);
      const height = cached?.height ??
        (getItemHeight ? getItemHeight(items[i], i) : estimatedItemHeight);

      positions.push({ offset, height });
      offset += height;
    }

    return { totalHeight: offset, itemPositions: positions };
  }, [items, estimatedItemHeight, getItemHeight]);

  // 二分查找起始索引
  const findStartIndex = useCallback(
    (scrollTop: number): number => {
      let low = 0;
      let high = itemPositions.length - 1;

      while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const { offset, height } = itemPositions[mid];

        if (offset + height < scrollTop) {
          low = mid + 1;
        } else if (offset > scrollTop) {
          high = mid - 1;
        } else {
          return mid;
        }
      }

      return Math.max(0, low);
    },
    [itemPositions]
  );

  // 计算可见项
  const visibleItems = useMemo(() => {
    if (itemPositions.length === 0) return [];

    const startIndex = Math.max(0, findStartIndex(scrollTop) - overscan);

    let endIndex = startIndex;
    let accumulatedHeight = itemPositions[startIndex]?.offset ?? 0;

    while (endIndex < items.length && accumulatedHeight < scrollTop + containerHeight) {
      accumulatedHeight += itemPositions[endIndex].height;
      endIndex++;
    }

    endIndex = Math.min(items.length - 1, endIndex + overscan);

    const visible: VirtualItem<T>[] = [];
    for (let i = startIndex; i <= endIndex; i++) {
      const { offset, height } = itemPositions[i];
      visible.push({
        item: items[i],
        index: i,
        style: {
          position: 'absolute',
          top: offset,
          left: 0,
          right: 0,
          height,
        },
      });
    }

    return visible;
  }, [items, itemPositions, scrollTop, containerHeight, overscan, findStartIndex]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  const scrollToIndex = useCallback(
    (index: number, align: 'start' | 'center' | 'end' = 'start') => {
      if (!containerRef.current || !itemPositions[index]) return;

      const { offset, height } = itemPositions[index];
      let targetScrollTop: number;

      switch (align) {
        case 'start':
          targetScrollTop = offset;
          break;
        case 'center':
          targetScrollTop = offset - containerHeight / 2 + height / 2;
          break;
        case 'end':
          targetScrollTop = offset - containerHeight + height;
          break;
      }

      targetScrollTop = Math.max(0, Math.min(targetScrollTop, totalHeight - containerHeight));
      containerRef.current.scrollTop = targetScrollTop;
    },
    [itemPositions, containerHeight, totalHeight]
  );

  return {
    visibleItems,
    containerProps: {
      ref: containerRef,
      style: {
        height: containerHeight,
        overflow: 'auto',
        position: 'relative',
      },
      onScroll: handleScroll,
    },
    wrapperProps: {
      style: {
        height: totalHeight,
        position: 'relative',
      },
    },
    totalHeight,
    scrollTop,
    scrollToIndex,
  };
}

