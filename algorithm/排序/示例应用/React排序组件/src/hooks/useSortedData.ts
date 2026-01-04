/**
 * useSortedData - 排序数据 Hook
 *
 * 功能：
 * - useMemo 缓存排序结果
 * - 支持 Web Worker 后台排序
 * - 自动根据数据量选择最优策略
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';

// ============================================================================
// 类型定义
// ============================================================================

export interface SortConfig<T> {
  key: keyof T;
  order: 'asc' | 'desc';
}

export interface UseSortedDataOptions {
  /** 是否启用 Web Worker */
  useWorker?: boolean;
  /** 超过此数量时使用 Worker */
  workerThreshold?: number;
}

export interface UseSortedDataResult<T> {
  /** 排序后的数据 */
  sortedData: T[];
  /** 是否正在排序 */
  isLoading: boolean;
  /** 排序耗时（毫秒） */
  sortTime: number;
}

// ============================================================================
// Worker 代码（内联）
// ============================================================================

const workerCode = `
self.onmessage = function(e) {
  const { id, data, sortKey, sortOrder } = e.data;
  const startTime = performance.now();

  const sorted = [...data].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];

    let result;
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      result = aVal.localeCompare(bVal);
    } else if (aVal == null) {
      result = 1;
    } else if (bVal == null) {
      result = -1;
    } else {
      result = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    }

    return sortOrder === 'asc' ? result : -result;
  });

  const sortTime = performance.now() - startTime;
  self.postMessage({ id, sorted, sortTime });
};
`;

// ============================================================================
// 创建 Worker
// ============================================================================

function createSortWorker(): Worker | null {
  if (typeof window === 'undefined' || !window.Worker) {
    return null;
  }

  try {
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    URL.revokeObjectURL(url);
    return worker;
  } catch {
    console.warn('Failed to create Web Worker');
    return null;
  }
}

// ============================================================================
// Hook 实现
// ============================================================================

export function useSortedData<T extends Record<string, unknown>>(
  data: T[],
  sortConfig: SortConfig<T> | null,
  options: UseSortedDataOptions = {}
): UseSortedDataResult<T> {
  const { useWorker = false, workerThreshold = 10000 } = options;

  const [workerSortedData, setWorkerSortedData] = useState<T[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sortTime, setSortTime] = useState(0);

  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const isUsingWorker = useRef(false);

  // 决定是否使用 Worker
  const shouldUseWorker = useWorker && data.length > workerThreshold;

  // 同步排序（使用 useMemo）
  const memoizedSortedData = useMemo(() => {
    if (shouldUseWorker || !sortConfig) {
      return data;
    }

    const startTime = performance.now();

    const sorted = [...data].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];

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

      return sortConfig.order === 'asc' ? result : -result;
    });

    // 注意：不能在 useMemo 里 setState，这里只返回数据
    return sorted;
  }, [data, sortConfig, shouldUseWorker]);

  // 计算同步排序耗时
  useEffect(() => {
    if (!shouldUseWorker && sortConfig) {
      const startTime = performance.now();
      // 重新执行一次排序来计时（实际上 useMemo 已经缓存了结果）
      [...data].sort((a, b) => {
        const aVal = a[sortConfig.key];
        const bVal = b[sortConfig.key];
        if (typeof aVal === 'string' && typeof bVal === 'string') {
          return sortConfig.order === 'asc'
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal);
        }
        const diff = (aVal as number) - (bVal as number);
        return sortConfig.order === 'asc' ? diff : -diff;
      });
      setSortTime(performance.now() - startTime);
    }
  }, [data, sortConfig, shouldUseWorker]);

  // Worker 排序
  useEffect(() => {
    if (!shouldUseWorker) {
      isUsingWorker.current = false;
      return;
    }

    if (!sortConfig) {
      setWorkerSortedData(data);
      setIsLoading(false);
      return;
    }

    isUsingWorker.current = true;
    setIsLoading(true);

    // 创建或复用 Worker
    if (!workerRef.current) {
      workerRef.current = createSortWorker();
    }

    const worker = workerRef.current;
    if (!worker) {
      // Worker 创建失败，回退到同步排序
      console.warn('Web Worker unavailable, falling back to sync sort');
      isUsingWorker.current = false;
      setIsLoading(false);
      return;
    }

    const currentId = ++requestIdRef.current;

    worker.onmessage = (e: MessageEvent) => {
      const { id, sorted, sortTime: workerSortTime } = e.data;

      // 只处理最新请求的响应
      if (id === requestIdRef.current) {
        setWorkerSortedData(sorted);
        setSortTime(workerSortTime);
        setIsLoading(false);
      }
    };

    worker.onerror = (error) => {
      console.error('Worker error:', error);
      setIsLoading(false);
    };

    worker.postMessage({
      id: currentId,
      data,
      sortKey: sortConfig.key,
      sortOrder: sortConfig.order,
    });

    return () => {
      // 不在这里终止 Worker，因为可能有其他排序请求
    };
  }, [data, sortConfig, shouldUseWorker]);

  // 组件卸载时清理 Worker
  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  // 返回正确的数据源
  const sortedData = isUsingWorker.current ? workerSortedData : memoizedSortedData;

  return {
    sortedData: sortConfig ? sortedData : data,
    isLoading,
    sortTime,
  };
}

// ============================================================================
// 多列排序 Hook
// ============================================================================

export interface MultiSortConfig<T> {
  columns: Array<{
    key: keyof T;
    order: 'asc' | 'desc';
  }>;
}

export function useMultiSortedData<T extends Record<string, unknown>>(
  data: T[],
  sortConfig: MultiSortConfig<T> | null
): T[] {
  return useMemo(() => {
    if (!sortConfig || sortConfig.columns.length === 0) {
      return data;
    }

    return [...data].sort((a, b) => {
      for (const { key, order } of sortConfig.columns) {
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
  }, [data, sortConfig]);
}

// ============================================================================
// 排序状态管理 Hook
// ============================================================================

export interface UseSortStateResult<T> {
  sortConfig: SortConfig<T> | null;
  handleSort: (key: keyof T) => void;
  clearSort: () => void;
}

export function useSortState<T>(): UseSortStateResult<T> {
  const [sortConfig, setSortConfig] = useState<SortConfig<T> | null>(null);

  const handleSort = useCallback((key: keyof T) => {
    setSortConfig(prev => {
      if (prev?.key === key) {
        // 切换顺序：asc -> desc -> null
        if (prev.order === 'asc') {
          return { key, order: 'desc' };
        }
        return null; // 取消排序
      }
      return { key, order: 'asc' };
    });
  }, []);

  const clearSort = useCallback(() => {
    setSortConfig(null);
  }, []);

  return { sortConfig, handleSort, clearSort };
}

