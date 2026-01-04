/**
 * Web Worker 排序工具
 *
 * 提供创建和管理排序 Worker 的功能
 */

// ============================================================================
// 类型定义
// ============================================================================

export interface SortWorkerMessage<T> {
  id: string | number;
  data: T[];
  sortKey: string;
  sortOrder: 'asc' | 'desc';
}

export interface SortWorkerResult<T> {
  id: string | number;
  sorted: T[];
  sortTime: number;
}

// ============================================================================
// Worker 代码
// ============================================================================

const WORKER_CODE = `
// Web Worker 排序实现
self.onmessage = function(e) {
  const { id, data, sortKey, sortOrder } = e.data;
  const startTime = performance.now();

  // 排序
  const sorted = [...data].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];

    let result;

    // 处理 null/undefined
    if (aVal == null && bVal == null) return 0;
    if (aVal == null) return 1;
    if (bVal == null) return -1;

    // 字符串比较
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      result = aVal.localeCompare(bVal);
    }
    // 数值比较
    else if (typeof aVal === 'number' && typeof bVal === 'number') {
      result = aVal - bVal;
    }
    // 日期比较
    else if (aVal instanceof Date && bVal instanceof Date) {
      result = aVal.getTime() - bVal.getTime();
    }
    // 其他类型转字符串比较
    else {
      result = String(aVal).localeCompare(String(bVal));
    }

    return sortOrder === 'asc' ? result : -result;
  });

  const sortTime = performance.now() - startTime;

  self.postMessage({ id, sorted, sortTime });
};
`;

// ============================================================================
// Worker 管理器
// ============================================================================

export class SortWorkerManager {
  private worker: Worker | null = null;
  private requestId = 0;
  private pendingRequests = new Map<
    number,
    {
      resolve: (result: SortWorkerResult<unknown>) => void;
      reject: (error: Error) => void;
    }
  >();

  /**
   * 初始化 Worker
   */
  init(): boolean {
    if (typeof window === 'undefined' || !window.Worker) {
      console.warn('Web Worker not supported');
      return false;
    }

    if (this.worker) {
      return true;
    }

    try {
      const blob = new Blob([WORKER_CODE], { type: 'application/javascript' });
      const url = URL.createObjectURL(blob);
      this.worker = new Worker(url);
      URL.revokeObjectURL(url);

      this.worker.onmessage = (e: MessageEvent<SortWorkerResult<unknown>>) => {
        const { id, sorted, sortTime } = e.data;
        const pending = this.pendingRequests.get(id as number);
        if (pending) {
          pending.resolve({ id, sorted, sortTime });
          this.pendingRequests.delete(id as number);
        }
      };

      this.worker.onerror = (error) => {
        console.error('Worker error:', error);
        // 拒绝所有待处理的请求
        this.pendingRequests.forEach(({ reject }) => {
          reject(new Error('Worker error'));
        });
        this.pendingRequests.clear();
      };

      return true;
    } catch (error) {
      console.error('Failed to create worker:', error);
      return false;
    }
  }

  /**
   * 执行排序
   */
  sort<T>(
    data: T[],
    sortKey: string,
    sortOrder: 'asc' | 'desc'
  ): Promise<SortWorkerResult<T>> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        if (!this.init()) {
          reject(new Error('Worker initialization failed'));
          return;
        }
      }

      const id = ++this.requestId;

      this.pendingRequests.set(id, {
        resolve: resolve as (result: SortWorkerResult<unknown>) => void,
        reject,
      });

      this.worker!.postMessage({
        id,
        data,
        sortKey,
        sortOrder,
      });
    });
  }

  /**
   * 终止 Worker
   */
  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.pendingRequests.clear();
  }
}

// ============================================================================
// 单例实例
// ============================================================================

let globalWorkerManager: SortWorkerManager | null = null;

export function getSortWorkerManager(): SortWorkerManager {
  if (!globalWorkerManager) {
    globalWorkerManager = new SortWorkerManager();
  }
  return globalWorkerManager;
}

// ============================================================================
// 便捷函数
// ============================================================================

/**
 * 使用 Worker 排序数据
 */
export async function sortWithWorker<T>(
  data: T[],
  sortKey: keyof T,
  sortOrder: 'asc' | 'desc' = 'asc'
): Promise<{ sorted: T[]; sortTime: number }> {
  const manager = getSortWorkerManager();
  const result = await manager.sort(data, String(sortKey), sortOrder);
  return {
    sorted: result.sorted as T[],
    sortTime: result.sortTime,
  };
}

/**
 * 清理全局 Worker
 */
export function terminateGlobalWorker(): void {
  if (globalWorkerManager) {
    globalWorkerManager.terminate();
    globalWorkerManager = null;
  }
}

