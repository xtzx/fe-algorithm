/**
 * Worker 排序实现
 *
 * 这个文件可以作为独立的 Worker 文件使用，
 * 也可以参考其逻辑用于内联 Worker 代码
 */

// ============================================================================
// 类型定义
// ============================================================================

interface SortMessage<T> {
  id: number;
  data: T[];
  comparatorStr: string;
}

interface SortResult<T> {
  id: number;
  sorted: T[];
  sortTime: number;
}

// ============================================================================
// Worker 消息处理
// ============================================================================

/**
 * 处理排序请求
 */
function handleSortRequest<T>(message: SortMessage<T>): SortResult<T> {
  const { id, data, comparatorStr } = message;
  const startTime = performance.now();

  // 重建比较函数
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const cmp = new Function('a', 'b', `return (${comparatorStr})(a, b)`) as (
    a: T,
    b: T
  ) => number;

  // 执行排序
  const sorted = mergeSort(data, cmp);

  const sortTime = performance.now() - startTime;

  return { id, sorted, sortTime };
}

// ============================================================================
// 排序实现
// ============================================================================

/**
 * 归并排序（Worker 内部实现）
 */
function mergeSort<T>(arr: T[], cmp: (a: T, b: T) => number): T[] {
  if (arr.length <= 1) return arr;

  const mid = arr.length >>> 1;
  const left = mergeSort(arr.slice(0, mid), cmp);
  const right = mergeSort(arr.slice(mid), cmp);

  return merge(left, right, cmp);
}

/**
 * 合并两个有序数组
 */
function merge<T>(
  left: T[],
  right: T[],
  cmp: (a: T, b: T) => number
): T[] {
  const result: T[] = [];
  let i = 0, j = 0;

  while (i < left.length && j < right.length) {
    if (cmp(left[i], right[j]) <= 0) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }

  while (i < left.length) result.push(left[i++]);
  while (j < right.length) result.push(right[j++]);

  return result;
}

// ============================================================================
// Worker 入口
// ============================================================================

// 检测是否在 Worker 环境
const isWorkerContext = typeof self !== 'undefined' &&
  typeof (self as unknown as { importScripts?: unknown }).importScripts === 'function';

if (isWorkerContext) {
  self.onmessage = function <T>(e: MessageEvent<SortMessage<T>>) {
    const result = handleSortRequest(e.data);
    self.postMessage(result);
  };
}

// ============================================================================
// 导出（用于测试）
// ============================================================================

export { mergeSort, merge, handleSortRequest };

// ============================================================================
// Worker 代码字符串（用于内联创建）
// ============================================================================

export const WORKER_CODE_STRING = `
// 归并排序
function mergeSort(arr, cmp) {
  if (arr.length <= 1) return arr;

  const mid = arr.length >>> 1;
  const left = mergeSort(arr.slice(0, mid), cmp);
  const right = mergeSort(arr.slice(mid), cmp);

  return merge(left, right, cmp);
}

function merge(left, right, cmp) {
  const result = [];
  let i = 0, j = 0;

  while (i < left.length && j < right.length) {
    if (cmp(left[i], right[j]) <= 0) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }

  while (i < left.length) result.push(left[i++]);
  while (j < right.length) result.push(right[j++]);

  return result;
}

// 消息处理
self.onmessage = function(e) {
  const { id, data, comparatorStr } = e.data;
  const startTime = performance.now();

  // 重建比较函数
  const cmp = new Function('a', 'b', 'return (' + comparatorStr + ')(a, b)');

  // 排序
  const sorted = mergeSort(data, cmp);

  const sortTime = performance.now() - startTime;

  self.postMessage({ id, sorted, sortTime });
};
`;

