/**
 * 并行归并排序
 *
 * 利用 Web Worker 实现多线程并行排序
 */

// ============================================================================
// 类型定义
// ============================================================================

export type Comparator<T> = (a: T, b: T) => number;

export interface ParallelSortOptions {
  /** Worker 数量，默认 navigator.hardwareConcurrency */
  workerCount?: number;
  /** 低于此阈值使用单线程，默认 10000 */
  threshold?: number;
}

interface WorkerMessage<T> {
  id: number;
  data: T[];
  comparatorStr: string;
}

interface WorkerResult<T> {
  id: number;
  sorted: T[];
}

// ============================================================================
// 常量
// ============================================================================

const DEFAULT_THRESHOLD = 10000;
const DEFAULT_WORKER_COUNT = typeof navigator !== 'undefined'
  ? navigator.hardwareConcurrency || 4
  : 4;

// ============================================================================
// Worker 代码（内联）
// ============================================================================

const WORKER_CODE = `
self.onmessage = function(e) {
  const { id, data, comparatorStr } = e.data;

  // 重建比较函数
  const cmp = new Function('a', 'b', 'return (' + comparatorStr + ')(a, b)');

  // 排序
  const sorted = [...data].sort(cmp);

  self.postMessage({ id, sorted });
};
`;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 判断是否值得使用并行排序
 */
export function shouldUseParallel(
  dataSize: number,
  threshold = DEFAULT_THRESHOLD
): boolean {
  if (typeof Worker === 'undefined') {
    return false;
  }
  return dataSize >= threshold;
}

/**
 * 将数组分成多个块
 */
function splitIntoChunks<T>(arr: readonly T[], chunkCount: number): T[][] {
  const chunks: T[][] = [];
  const chunkSize = Math.ceil(arr.length / chunkCount);

  for (let i = 0; i < arr.length; i += chunkSize) {
    chunks.push(arr.slice(i, i + chunkSize) as T[]);
  }

  return chunks;
}

/**
 * K 路归并
 */
function kWayMerge<T>(
  sortedArrays: T[][],
  cmp: Comparator<T>
): T[] {
  if (sortedArrays.length === 0) return [];
  if (sortedArrays.length === 1) return sortedArrays[0];

  // 使用最小堆优化
  // 这里用简单实现，对于 K 较小的情况足够高效
  const result: T[] = [];
  const indices = new Array(sortedArrays.length).fill(0);
  const totalLength = sortedArrays.reduce((sum, arr) => sum + arr.length, 0);

  for (let i = 0; i < totalLength; i++) {
    let minIdx = -1;
    let minVal: T | undefined;

    for (let k = 0; k < sortedArrays.length; k++) {
      const arr = sortedArrays[k];
      const idx = indices[k];

      if (idx < arr.length) {
        if (minIdx === -1 || cmp(arr[idx], minVal!) < 0) {
          minIdx = k;
          minVal = arr[idx];
        }
      }
    }

    if (minIdx !== -1) {
      result.push(minVal!);
      indices[minIdx]++;
    }
  }

  return result;
}

/**
 * 创建 Worker
 */
function createWorker(): Worker | null {
  if (typeof Worker === 'undefined') {
    return null;
  }

  try {
    const blob = new Blob([WORKER_CODE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    URL.revokeObjectURL(url);
    return worker;
  } catch {
    console.warn('Failed to create Web Worker');
    return null;
  }
}

/**
 * 将比较函数转为字符串
 */
function comparatorToString<T>(cmp: Comparator<T>): string {
  const str = cmp.toString();
  // 处理箭头函数和普通函数
  if (str.startsWith('function')) {
    return str;
  }
  // 箭头函数：(a, b) => ... 或 a => ...
  return str;
}

// ============================================================================
// 主函数
// ============================================================================

/**
 * 并行归并排序
 *
 * @param arr 待排序数组
 * @param cmp 比较函数
 * @param options 配置选项
 * @returns 排序后的数组
 *
 * @example
 * const sorted = await parallelMergeSort(
 *   [3, 1, 4, 1, 5, 9, 2, 6],
 *   (a, b) => a - b
 * );
 */
export async function parallelMergeSort<T>(
  arr: readonly T[],
  cmp: Comparator<T>,
  options: ParallelSortOptions = {}
): Promise<T[]> {
  const {
    workerCount = DEFAULT_WORKER_COUNT,
    threshold = DEFAULT_THRESHOLD,
  } = options;

  // 数据量太小，直接单线程排序
  if (arr.length < threshold) {
    return [...arr].sort(cmp);
  }

  // Worker 不可用，回退到单线程
  if (typeof Worker === 'undefined') {
    console.warn('Web Worker not available, falling back to single-threaded sort');
    return [...arr].sort(cmp);
  }

  // 实际使用的 Worker 数量（不超过数据块数）
  const actualWorkerCount = Math.min(workerCount, Math.ceil(arr.length / 1000));

  // 分块
  const chunks = splitIntoChunks(arr, actualWorkerCount);

  // 序列化比较函数
  const comparatorStr = comparatorToString(cmp);

  // 并行排序
  const sortedChunks = await Promise.all(
    chunks.map((chunk, idx) => sortInWorker(chunk, comparatorStr, idx))
  );

  // K 路归并
  return kWayMerge(sortedChunks, cmp);
}

/**
 * 在 Worker 中排序
 */
function sortInWorker<T>(
  data: T[],
  comparatorStr: string,
  id: number
): Promise<T[]> {
  return new Promise((resolve, reject) => {
    const worker = createWorker();

    if (!worker) {
      // 回退到主线程
      const cmp = new Function('a', 'b', `return (${comparatorStr})(a, b)`) as Comparator<T>;
      resolve([...data].sort(cmp));
      return;
    }

    const timeoutId = setTimeout(() => {
      worker.terminate();
      reject(new Error('Worker timeout'));
    }, 30000);

    worker.onmessage = (e: MessageEvent<WorkerResult<T>>) => {
      clearTimeout(timeoutId);
      worker.terminate();
      resolve(e.data.sorted);
    };

    worker.onerror = (error) => {
      clearTimeout(timeoutId);
      worker.terminate();
      reject(error);
    };

    worker.postMessage({
      id,
      data,
      comparatorStr,
    } as WorkerMessage<T>);
  });
}

// ============================================================================
// 同步版本（用于对比）
// ============================================================================

/**
 * 单线程归并排序（用于对比）
 */
export function singleThreadMergeSort<T>(
  arr: readonly T[],
  cmp: Comparator<T>
): T[] {
  const result = [...arr];

  function mergeSort(lo: number, hi: number): void {
    if (lo >= hi) return;

    const mid = (lo + hi) >>> 1;
    mergeSort(lo, mid);
    mergeSort(mid + 1, hi);
    merge(lo, mid, hi);
  }

  function merge(lo: number, mid: number, hi: number): void {
    const temp: T[] = [];
    let i = lo, j = mid + 1;

    while (i <= mid && j <= hi) {
      if (cmp(result[i], result[j]) <= 0) {
        temp.push(result[i++]);
      } else {
        temp.push(result[j++]);
      }
    }

    while (i <= mid) temp.push(result[i++]);
    while (j <= hi) temp.push(result[j++]);

    for (let k = 0; k < temp.length; k++) {
      result[lo + k] = temp[k];
    }
  }

  mergeSort(0, result.length - 1);
  return result;
}

// ============================================================================
// 性能测试工具
// ============================================================================

export interface BenchmarkResult {
  singleThread: number;
  parallel: number;
  speedup: number;
}

/**
 * 性能对比测试
 */
export async function benchmark<T>(
  arr: readonly T[],
  cmp: Comparator<T>,
  workerCount = DEFAULT_WORKER_COUNT
): Promise<BenchmarkResult> {
  // 单线程
  const singleStart = performance.now();
  singleThreadMergeSort(arr, cmp);
  const singleThread = performance.now() - singleStart;

  // 并行
  const parallelStart = performance.now();
  await parallelMergeSort(arr, cmp, { workerCount, threshold: 0 });
  const parallel = performance.now() - parallelStart;

  return {
    singleThread,
    parallel,
    speedup: singleThread / parallel,
  };
}

// ============================================================================
// 元信息
// ============================================================================

export const meta = {
  name: '并行归并排序',
  timeComplexity: 'O(n/P · log(n/P) + n log P)',
  spaceComplexity: 'O(n)',
  stable: true,
  bestCase: '数据量大、CPU 核心多',
  worstCase: '数据量小、Worker 创建开销',
  description: '利用多线程并行排序，加速大规模数据处理',
};

