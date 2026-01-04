/**
 * 性能计时模块
 *
 * 测量排序算法的性能指标：时间、比较次数、交换次数。
 */

import type { Comparator } from './比较器';

// ============================================================================
// 类型定义
// ============================================================================

/** 性能指标 */
export interface Metrics {
  /** 执行时间（毫秒） */
  timeMs: number;
  /** 比较次数 */
  comparisons: number;
  /** 赋值/交换次数 */
  swaps: number;
}

/** 可计数的比较函数 */
export interface CountingComparator<T> {
  compare: Comparator<T>;
  getCount: () => number;
  reset: () => void;
}

/** 排序函数类型 */
export type SortFunction<T> = (
  arr: readonly T[],
  cmp: Comparator<T>
) => T[];

// ============================================================================
// 比较计数器
// ============================================================================

/**
 * 创建可计数的比较函数
 *
 * @param cmp 原始比较函数
 */
export function createCountingComparator<T>(
  cmp: Comparator<T>
): CountingComparator<T> {
  let count = 0;

  return {
    compare: (a: T, b: T) => {
      count++;
      return cmp(a, b);
    },
    getCount: () => count,
    reset: () => { count = 0; },
  };
}

// ============================================================================
// 性能测量
// ============================================================================

/**
 * 测量排序算法性能
 *
 * @param sortFn 排序函数
 * @param data 测试数据
 * @param cmp 比较函数
 * @returns 性能指标
 */
export function measureSort<T>(
  sortFn: SortFunction<T>,
  data: readonly T[],
  cmp: Comparator<T>
): Metrics {
  // 创建计数比较器
  const countingCmp = createCountingComparator(cmp);

  // 复制数据（避免影响原数组）
  const testData = [...data];

  // 计时
  const startTime = performance.now();
  const result = sortFn(testData, countingCmp.compare);
  const endTime = performance.now();

  return {
    timeMs: endTime - startTime,
    comparisons: countingCmp.getCount(),
    swaps: 0, // 需要算法内部统计
  };
}

/**
 * 多次运行取平均（更准确）
 *
 * @param sortFn 排序函数
 * @param dataGenerator 数据生成器（每次生成新数据）
 * @param cmp 比较函数
 * @param runs 运行次数
 */
export function benchmarkSort<T>(
  sortFn: SortFunction<T>,
  dataGenerator: () => T[],
  cmp: Comparator<T>,
  runs: number = 10
): Metrics & { times: number[] } {
  const times: number[] = [];
  let totalComparisons = 0;

  // 预热运行
  for (let i = 0; i < 3; i++) {
    sortFn(dataGenerator(), cmp);
  }

  // 正式运行
  for (let i = 0; i < runs; i++) {
    const data = dataGenerator();
    const metrics = measureSort(sortFn, data, cmp);
    times.push(metrics.timeMs);
    totalComparisons += metrics.comparisons;
  }

  // 去掉最高最低，计算平均
  const sortedTimes = [...times].sort((a, b) => a - b);
  const trimmedTimes = sortedTimes.slice(1, -1);
  const avgTime = trimmedTimes.length > 0
    ? trimmedTimes.reduce((a, b) => a + b, 0) / trimmedTimes.length
    : times[0] || 0;

  return {
    timeMs: avgTime,
    comparisons: Math.round(totalComparisons / runs),
    swaps: 0,
    times,
  };
}

// ============================================================================
// 对比基准测试
// ============================================================================

/** 算法条目 */
export interface AlgorithmEntry<T> {
  name: string;
  sort: SortFunction<T>;
  stable?: boolean;
}

/** 基准测试结果 */
export interface BenchmarkResult {
  algorithm: string;
  stable: boolean;
  results: {
    size: number;
    timeMs: number;
    comparisons: number;
  }[];
}

/**
 * 对比多个算法
 *
 * @param algorithms 算法列表
 * @param sizes 数据规模列表
 * @param dataGenerator 数据生成器
 * @param cmp 比较函数
 */
export function compareBenchmark<T>(
  algorithms: AlgorithmEntry<T>[],
  sizes: number[],
  dataGenerator: (size: number) => T[],
  cmp: Comparator<T>,
  runs: number = 5
): BenchmarkResult[] {
  const results: BenchmarkResult[] = [];

  for (const algo of algorithms) {
    const algoResults: BenchmarkResult['results'] = [];

    for (const size of sizes) {
      // 对于 O(n²) 算法，大规模时跳过
      if (size > 10000 && !algo.name.includes('快') && !algo.name.includes('归')) {
        algoResults.push({ size, timeMs: -1, comparisons: -1 }); // -1 表示跳过
        continue;
      }

      const metrics = benchmarkSort(
        algo.sort,
        () => dataGenerator(size),
        cmp,
        runs
      );

      algoResults.push({
        size,
        timeMs: metrics.timeMs,
        comparisons: metrics.comparisons,
      });
    }

    results.push({
      algorithm: algo.name,
      stable: algo.stable ?? false,
      results: algoResults,
    });
  }

  return results;
}

// ============================================================================
// 报告输出
// ============================================================================

/**
 * 格式化基准测试报告
 */
export function formatBenchmarkReport(results: BenchmarkResult[]): string {
  if (results.length === 0) return '无结果';

  const sizes = results[0].results.map(r => r.size);

  // 表头
  const header = ['算法', ...sizes.map(s => `n=${s}`), '稳定'].join('\t');

  // 数据行
  const rows = results.map(r => {
    const times = r.results.map(res =>
      res.timeMs < 0 ? '跳过' : `${res.timeMs.toFixed(2)}ms`
    );
    return [r.algorithm, ...times, r.stable ? '✅' : '❌'].join('\t');
  });

  return [header, ...rows].join('\n');
}

// ============================================================================
// 导出
// ============================================================================

export default {
  createCountingComparator,
  measureSort,
  benchmarkSort,
  compareBenchmark,
  formatBenchmarkReport,
};

