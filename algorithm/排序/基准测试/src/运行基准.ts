/**
 * 运行基准
 *
 * 对已注册的排序算法进行基准测试。
 */

import { numberAsc } from '../../算法包/公共库/src/比较器';
import {
  benchmarkSort,
  formatBenchmarkReport,
  type AlgorithmEntry,
  type BenchmarkResult,
} from '../../算法包/公共库/src/性能计时';
import {
  generateNumbers,
  type Distribution,
} from '../../算法包/公共库/src/数据生成器';
import {
  standardSuite,
  generateTestCases,
  distributionDescriptions,
} from './测试用例集';

// ============================================================================
// 算法注册
// ============================================================================

/**
 * 已注册的算法列表
 *
 * 说明：S1a/S1b/S2/S3 步骤完成后，在这里添加导入
 */
const algorithms: AlgorithmEntry<number>[] = [
  // ============ 占位：待 S1a 补充 ============
  // {
  //   name: '冒泡排序',
  //   sort: bubbleSort,
  //   stable: true,
  // },
  // {
  //   name: '选择排序',
  //   sort: selectionSort,
  //   stable: false,
  // },
  // {
  //   name: '插入排序',
  //   sort: insertionSort,
  //   stable: true,
  // },
  // {
  //   name: '希尔排序',
  //   sort: shellSort,
  //   stable: false,
  // },

  // ============ 占位：待 S1b 补充 ============
  // {
  //   name: '归并排序',
  //   sort: mergeSort,
  //   stable: true,
  // },
  // {
  //   name: '快速排序',
  //   sort: quickSort,
  //   stable: false,
  // },
  // {
  //   name: '堆排序',
  //   sort: heapSort,
  //   stable: false,
  // },

  // ============ 占位：待 S2 补充 ============
  // {
  //   name: '三路快排',
  //   sort: quickSort3Way,
  //   stable: false,
  // },
  // {
  //   name: 'Introsort',
  //   sort: introsort,
  //   stable: false,
  // },
  // {
  //   name: 'TimSort',
  //   sort: timSort,
  //   stable: true,
  // },

  // ============ 临时：原生 sort 作为基准 ============
  {
    name: 'Array.sort（原生）',
    sort: (arr, cmp) => [...arr].sort(cmp),
    stable: true,
  },
];

// ============================================================================
// 基准测试运行器
// ============================================================================

interface BenchmarkConfig {
  sizes: number[];
  distributions: Distribution[];
  runs: number;
}

const defaultConfig: BenchmarkConfig = {
  sizes: [1000, 10000, 100000],
  distributions: ['random'],
  runs: 5,
};

/**
 * 运行基准测试
 */
function runBenchmark(config: BenchmarkConfig = defaultConfig): void {
  console.log('='.repeat(60));
  console.log('排序算法基准测试');
  console.log('='.repeat(60));
  console.log();

  if (algorithms.length === 0) {
    console.log('⚠️  没有注册任何算法！');
    console.log('请在 S1a/S1b/S2/S3 步骤完成后添加算法导入。');
    return;
  }

  for (const distribution of config.distributions) {
    console.log(`📊 分布：${distributionDescriptions[distribution] || distribution}`);
    console.log('-'.repeat(60));

    const results: BenchmarkResult[] = [];

    for (const algo of algorithms) {
      const algoResults: BenchmarkResult['results'] = [];

      for (const size of config.sizes) {
        // 对于 O(n²) 算法，大规模时跳过
        const isSlowAlgo = ['冒泡', '选择', '插入'].some(s => algo.name.includes(s));
        if (isSlowAlgo && size > 10000) {
          algoResults.push({ size, timeMs: -1, comparisons: -1 });
          continue;
        }

        try {
          const metrics = benchmarkSort(
            algo.sort,
            () => generateNumbers(size, distribution),
            numberAsc,
            config.runs
          );

          algoResults.push({
            size,
            timeMs: metrics.timeMs,
            comparisons: metrics.comparisons,
          });
        } catch (error) {
          console.error(`❌ ${algo.name} 在 n=${size} 时出错:`, error);
          algoResults.push({ size, timeMs: -2, comparisons: -2 });
        }
      }

      results.push({
        algorithm: algo.name,
        stable: algo.stable ?? false,
        results: algoResults,
      });
    }

    // 输出表格
    printResultTable(results, config.sizes);
    console.log();
  }
}

/**
 * 打印结果表格
 */
function printResultTable(results: BenchmarkResult[], sizes: number[]): void {
  // 计算列宽
  const nameWidth = Math.max(
    '算法'.length,
    ...results.map(r => r.algorithm.length)
  ) + 2;

  const sizeWidth = 12;

  // 表头
  let header = '算法'.padEnd(nameWidth);
  for (const size of sizes) {
    header += `n=${size}`.padStart(sizeWidth);
  }
  header += '稳定'.padStart(6);
  console.log(header);
  console.log('-'.repeat(header.length));

  // 数据行
  for (const result of results) {
    let row = result.algorithm.padEnd(nameWidth);

    for (const res of result.results) {
      let cell: string;
      if (res.timeMs === -1) {
        cell = '跳过';
      } else if (res.timeMs === -2) {
        cell = '错误';
      } else {
        cell = `${res.timeMs.toFixed(2)}ms`;
      }
      row += cell.padStart(sizeWidth);
    }

    row += (result.stable ? ' ✅' : ' ❌').padStart(6);
    console.log(row);
  }
}

// ============================================================================
// 主入口
// ============================================================================

// 如果直接运行此文件
if (typeof require !== 'undefined' && require.main === module) {
  runBenchmark();
}

// ============================================================================
// 导出
// ============================================================================

export { runBenchmark, algorithms };
export type { BenchmarkConfig };

