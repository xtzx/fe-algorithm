/**
 * 测试用例集
 *
 * 定义基准测试使用的数据规模和分布。
 */

import {
  generateNumbers,
  generateTableRows,
  generateSearchResults,
  type Distribution,
  type TableRow,
  type SearchResult,
} from '../../算法包/公共库/src/数据生成器';

// ============================================================================
// 类型定义
// ============================================================================

/** 测试套件配置 */
export interface TestSuite {
  name: string;
  sizes: number[];
  distributions: Distribution[];
}

/** 测试用例 */
export interface TestCase {
  name: string;
  size: number;
  distribution: Distribution;
  getData: () => number[];
}

/** 对象测试用例 */
export interface ObjectTestCase<T> {
  name: string;
  size: number;
  getData: () => T[];
}

// ============================================================================
// 预定义测试套件
// ============================================================================

/** 快速测试（小规模） */
export const quickSuite: TestSuite = {
  name: '快速测试',
  sizes: [100, 1000],
  distributions: ['random', 'sorted', 'reversed'],
};

/** 标准测试 */
export const standardSuite: TestSuite = {
  name: '标准测试',
  sizes: [100, 1000, 10000],
  distributions: ['random', 'sorted', 'reversed', 'nearlySorted', 'fewUnique'],
};

/** 完整测试（包含大规模） */
export const fullSuite: TestSuite = {
  name: '完整测试',
  sizes: [100, 1000, 10000, 100000],
  distributions: ['random', 'sorted', 'reversed', 'nearlySorted', 'fewUnique', 'sawtooth'],
};

/** 压力测试（仅 O(n log n) 算法） */
export const stressSuite: TestSuite = {
  name: '压力测试',
  sizes: [100000, 500000, 1000000],
  distributions: ['random'],
};

// ============================================================================
// 测试用例生成
// ============================================================================

/**
 * 从套件配置生成测试用例
 */
export function generateTestCases(suite: TestSuite): TestCase[] {
  const cases: TestCase[] = [];

  for (const size of suite.sizes) {
    for (const distribution of suite.distributions) {
      cases.push({
        name: `${distribution}-${size}`,
        size,
        distribution,
        getData: () => generateNumbers(size, distribution),
      });
    }
  }

  return cases;
}

/**
 * 生成表格数据测试用例
 */
export function generateTableTestCases(
  sizes: number[] = [100, 1000, 10000]
): ObjectTestCase<TableRow>[] {
  return sizes.map(size => ({
    name: `表格数据-${size}`,
    size,
    getData: () => generateTableRows(size),
  }));
}

/**
 * 生成搜索结果测试用例
 */
export function generateSearchTestCases(
  sizes: number[] = [100, 1000, 10000]
): ObjectTestCase<SearchResult>[] {
  return sizes.map(size => ({
    name: `搜索结果-${size}`,
    size,
    getData: () => generateSearchResults(size),
  }));
}

// ============================================================================
// 数据分布描述
// ============================================================================

/** 分布类型描述 */
export const distributionDescriptions: Record<Distribution, string> = {
  random: '完全随机数据',
  sorted: '已排序数据（升序）',
  reversed: '完全逆序数据',
  nearlySorted: '近乎有序（5% 乱序）',
  fewUnique: '少量唯一值（10 种）',
  sawtooth: '锯齿形数据',
};

/** 分布对应的最佳算法 */
export const distributionBestAlgorithm: Record<Distribution, string[]> = {
  random: ['快速排序', 'Introsort', '归并排序'],
  sorted: ['插入排序', 'TimSort'],
  reversed: ['堆排序', '归并排序'],
  nearlySorted: ['TimSort', '插入排序'],
  fewUnique: ['三路快排', '计数排序'],
  sawtooth: ['快速排序', '归并排序'],
};

// ============================================================================
// 导出
// ============================================================================

export {
  generateNumbers,
  generateTableRows,
  generateSearchResults,
};

export default {
  quickSuite,
  standardSuite,
  fullSuite,
  stressSuite,
  generateTestCases,
  generateTableTestCases,
  generateSearchTestCases,
  distributionDescriptions,
  distributionBestAlgorithm,
};

