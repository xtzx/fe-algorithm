/**
 * 快速选择 - 使用示例
 */

import {
  quickSelectCopy,
  topKSmallest,
  topKLargest,
  topKSmallestSorted,
  topKLargestSorted,
  kthSmallest,
  kthLargest,
  median,
  percentile,
  percentiles,
  meta,
} from './index';
import { numberAsc, byField } from '../../公共库/src/比较器';
import { generateNumbers } from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} Demo =====\n`);

// ============================================================================
// 1. 基础用法：找第 K 小的元素
// ============================================================================

console.log('--- 1. 找第 K 小的元素 ---');

const numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
console.log('原始数组:', numbers);

const k2 = quickSelectCopy(numbers, 2, numberAsc);
console.log(`第 3 小的元素（索引 2）: ${k2}`); // 应该是 2

const k0 = quickSelectCopy(numbers, 0, numberAsc);
console.log(`最小的元素（索引 0）: ${k0}`); // 应该是 1

const kLast = quickSelectCopy(numbers, numbers.length - 1, numberAsc);
console.log(`最大的元素（索引 ${numbers.length - 1}）: ${kLast}`); // 应该是 9

// ============================================================================
// 2. LeetCode 风格：第 K 大/小（1-indexed）
// ============================================================================

console.log('\n--- 2. LeetCode 风格（1-indexed）---');

const arr = [3, 2, 1, 5, 6, 4];
console.log('数组:', arr);

console.log(`第 1 小: ${kthSmallest(arr, 1, numberAsc)}`); // 1
console.log(`第 2 小: ${kthSmallest(arr, 2, numberAsc)}`); // 2
console.log(`第 1 大: ${kthLargest(arr, 1, numberAsc)}`); // 6
console.log(`第 2 大: ${kthLargest(arr, 2, numberAsc)}`); // 5

// ============================================================================
// 3. TopK 问题
// ============================================================================

console.log('\n--- 3. TopK 问题 ---');

const data = [64, 34, 25, 12, 22, 11, 90, 45, 78, 33];
console.log('原始数据:', data);

const smallest3 = topKSmallest(data, 3, numberAsc);
console.log('最小的 3 个（无序）:', smallest3);

const smallest3Sorted = topKSmallestSorted(data, 3, numberAsc);
console.log('最小的 3 个（有序）:', smallest3Sorted);

const largest3 = topKLargest(data, 3, numberAsc);
console.log('最大的 3 个（无序）:', largest3);

const largest3Sorted = topKLargestSorted(data, 3, numberAsc);
console.log('最大的 3 个（有序）:', largest3Sorted);

// ============================================================================
// 4. 中位数
// ============================================================================

console.log('\n--- 4. 中位数 ---');

const oddArray = [7, 3, 1, 5, 9];
console.log(`奇数个元素 ${JSON.stringify(oddArray)} 的中位数: ${median(oddArray)}`);

const evenArray = [7, 3, 1, 5, 9, 2];
console.log(`偶数个元素 ${JSON.stringify(evenArray)} 的中位数: ${median(evenArray)}`);

// ============================================================================
// 5. 百分位数（性能监控场景）
// ============================================================================

console.log('\n--- 5. 百分位数（性能监控场景）---');

// 模拟 API 延迟数据（毫秒）
const latencies = [
  45, 52, 48, 51, 49, 47, 53, 46, 50, 48,
  55, 60, 58, 62, 57, 59, 61, 56, 63, 54,
  120, 150, 200, // 一些慢请求
];

console.log('API 延迟数据（毫秒）:', latencies);

const p50 = percentile(latencies, 0.5);
const p90 = percentile(latencies, 0.9);
const p99 = percentile(latencies, 0.99);

console.log(`P50（中位数）: ${p50}ms`);
console.log(`P90: ${p90}ms`);
console.log(`P99: ${p99}ms`);

// 批量计算多个百分位
const allPercentiles = percentiles(latencies, [0.5, 0.75, 0.9, 0.95, 0.99]);
console.log('\n批量百分位:');
allPercentiles.forEach((value, key) => {
  console.log(`  P${key * 100}: ${value}ms`);
});

// ============================================================================
// 6. 搜索结果 TopK
// ============================================================================

console.log('\n--- 6. 搜索结果 TopK ---');

interface SearchResult {
  id: string;
  title: string;
  relevance: number;
}

const searchResults: SearchResult[] = [
  { id: '1', title: 'Node.js Tutorial', relevance: 0.85 },
  { id: '2', title: 'React Hooks Guide', relevance: 0.92 },
  { id: '3', title: 'Vue 3 Basics', relevance: 0.78 },
  { id: '4', title: 'TypeScript Deep Dive', relevance: 0.95 },
  { id: '5', title: 'Docker for Beginners', relevance: 0.70 },
  { id: '6', title: 'Kubernetes 101', relevance: 0.88 },
];

console.log('搜索结果:');
searchResults.forEach((r) => console.log(`  [${r.relevance}] ${r.title}`));

const relevanceCmp = (a: SearchResult, b: SearchResult) => a.relevance - b.relevance;
const top3Results = topKLargestSorted(searchResults, 3, relevanceCmp);

console.log('\n相关度 Top 3:');
top3Results.forEach((r, i) => console.log(`  ${i + 1}. [${r.relevance}] ${r.title}`));

// ============================================================================
// 7. 性能对比：快速选择 vs 排序
// ============================================================================

console.log('\n--- 7. 性能对比：快速选择 vs 排序 ---');

const largeArray = generateNumbers(100000, 'random');
const k = 100; // 找第 100 小

console.log(`数组大小: ${largeArray.length}, 找第 ${k} 小的元素`);

// 方法 1：排序后取
console.time('  排序后取');
const sorted = [...largeArray].sort((a, b) => a - b);
const _result1 = sorted[k - 1];
console.timeEnd('  排序后取');

// 方法 2：快速选择
console.time('  快速选择');
const _result2 = kthSmallest(largeArray, k, numberAsc);
console.timeEnd('  快速选择');

console.log('\n⭐ 快速选择在 TopK 问题上通常比全排序快很多！');

// ============================================================================
// 8. TopK 不同 K 值的表现
// ============================================================================

console.log('\n--- 8. TopK 不同 K 值的表现 ---');

const testArray = generateNumbers(50000, 'random');
const kValues = [10, 100, 1000, 5000];

for (const kVal of kValues) {
  console.time(`  TopK (k=${kVal})`);
  topKSmallest(testArray, kVal, numberAsc);
  console.timeEnd(`  TopK (k=${kVal})`);
}

console.log('\n===== Demo 完成 =====');

