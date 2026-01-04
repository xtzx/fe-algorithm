/**
 * 快速排序 - 使用示例
 */

import { sort, sortRobust, sortWithStats, meta } from './index';
import {
  numberAsc,
  stringAsc,
  byField,
  compose,
  reverse,
} from '../../公共库/src/比较器';
import { generateNumbers } from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} Demo =====\n`);

// ============================================================================
// 1. 基础数字排序
// ============================================================================

console.log('--- 1. 基础数字排序 ---');

const numbers = [38, 27, 43, 3, 9, 82, 10];
console.log('原始数组:', numbers);

const sortedNumbers = sort(numbers, numberAsc);
console.log('升序排序:', sortedNumbers);

const sortedDesc = sort(numbers, reverse(numberAsc));
console.log('降序排序:', sortedDesc);

// ============================================================================
// 2. 表格数据排序
// ============================================================================

console.log('\n--- 2. 表格数据排序 ---');

interface TableRow {
  id: number;
  name: string;
  score: number;
  department: string;
}

const employees: TableRow[] = [
  { id: 1, name: 'Alice', score: 85, department: 'Dev' },
  { id: 2, name: 'Bob', score: 92, department: 'HR' },
  { id: 3, name: 'Charlie', score: 78, department: 'Dev' },
  { id: 4, name: 'David', score: 92, department: 'Dev' },
  { id: 5, name: 'Eve', score: 88, department: 'HR' },
];

console.log('原始数据:');
employees.forEach((e) => console.log(`  ${e.name}: ${e.score} (${e.department})`));

// 按分数降序
const byScore = byField<TableRow, 'score'>('score', reverse(numberAsc));
const byScoreSorted = sort(employees, byScore);

console.log('\n按分数降序:');
byScoreSorted.forEach((e) => console.log(`  ${e.name}: ${e.score}`));

// 多列排序：部门升序 + 分数降序
const multiSort = compose(
  byField<TableRow, 'department'>('department', stringAsc),
  byField<TableRow, 'score'>('score', reverse(numberAsc))
);

const multiSorted = sort(employees, multiSort);
console.log('\n按部门升序，同部门按分数降序:');
multiSorted.forEach((e) =>
  console.log(`  ${e.department} | ${e.name}: ${e.score}`)
);

// ============================================================================
// 3. 搜索结果排序
// ============================================================================

console.log('\n--- 3. 搜索结果排序 ---');

interface SearchResult {
  id: string;
  title: string;
  relevance: number;
  publishTime: number;
}

const searchResults: SearchResult[] = [
  { id: 'a', title: 'Node.js Tutorial', relevance: 0.8, publishTime: 1700000000 },
  { id: 'b', title: 'React Hooks Guide', relevance: 0.9, publishTime: 1700000100 },
  { id: 'c', title: 'Vue 3 Basics', relevance: 0.8, publishTime: 1700000200 },
  { id: 'd', title: 'Node.js Performance', relevance: 0.9, publishTime: 1700000050 },
];

const searchSort = compose(
  byField<SearchResult, 'relevance'>('relevance', reverse(numberAsc)),
  byField<SearchResult, 'publishTime'>('publishTime', reverse(numberAsc))
);

const sortedResults = sort(searchResults, searchSort);
console.log('按相关度降序，同分按时间降序:');
sortedResults.forEach((r) => console.log(`  [${r.relevance}] ${r.title}`));

// ============================================================================
// 4. 排序统计信息
// ============================================================================

console.log('\n--- 4. 排序统计信息 ---');

const randomData = generateNumbers(100, 'random');
const { result, comparisons, swaps, recursionDepth } = sortWithStats(
  randomData,
  numberAsc
);

console.log(`数组大小: ${randomData.length}`);
console.log(`比较次数: ${comparisons}`);
console.log(`交换次数: ${swaps}`);
console.log(`最大递归深度: ${recursionDepth}`);
console.log(`排序结果（前10个）: [${result.slice(0, 10).join(', ')}...]`);

// ============================================================================
// 5. 不同数据分布的表现
// ============================================================================

console.log('\n--- 5. 不同数据分布的表现 ---');

const distributions = ['random', 'sorted', 'reversed', 'fewUnique'] as const;

for (const dist of distributions) {
  const data = generateNumbers(1000, dist);

  console.time(`  ${dist}`);
  sortRobust(data, numberAsc);
  console.timeEnd(`  ${dist}`);
}

// ============================================================================
// 6. 健壮版 vs 基础版性能对比
// ============================================================================

console.log('\n--- 6. 健壮版 vs 基础版（有序数据）---');

// 有序数据是快排的最坏情况
const sortedData = generateNumbers(5000, 'sorted');

console.log('测试有序数据（5000 个元素）:');

console.time('  基础版（可能较慢）');
try {
  sort(sortedData.slice(0, 1000), numberAsc); // 减少规模避免栈溢出
  console.timeEnd('  基础版（可能较慢）');
} catch (e) {
  console.log('  基础版：栈溢出！');
}

console.time('  健壮版');
sortRobust(sortedData, numberAsc);
console.timeEnd('  健壮版');

// ============================================================================
// 7. 大规模数据
// ============================================================================

console.log('\n--- 7. 大规模数据测试 ---');

const largeArray = generateNumbers(100000, 'random');

console.time('  100000 个随机数');
sortRobust(largeArray, numberAsc);
console.timeEnd('  100000 个随机数');

console.log('\n===== Demo 完成 =====');

