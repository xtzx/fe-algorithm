/**
 * 希尔排序使用示例
 */

import {
  sort,
  sortInPlace,
  sortShell,
  sortHibbard,
  sortWithStats,
  getKnuthGaps,
  getShellGaps,
  getHibbardGaps,
  meta
} from './index';

// ============================================================================
// 示例 1：基础数字排序
// ============================================================================

console.log('=== 示例 1：基础数字排序 ===');

const numbers = [5, 3, 8, 4, 2, 1, 7, 6, 9, 0];
console.log('原始数组:', numbers);

const sortedAsc = sort(numbers, (a, b) => a - b);
console.log('升序排序:', sortedAsc);

const sortedDesc = sort(numbers, (a, b) => b - a);
console.log('降序排序:', sortedDesc);

console.log('原数组未变:', numbers);
console.log();

// ============================================================================
// 示例 2：表格行对象排序
// ============================================================================

console.log('=== 示例 2：表格行对象排序 ===');

interface TableRow {
  id: number;
  name: string;
  score: number;
}

const tableData: TableRow[] = [
  { id: 1, name: 'Alice', score: 85 },
  { id: 2, name: 'Bob', score: 90 },
  { id: 3, name: 'Charlie', score: 78 },
  { id: 4, name: 'David', score: 92 },
  { id: 5, name: 'Eve', score: 88 },
];

console.log('原始数据:');
tableData.forEach(row => console.log(`  ${row.name}: ${row.score}分`));

// 按分数降序排序
const sortedByScore = sort(tableData, (a, b) => b.score - a.score);
console.log('\n按分数降序:');
sortedByScore.forEach(row => console.log(`  ${row.name}: ${row.score}分`));

console.log();

// ============================================================================
// 示例 3：搜索结果对象排序
// ============================================================================

console.log('=== 示例 3：搜索结果对象排序 ===');

interface SearchResult {
  id: string;
  title: string;
  relevance: number;
  publishTime: number;
}

const searchResults: SearchResult[] = [
  { id: 'a', title: 'Vue 入门', relevance: 0.95, publishTime: 1700000000 },
  { id: 'b', title: 'React 实践', relevance: 0.88, publishTime: 1705000000 },
  { id: 'c', title: 'Angular 进阶', relevance: 0.82, publishTime: 1702000000 },
  { id: 'd', title: 'JS 基础', relevance: 0.75, publishTime: 1698000000 },
  { id: 'e', title: 'TS 指南', relevance: 0.90, publishTime: 1703000000 },
];

console.log('原始搜索结果:');
searchResults.forEach(r => console.log(`  ${r.title}: 相关度=${r.relevance}`));

// 按相关度降序排序
const sortedResults = sort(searchResults, (a, b) => b.relevance - a.relevance);
console.log('\n按相关度降序:');
sortedResults.forEach(r => console.log(`  ${r.title}: 相关度=${r.relevance}`));

console.log();

// ============================================================================
// 示例 4：演示不稳定性 ⚠️
// ============================================================================

console.log('=== 示例 4：演示不稳定性 ⚠️ ===');

interface Item {
  key: number;
  id: string;
}

const unstableDemo: Item[] = [
  { key: 3, id: 'a' },
  { key: 1, id: 'b' },
  { key: 3, id: 'c' },
  { key: 2, id: 'd' },
];

console.log('原始数据:');
unstableDemo.forEach(item => console.log(`  key=${item.key}, id=${item.id}`));

const sortedUnstable = sort(unstableDemo, (a, b) => a.key - b.key);
console.log('\n排序后:');
sortedUnstable.forEach(item => console.log(`  key=${item.key}, id=${item.id}`));

console.log('\n⚠️ 注意：key=3 的元素顺序可能改变（不稳定）');

console.log();

// ============================================================================
// 示例 5：间隔序列对比
// ============================================================================

console.log('=== 示例 5：间隔序列对比 ===');

const n = 100;
console.log(`数组长度: ${n}`);
console.log('Knuth 序列:', getKnuthGaps(n));
console.log('Shell 序列:', getShellGaps(n));
console.log('Hibbard 序列:', getHibbardGaps(n));

console.log();

// ============================================================================
// 示例 6：不同间隔序列的排序
// ============================================================================

console.log('=== 示例 6：不同间隔序列的排序 ===');

const testData = [8, 3, 7, 1, 9, 2, 6, 4, 5, 0];
console.log('测试数据:', testData);

console.log('Knuth 序列排序:', sort(testData, (a, b) => a - b));
console.log('Shell 序列排序:', sortShell(testData, (a, b) => a - b));
console.log('Hibbard 序列排序:', sortHibbard(testData, (a, b) => a - b));

console.log();

// ============================================================================
// 示例 7：排序统计
// ============================================================================

console.log('=== 示例 7：排序统计 ===');

const statsData = [5, 3, 8, 4, 2, 1, 7, 6, 9, 0];
console.log('测试数据:', statsData);

const stats = sortWithStats(statsData, (a, b) => a - b);
console.log('排序结果:', stats.result);
console.log('使用的间隔序列:', stats.gaps);
console.log('比较次数:', stats.comparisons);
console.log('移动次数:', stats.moves);

console.log();

// ============================================================================
// 示例 8：与 O(n²) 算法对比
// ============================================================================

console.log('=== 示例 8：希尔排序 vs 插入排序 ===');

// 生成随机数据
const size = 1000;
const randomData = Array.from({ length: size }, () => Math.floor(Math.random() * 10000));

// 希尔排序
const shellStart = performance.now();
sort([...randomData], (a, b) => a - b);
const shellTime = performance.now() - shellStart;

console.log(`数据规模: ${size}`);
console.log(`希尔排序耗时: ${shellTime.toFixed(2)}ms`);
console.log('💡 希尔排序在中等规模数据上比纯插入排序快很多');

console.log();

// ============================================================================
// 元信息展示
// ============================================================================

console.log('=== 算法元信息 ===');
console.log('名称:', meta.name);
console.log('稳定性:', meta.stable ? '稳定' : '⚠️ 不稳定');
console.log('原地:', meta.inPlace ? '是' : '否');
console.log('时间复杂度:', meta.timeComplexity);
console.log('空间复杂度:', meta.spaceComplexity);
console.log('适用场景:', meta.适用场景);
console.log('不适用场景:', meta.不适用场景);
