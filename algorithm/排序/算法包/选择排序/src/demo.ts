/**
 * 选择排序使用示例
 */

import { sort, sortInPlace, sortBidirectional, sortWithStats, meta } from './index';

// ============================================================================
// 示例 1：基础数字排序
// ============================================================================

console.log('=== 示例 1：基础数字排序 ===');

const numbers = [5, 3, 8, 4, 2, 1, 7, 6];
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

// 检查 key=3 的元素顺序
const key3Items = sortedUnstable.filter(item => item.key === 3);
console.log('\nkey=3 的元素顺序:', key3Items.map(i => i.id).join(', '));
console.log('⚠️ 注意：可能是 c, a（与原始顺序 a, c 不同）');

console.log();

// ============================================================================
// 示例 5：交换次数对比
// ============================================================================

console.log('=== 示例 5：交换次数对比 ===');

const testData = [5, 3, 8, 4, 2, 1, 7, 6, 9, 0];
console.log('测试数据:', testData);
console.log('数据长度:', testData.length);

const stats = sortWithStats(testData, (a, b) => a - b);
console.log('排序结果:', stats.result);
console.log('比较次数:', stats.comparisons);
console.log('交换次数:', stats.swaps);
console.log('理论最大交换次数: n-1 =', testData.length - 1);
console.log('');
console.log('💡 选择排序的优势：交换次数最多只有 O(n)');
console.log('   对比冒泡排序：交换次数可能达到 O(n²)');

console.log();

// ============================================================================
// 示例 6：双向选择排序
// ============================================================================

console.log('=== 示例 6：双向选择排序 ===');

const bidirData = [5, 3, 8, 4, 2, 1, 7, 6];
console.log('原始数据:', bidirData);

const bidirResult = sortBidirectional(bidirData, (a, b) => a - b);
console.log('双向选择排序结果:', bidirResult);
console.log('💡 双向选择：每轮同时找最小和最大，比较次数减半');

console.log();

// ============================================================================
// 示例 7：原地排序
// ============================================================================

console.log('=== 示例 7：原地排序 ===');

const arr = [5, 2, 8, 1, 9];
console.log('排序前:', arr);
sortInPlace(arr, (a, b) => a - b);
console.log('原地排序后:', arr);

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
