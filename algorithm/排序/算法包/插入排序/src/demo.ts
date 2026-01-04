/**
 * 插入排序使用示例
 */

import { sort, sortInPlace, sortBinary, insertSorted, sortWithStats, meta } from './index';

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
// 示例 2：表格行对象排序（稳定排序）
// ============================================================================

console.log('=== 示例 2：表格行对象排序（稳定排序）===');

interface TableRow {
  id: number;
  name: string;
  score: number;
  department: string;
}

const tableData: TableRow[] = [
  { id: 1, name: 'Alice', score: 85, department: 'HR' },
  { id: 2, name: 'Bob', score: 90, department: 'IT' },
  { id: 3, name: 'Charlie', score: 85, department: 'Finance' },
  { id: 4, name: 'David', score: 78, department: 'IT' },
  { id: 5, name: 'Eve', score: 90, department: 'HR' },
];

console.log('原始数据:');
tableData.forEach(row => console.log(`  ${row.name}(id=${row.id}): ${row.score}分`));

// 按分数降序排序
const sortedByScore = sort(tableData, (a, b) => b.score - a.score);
console.log('\n按分数降序（注意同分学生顺序）:');
sortedByScore.forEach(row => console.log(`  ${row.name}(id=${row.id}): ${row.score}分`));
console.log('✓ Bob(id=2) 和 Eve(id=5) 同为90分，保持原始顺序');
console.log('✓ Alice(id=1) 和 Charlie(id=3) 同为85分，保持原始顺序');

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
  { id: 'a', title: 'Vue 入门教程', relevance: 0.95, publishTime: 1700000000 },
  { id: 'b', title: 'React 最佳实践', relevance: 0.88, publishTime: 1705000000 },
  { id: 'c', title: 'Angular 进阶', relevance: 0.88, publishTime: 1702000000 },
  { id: 'd', title: 'JavaScript 基础', relevance: 0.75, publishTime: 1698000000 },
];

console.log('原始搜索结果:');
searchResults.forEach(r => console.log(`  ${r.title}: 相关度=${r.relevance}`));

// 多列排序：相关度降序，同相关度按时间降序
const sortedResults = sort(searchResults, (a, b) => {
  if (a.relevance !== b.relevance) {
    return b.relevance - a.relevance;
  }
  return b.publishTime - a.publishTime;
});

console.log('\n排序后（相关度↓，同分时间↓）:');
sortedResults.forEach(r => console.log(`  ${r.title}: 相关度=${r.relevance}`));

console.log();

// ============================================================================
// 示例 4：近乎有序数据（插入排序优势场景）⭐
// ============================================================================

console.log('=== 示例 4：近乎有序数据（插入排序优势场景）⭐ ===');

const nearlySorted = [1, 2, 4, 3, 5, 6, 8, 7, 9, 10];
console.log('近乎有序数据:', nearlySorted);

const stats = sortWithStats(nearlySorted, (a, b) => a - b);
console.log('排序结果:', stats.result);
console.log('比较次数:', stats.comparisons);
console.log('移动次数:', stats.moves);
console.log('');
console.log('💡 对于近乎有序的数据，插入排序接近 O(n) 性能！');

// 对比完全随机数据
const randomData = [5, 3, 8, 4, 2, 1, 7, 6, 9, 0];
const randomStats = sortWithStats(randomData, (a, b) => a - b);
console.log('\n完全随机数据:', randomData);
console.log('比较次数:', randomStats.comparisons);
console.log('移动次数:', randomStats.moves);

console.log();

// ============================================================================
// 示例 5：在线排序（增量插入）
// ============================================================================

console.log('=== 示例 5：在线排序（增量插入）===');

const sortedList: number[] = [];
const stream = [5, 3, 8, 1, 9, 2, 7];

console.log('模拟流式数据插入:');
for (const num of stream) {
  const pos = insertSorted(sortedList, num, (a, b) => a - b);
  console.log(`  插入 ${num} 到位置 ${pos} → [${sortedList.join(', ')}]`);
}

console.log();

// ============================================================================
// 示例 6：二分插入排序
// ============================================================================

console.log('=== 示例 6：二分插入排序 ===');

const binaryData = [5, 3, 8, 4, 2, 1, 7, 6];
console.log('原始数据:', binaryData);

const binarySorted = sortBinary(binaryData, (a, b) => a - b);
console.log('二分插入排序结果:', binarySorted);
console.log('💡 二分查找减少比较次数，但移动次数不变');

console.log();

// ============================================================================
// 示例 7：作为其他算法的子程序
// ============================================================================

console.log('=== 示例 7：作为其他算法的子程序 ===');

console.log('插入排序常被用作其他高级排序算法的子程序：');
console.log('1. TimSort：处理小 run（块）时使用插入排序');
console.log('2. Introsort：小数组（通常 <= 16）切换到插入排序');
console.log('3. 快速排序：小分区时切换到插入排序优化');

console.log();

// ============================================================================
// 元信息展示
// ============================================================================

console.log('=== 算法元信息 ===');
console.log('名称:', meta.name);
console.log('稳定性:', meta.stable ? '稳定' : '不稳定');
console.log('原地:', meta.inPlace ? '是' : '否');
console.log('时间复杂度:', meta.timeComplexity);
console.log('  ⭐ 最好情况 O(n)：近乎有序数据');
console.log('空间复杂度:', meta.spaceComplexity);
console.log('适用场景:', meta.适用场景);
console.log('不适用场景:', meta.不适用场景);
