/**
 * 冒泡排序使用示例
 */

import { sort, sortInPlace, sortCocktail, meta } from './index';

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
tableData.forEach(row => console.log(`  ${row.name}: ${row.score}分, ${row.department}`));

// 按分数降序排序
const sortedByScore = sort(tableData, (a, b) => b.score - a.score);
console.log('\n按分数降序:');
sortedByScore.forEach(row => console.log(`  ${row.name}: ${row.score}分`));

// 多列排序：分数降序，同分按姓名升序
const sortedMulti = sort(tableData, (a, b) => {
  if (a.score !== b.score) {
    return b.score - a.score;  // 分数降序
  }
  return a.name.localeCompare(b.name);  // 同分按姓名升序
});
console.log('\n分数降序，同分按姓名:');
sortedMulti.forEach(row => console.log(`  ${row.name}: ${row.score}分`));

// 利用稳定性实现多列排序
let stableResult = [...tableData];
// 先按次要列（姓名）排序
stableResult = sort(stableResult, (a, b) => a.name.localeCompare(b.name));
// 再按主要列（分数）排序，稳定排序保持同分时的姓名顺序
stableResult = sort(stableResult, (a, b) => b.score - a.score);
console.log('\n利用稳定性（先姓名后分数）:');
stableResult.forEach(row => console.log(`  ${row.name}: ${row.score}分`));

console.log();

// ============================================================================
// 示例 3：搜索结果对象排序
// ============================================================================

console.log('=== 示例 3：搜索结果对象排序 ===');

interface SearchResult {
  id: string;
  title: string;
  relevance: number;    // 相关度分数
  publishTime: number;  // 发布时间戳
}

const searchResults: SearchResult[] = [
  { id: 'a', title: 'Vue 入门教程', relevance: 0.95, publishTime: 1700000000 },
  { id: 'b', title: 'React 最佳实践', relevance: 0.88, publishTime: 1705000000 },
  { id: 'c', title: 'Angular 进阶', relevance: 0.88, publishTime: 1702000000 },
  { id: 'd', title: 'JavaScript 基础', relevance: 0.75, publishTime: 1698000000 },
  { id: 'e', title: 'TypeScript 指南', relevance: 0.95, publishTime: 1703000000 },
];

console.log('原始搜索结果:');
searchResults.forEach(r => console.log(`  ${r.title}: 相关度=${r.relevance}`));

// 按相关度降序，同相关度按发布时间降序（最新优先）
const sortedResults = sort(searchResults, (a, b) => {
  if (a.relevance !== b.relevance) {
    return b.relevance - a.relevance;  // 相关度降序
  }
  return b.publishTime - a.publishTime;  // 同相关度，时间降序
});

console.log('\n排序后（相关度↓，同分时间↓）:');
sortedResults.forEach(r => {
  const date = new Date(r.publishTime * 1000).toLocaleDateString();
  console.log(`  ${r.title}: 相关度=${r.relevance}, 日期=${date}`);
});

console.log();

// ============================================================================
// 示例 4：原地排序
// ============================================================================

console.log('=== 示例 4：原地排序 ===');

const arr = [5, 2, 8, 1, 9];
console.log('排序前:', arr);
sortInPlace(arr, (a, b) => a - b);
console.log('原地排序后:', arr);

console.log();

// ============================================================================
// 示例 5：鸡尾酒排序（双向冒泡）
// ============================================================================

console.log('=== 示例 5：鸡尾酒排序 ===');

// 对于"乌龟"元素（小元素在右边），鸡尾酒排序更有效
const turtleCase = [2, 3, 4, 5, 1];  // 1 是"乌龟"
console.log('乌龟元素案例:', turtleCase);

const cocktailResult = sortCocktail(turtleCase, (a, b) => a - b);
console.log('鸡尾酒排序结果:', cocktailResult);

console.log();

// ============================================================================
// 示例 6：检测数组是否有序
// ============================================================================

console.log('=== 示例 6：检测数组是否有序 ===');

function isSorted<T>(arr: T[], cmp: (a: T, b: T) => number): boolean {
  // 利用冒泡排序的特性：如果第一轮没有交换，则已有序
  for (let i = 0; i < arr.length - 1; i++) {
    if (cmp(arr[i], arr[i + 1]) > 0) {
      return false;
    }
  }
  return true;
}

console.log('[1,2,3,4,5] 是否有序:', isSorted([1, 2, 3, 4, 5], (a, b) => a - b));
console.log('[1,3,2,4,5] 是否有序:', isSorted([1, 3, 2, 4, 5], (a, b) => a - b));

console.log();

// ============================================================================
// 元信息展示
// ============================================================================

console.log('=== 算法元信息 ===');
console.log('名称:', meta.name);
console.log('稳定性:', meta.stable ? '稳定' : '不稳定');
console.log('原地:', meta.inPlace ? '是' : '否');
console.log('时间复杂度:', meta.timeComplexity);
console.log('空间复杂度:', meta.spaceComplexity);
console.log('适用场景:', meta.适用场景);
console.log('不适用场景:', meta.不适用场景);
