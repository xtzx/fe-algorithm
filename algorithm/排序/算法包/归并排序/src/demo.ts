/**
 * 归并排序 - 使用示例
 */

import { sort, sortIterative, merge, mergeKSorted, meta } from './index';
import {
  numberAsc,
  stringAsc,
  byField,
  compose,
  reverse,
} from '../../公共库/src/比较器';

console.log(`\n===== ${meta.name} Demo =====\n`);

// ============================================================================
// 1. 基础数字排序
// ============================================================================

console.log('--- 1. 基础数字排序 ---');

const numbers = [38, 27, 43, 3, 9, 82, 10];
console.log('原始数组:', numbers);

const sortedNumbers = sort(numbers, numberAsc);
console.log('升序排序:', sortedNumbers);
console.log('原数组未变:', numbers);

// ============================================================================
// 2. 稳定性演示
// ============================================================================

console.log('\n--- 2. 稳定性演示 ---');

interface Student {
  name: string;
  score: number;
  order: number; // 原始顺序标记
}

const students: Student[] = [
  { name: 'Alice', score: 85, order: 1 },
  { name: 'Bob', score: 90, order: 2 },
  { name: 'Charlie', score: 85, order: 3 },
  { name: 'David', score: 90, order: 4 },
  { name: 'Eve', score: 85, order: 5 },
];

console.log('原始顺序:');
students.forEach((s) => console.log(`  ${s.name}: ${s.score} (order: ${s.order})`));

// 按分数排序
const byScore = byField<Student, 'score'>('score', numberAsc);
const sortedStudents = sort(students, byScore);

console.log('\n按分数排序后（观察相同分数的相对顺序）:');
sortedStudents.forEach((s) =>
  console.log(`  ${s.name}: ${s.score} (order: ${s.order})`)
);
console.log('✅ 相同分数的学生保持原始相对顺序（稳定排序）');

// ============================================================================
// 3. 表格多列排序
// ============================================================================

console.log('\n--- 3. 表格多列排序 ---');

interface TableRow {
  id: number;
  name: string;
  department: string;
  salary: number;
}

const employees: TableRow[] = [
  { id: 1, name: 'Alice', department: 'HR', salary: 60000 },
  { id: 2, name: 'Bob', department: 'Dev', salary: 80000 },
  { id: 3, name: 'Charlie', department: 'HR', salary: 55000 },
  { id: 4, name: 'David', department: 'Dev', salary: 75000 },
  { id: 5, name: 'Eve', department: 'HR', salary: 60000 },
];

console.log('原始数据:', JSON.stringify(employees, null, 2));

// 先按部门升序，同部门按薪资降序
const multiSort = compose(
  byField<TableRow, 'department'>('department', stringAsc),
  byField<TableRow, 'salary'>('salary', reverse(numberAsc))
);

const sortedEmployees = sort(employees, multiSort);
console.log('\n按部门升序，同部门按薪资降序:');
sortedEmployees.forEach((e) =>
  console.log(`  ${e.department} | ${e.name} | $${e.salary}`)
);

// ============================================================================
// 4. 搜索结果排序
// ============================================================================

console.log('\n--- 4. 搜索结果排序 ---');

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

console.log('原始搜索结果:');
searchResults.forEach((r) =>
  console.log(`  [${r.relevance}] ${r.title}`)
);

// 按相关度降序，同相关度按时间降序
const searchSort = compose(
  byField<SearchResult, 'relevance'>('relevance', reverse(numberAsc)),
  byField<SearchResult, 'publishTime'>('publishTime', reverse(numberAsc))
);

const sortedResults = sort(searchResults, searchSort);
console.log('\n按相关度降序，同分按时间降序:');
sortedResults.forEach((r) =>
  console.log(`  [${r.relevance}] ${r.title}`)
);

// ============================================================================
// 5. 合并有序数组
// ============================================================================

console.log('\n--- 5. 合并有序数组 ---');

const sorted1 = [1, 3, 5, 7];
const sorted2 = [2, 4, 6, 8];
console.log('数组 1:', sorted1);
console.log('数组 2:', sorted2);

const merged = merge(sorted1, sorted2, numberAsc);
console.log('合并结果:', merged);

// ============================================================================
// 6. 合并 K 个有序数组
// ============================================================================

console.log('\n--- 6. 合并 K 个有序数组 ---');

const arrays = [
  [1, 4, 7],
  [2, 5, 8],
  [3, 6, 9],
  [0, 10],
];
console.log('K 个有序数组:', arrays);

const mergedK = mergeKSorted(arrays, numberAsc);
console.log('合并结果:', mergedK);

// ============================================================================
// 7. 迭代版（大数组）
// ============================================================================

console.log('\n--- 7. 迭代版性能测试 ---');

const largeArray = Array.from({ length: 10000 }, () =>
  Math.floor(Math.random() * 10000)
);

console.time('递归版');
sort(largeArray, numberAsc);
console.timeEnd('递归版');

console.time('迭代版');
sortIterative(largeArray, numberAsc);
console.timeEnd('迭代版');

console.log('\n===== Demo 完成 =====');

