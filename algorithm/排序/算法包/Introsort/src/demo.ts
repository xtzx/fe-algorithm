/**
 * Introsort - 使用示例
 */

import { sort, sortWithStats, meta } from './index';
import { numberAsc, byField, reverse } from '../../公共库/src/比较器';
import { generateNumbers } from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} Demo =====\n`);

// ============================================================================
// 1. 基础排序
// ============================================================================

console.log('--- 1. 基础排序 ---');

const numbers = [38, 27, 43, 3, 9, 82, 10];
console.log('原始数组:', numbers);

const sorted = sort(numbers, numberAsc);
console.log('排序结果:', sorted);

// ============================================================================
// 2. 算法组成统计
// ============================================================================

console.log('\n--- 2. 算法组成统计 ---');

const randomData = generateNumbers(1000, 'random');
const { result, insertionCalls, heapCalls, quickCalls, comparisons } =
  sortWithStats(randomData, numberAsc);

console.log(`数组大小: ${randomData.length}`);
console.log(`比较次数: ${comparisons}`);
console.log(`插入排序调用次数: ${insertionCalls}`);
console.log(`堆排序调用次数: ${heapCalls}`);
console.log(`快排调用次数: ${quickCalls}`);
console.log(`排序结果（前10个）: [${result.slice(0, 10).join(', ')}...]`);

// ============================================================================
// 3. 防止最坏情况演示
// ============================================================================

console.log('\n--- 3. 防止最坏情况演示 ---');

// 有序数据是快排的最坏情况
const sortedData = generateNumbers(5000, 'sorted');
console.log('有序数据（5000 个元素）- 快排最坏情况:');

const { heapCalls: heapForSorted, quickCalls: quickForSorted } = sortWithStats(
  sortedData,
  numberAsc
);

console.log(`  堆排序调用次数: ${heapForSorted}`);
console.log(`  快排调用次数: ${quickForSorted}`);

if (heapForSorted > 0) {
  console.log('  ⭐ 堆排序被触发，避免了 O(n²)！');
} else {
  console.log('  三数取中足以处理这种情况');
}

// ============================================================================
// 4. 逆序数据
// ============================================================================

console.log('\n--- 4. 逆序数据 ---');

const reversedData = generateNumbers(5000, 'reversed');
console.log('逆序数据（5000 个元素）:');

const { heapCalls: heapForReversed, quickCalls: quickForReversed } =
  sortWithStats(reversedData, numberAsc);

console.log(`  堆排序调用次数: ${heapForReversed}`);
console.log(`  快排调用次数: ${quickForReversed}`);

// ============================================================================
// 5. 不同数据分布的表现
// ============================================================================

console.log('\n--- 5. 不同数据分布的表现 ---');

const distributions = ['random', 'sorted', 'reversed', 'nearlySorted', 'fewUnique'] as const;

for (const dist of distributions) {
  const data = generateNumbers(10000, dist);

  console.time(`  ${dist}`);
  sort(data, numberAsc);
  console.timeEnd(`  ${dist}`);
}

console.log('\n⭐ 所有分布都是 O(n log n)，不会退化！');

// ============================================================================
// 6. 表格数据排序
// ============================================================================

console.log('\n--- 6. 表格数据排序 ---');

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

const byScore = (a: TableRow, b: TableRow) => b.score - a.score; // 降序
const sortedEmployees = sort(employees, byScore);

console.log('\n按分数降序:');
sortedEmployees.forEach((e) => console.log(`  ${e.name}: ${e.score}`));

// ============================================================================
// 7. 大规模数据
// ============================================================================

console.log('\n--- 7. 大规模数据 ---');

const sizes = [10000, 50000, 100000];

for (const size of sizes) {
  const data = generateNumbers(size, 'random');

  console.time(`  ${size} 个元素`);
  sort(data, numberAsc);
  console.timeEnd(`  ${size} 个元素`);
}

// ============================================================================
// 8. 元信息展示
// ============================================================================

console.log('\n--- 8. 算法信息 ---');
console.log(`名称: ${meta.name}`);
console.log(`组成: ${meta.组成.join(' + ')}`);
console.log(`插入排序阈值: ${meta.阈值.insertionThreshold}`);
console.log(`深度限制: ${meta.阈值.depthLimit}`);
console.log(`使用者: ${meta.使用者.join(', ')}`);

console.log('\n===== Demo 完成 =====');

