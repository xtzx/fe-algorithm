/**
 * 堆排序 - 使用示例
 */

import {
  sort,
  sortWithStats,
  findTopKLargest,
  findTopKSmallest,
  MaxPriorityQueue,
  MinPriorityQueue,
  meta,
} from './index';
import {
  numberAsc,
  byField,
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
// 2. TopK 问题
// ============================================================================

console.log('\n--- 2. TopK 问题 ---');

const data = [64, 34, 25, 12, 22, 11, 90, 45, 78, 33];
console.log('原始数据:', data);

const top3Largest = findTopKLargest(data, 3, numberAsc);
console.log('最大的 3 个:', top3Largest);

const top3Smallest = findTopKSmallest(data, 3, numberAsc);
console.log('最小的 3 个:', top3Smallest);

// ============================================================================
// 3. 优先队列使用
// ============================================================================

console.log('\n--- 3. 优先队列使用 ---');

// 任务调度示例
interface Task {
  name: string;
  priority: number;
}

const taskCmp = (a: Task, b: Task) => a.priority - b.priority;
const taskQueue = new MaxPriorityQueue<Task>(taskCmp);

taskQueue.push({ name: 'Low priority', priority: 1 });
taskQueue.push({ name: 'High priority', priority: 10 });
taskQueue.push({ name: 'Medium priority', priority: 5 });
taskQueue.push({ name: 'Urgent', priority: 15 });

console.log('任务队列大小:', taskQueue.size);

console.log('按优先级顺序处理任务:');
while (!taskQueue.isEmpty()) {
  const task = taskQueue.pop()!;
  console.log(`  处理: ${task.name} (优先级: ${task.priority})`);
}

// ============================================================================
// 4. 实时热榜示例
// ============================================================================

console.log('\n--- 4. 实时热榜示例 ---');

interface Article {
  id: string;
  title: string;
  heat: number;
}

const articles: Article[] = [
  { id: '1', title: 'Breaking News', heat: 1000 },
  { id: '2', title: 'Tech Update', heat: 500 },
  { id: '3', title: 'Sports Highlights', heat: 800 },
  { id: '4', title: 'Weather Report', heat: 200 },
  { id: '5', title: 'Market Analysis', heat: 750 },
  { id: '6', title: 'Celebrity Gossip', heat: 1200 },
  { id: '7', title: 'Science Discovery', heat: 600 },
];

const articleCmp = (a: Article, b: Article) => a.heat - b.heat;
const top5 = findTopKLargest(articles, 5, articleCmp);

console.log('热度 Top 5 文章:');
top5
  .sort((a, b) => b.heat - a.heat)
  .forEach((a, i) => console.log(`  ${i + 1}. ${a.title} (${a.heat})`));

// ============================================================================
// 5. 表格数据排序
// ============================================================================

console.log('\n--- 5. 表格数据排序 ---');

interface Student {
  name: string;
  score: number;
  grade: string;
}

const students: Student[] = [
  { name: 'Alice', score: 85, grade: 'B' },
  { name: 'Bob', score: 92, grade: 'A' },
  { name: 'Charlie', score: 78, grade: 'C' },
  { name: 'David', score: 92, grade: 'A' },
  { name: 'Eve', score: 88, grade: 'B' },
];

const byScore = byField<Student, 'score'>('score', reverse(numberAsc));
const sortedStudents = sort(students, byScore);

console.log('按分数降序:');
sortedStudents.forEach((s) => console.log(`  ${s.name}: ${s.score}`));

// ============================================================================
// 6. 排序统计
// ============================================================================

console.log('\n--- 6. 排序统计 ---');

const randomData = generateNumbers(100, 'random');
const { result, comparisons, swaps } = sortWithStats(randomData, numberAsc);

console.log(`数组大小: ${randomData.length}`);
console.log(`比较次数: ${comparisons}`);
console.log(`交换次数: ${swaps}`);
console.log(`排序结果（前10个）: [${result.slice(0, 10).join(', ')}...]`);

// ============================================================================
// 7. 不同数据分布的表现
// ============================================================================

console.log('\n--- 7. 不同数据分布的表现 ---');

const distributions = ['random', 'sorted', 'reversed', 'fewUnique'] as const;

for (const dist of distributions) {
  const testData = generateNumbers(10000, dist);

  console.time(`  ${dist}`);
  sort(testData, numberAsc);
  console.timeEnd(`  ${dist}`);
}

console.log('\n⭐ 注意：堆排序在所有分布下表现都很稳定！');

// ============================================================================
// 8. 与快排对比
// ============================================================================

console.log('\n--- 8. 与快排对比（有序数据）---');

const sortedData = generateNumbers(10000, 'sorted');

console.log('有序数据（10000 个元素）:');
console.time('  堆排序');
sort(sortedData, numberAsc);
console.timeEnd('  堆排序');

console.log('\n===== Demo 完成 =====');

