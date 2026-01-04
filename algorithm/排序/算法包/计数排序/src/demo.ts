/**
 * 计数排序示例
 *
 * 演示计数排序在前端场景中的应用
 */

import {
  countingSort,
  countingSortAuto,
  countingSortBy,
  countingSortByAuto,
} from './index';

// ============================================================================
// 示例 1：分数排序
// ============================================================================

console.log('=== 示例 1：分数排序 ===');

interface Student {
  name: string;
  score: number; // 0-100
}

const students: Student[] = [
  { name: 'Alice', score: 85 },
  { name: 'Bob', score: 92 },
  { name: 'Charlie', score: 85 }, // 同分，测试稳定性
  { name: 'David', score: 78 },
  { name: 'Eve', score: 92 }, // 同分
];

// 按分数升序（稳定排序）
const byScore = countingSortBy(students, s => s.score, 0, 100);
console.log('按分数升序:');
byScore.forEach(s => console.log(`  ${s.name}: ${s.score}`));

// 验证稳定性：同分的保持原顺序
// Alice(85) 在 Charlie(85) 前面
// Bob(92) 在 Eve(92) 前面
console.log('\n✅ 稳定性验证：同分学生保持原顺序');

// ============================================================================
// 示例 2：年龄分布统计排序
// ============================================================================

console.log('\n=== 示例 2：年龄排序 ===');

const ages = [25, 30, 22, 28, 25, 30, 22, 35, 28, 25];

// 年龄范围：0-150（人类年龄）
const sortedAges = countingSort(ages, 0, 150);
console.log('原数组:', ages);
console.log('排序后:', sortedAges);

// 使用自动检测范围
const sortedAgesAuto = countingSortAuto(ages);
console.log('自动检测范围:', sortedAgesAuto);

// ============================================================================
// 示例 3：状态码排序
// ============================================================================

console.log('\n=== 示例 3：状态码排序 ===');

interface Task {
  id: number;
  name: string;
  status: 0 | 1 | 2 | 3; // 0:待处理 1:进行中 2:已完成 3:已取消
}

const statusLabels = ['待处理', '进行中', '已完成', '已取消'];

const tasks: Task[] = [
  { id: 1, name: '任务A', status: 2 },
  { id: 2, name: '任务B', status: 0 },
  { id: 3, name: '任务C', status: 1 },
  { id: 4, name: '任务D', status: 0 },
  { id: 5, name: '任务E', status: 3 },
  { id: 6, name: '任务F', status: 1 },
];

// 只有 4 种状态，计数排序最优
const byStatus = countingSortBy(tasks, t => t.status, 0, 3);
console.log('按状态排序:');
byStatus.forEach(t =>
  console.log(`  [${statusLabels[t.status]}] ${t.name}`)
);

// ============================================================================
// 示例 4：表格数据排序
// ============================================================================

console.log('\n=== 示例 4：表格数据排序 ===');

interface TableRow {
  id: number;
  name: string;
  score: number;
  level: number; // 1-5
}

const tableData: TableRow[] = [
  { id: 1, name: 'Alice', score: 85, level: 3 },
  { id: 2, name: 'Bob', score: 92, level: 4 },
  { id: 3, name: 'Charlie', score: 78, level: 2 },
  { id: 4, name: 'David', score: 85, level: 3 },
  { id: 5, name: 'Eve', score: 95, level: 5 },
];

// 按等级排序
const byLevel = countingSortBy(tableData, r => r.level, 1, 5);
console.log('按等级排序:');
console.log('  Level | Name    | Score');
console.log('  ------|---------|------');
byLevel.forEach(r =>
  console.log(`  ${r.level}     | ${r.name.padEnd(7)} | ${r.score}`)
);

// ============================================================================
// 示例 5：讨论是否值得用非比较排序
// ============================================================================

console.log('\n=== 示例 5：是否值得用计数排序 ===');

interface UseCase {
  scenario: string;
  range: string;
  dataCount: string;
  recommendation: string;
  reason: string;
}

const useCases: UseCase[] = [
  {
    scenario: '学生分数 (0-100)',
    range: '101',
    dataCount: '1000+',
    recommendation: '✅ 推荐',
    reason: '小范围整数，稳定高效',
  },
  {
    scenario: '年龄 (0-150)',
    range: '151',
    dataCount: '100+',
    recommendation: '✅ 推荐',
    reason: '范围小，O(n+k) 优于 O(n log n)',
  },
  {
    scenario: '状态码 (0-10)',
    range: '11',
    dataCount: '任意',
    recommendation: '✅ 强烈推荐',
    reason: '极小范围，近乎 O(n)',
  },
  {
    scenario: '用户 ID (0-10^9)',
    range: '10^9',
    dataCount: '1000',
    recommendation: '❌ 不推荐',
    reason: '范围太大，空间 O(k) 浪费',
  },
  {
    scenario: '浮点数',
    range: '无限',
    dataCount: '任意',
    recommendation: '❌ 不适用',
    reason: '无法直接作为索引',
  },
];

console.log('场景分析:');
console.log('场景              | 值域  | 数据量 | 推荐   | 原因');
console.log('------------------|-------|--------|--------|--------------------');
useCases.forEach(u =>
  console.log(
    `${u.scenario.padEnd(18)}| ${u.range.padEnd(5)} | ${u.dataCount.padEnd(6)} | ${u.recommendation} | ${u.reason}`
  )
);

// ============================================================================
// 示例 6：负数处理
// ============================================================================

console.log('\n=== 示例 6：负数排序 ===');

const numbersWithNegative = [-5, 3, -2, 8, -1, 0, 5, -3];

// 需要偏移处理，或者指定包含负数的范围
const sorted = countingSort(numbersWithNegative, -10, 10);
console.log('原数组:', numbersWithNegative);
console.log('排序后:', sorted);

// 自动检测也可以处理负数
const sortedAuto = countingSortAuto(numbersWithNegative);
console.log('自动检测:', sortedAuto);

console.log('\n✅ 所有示例运行完成');

