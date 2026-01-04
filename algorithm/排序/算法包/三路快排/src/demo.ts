/**
 * 三路快速排序 - 使用示例
 */

import { sort, sortWithStats, meta } from './index';
import { numberAsc, byField, reverse } from '../../公共库/src/比较器';
import { generateNumbers } from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} Demo =====\n`);

// ============================================================================
// 1. 大量重复元素的优势
// ============================================================================

console.log('--- 1. 大量重复元素的优势 ---');

// 只有 3 种值的数组
const manyDuplicates = Array.from({ length: 1000 }, () =>
  Math.floor(Math.random() * 3)
);
console.log(`数组大小: ${manyDuplicates.length}, 只有 3 种不同值`);

const stats = sortWithStats(manyDuplicates, numberAsc);
console.log(`比较次数: ${stats.comparisons}`);
console.log(`交换次数: ${stats.swaps}`);
console.log(`递归次数: ${stats.recursions}`);
console.log(`排序结果（前20个）: [${stats.result.slice(0, 20).join(', ')}...]`);

// ============================================================================
// 2. 全相同元素的极端情况
// ============================================================================

console.log('\n--- 2. 全相同元素的极端情况 ---');

const allSame = Array(1000).fill(5);
console.log(`数组大小: ${allSame.length}, 全部相同`);

const sameStats = sortWithStats(allSame, numberAsc);
console.log(`比较次数: ${sameStats.comparisons}`);
console.log(`递归次数: ${sameStats.recursions}`);
console.log('⭐ 三路快排只需要 O(n) 次比较！');

// ============================================================================
// 3. 状态码/枚举值排序
// ============================================================================

console.log('\n--- 3. 状态码排序 ---');

type Status = 'pending' | 'processing' | 'completed' | 'failed';

interface Task {
  id: number;
  name: string;
  status: Status;
}

const statusOrder: Record<Status, number> = {
  pending: 0,
  processing: 1,
  completed: 2,
  failed: 3,
};

const tasks: Task[] = [
  { id: 1, name: 'Task A', status: 'completed' },
  { id: 2, name: 'Task B', status: 'pending' },
  { id: 3, name: 'Task C', status: 'failed' },
  { id: 4, name: 'Task D', status: 'pending' },
  { id: 5, name: 'Task E', status: 'completed' },
  { id: 6, name: 'Task F', status: 'pending' },
  { id: 7, name: 'Task G', status: 'processing' },
  { id: 8, name: 'Task H', status: 'completed' },
];

console.log('原始任务:');
tasks.forEach((t) => console.log(`  ${t.name}: ${t.status}`));

const statusCmp = (a: Task, b: Task) =>
  statusOrder[a.status] - statusOrder[b.status];
const sortedTasks = sort(tasks, statusCmp);

console.log('\n按状态排序后:');
sortedTasks.forEach((t) => console.log(`  ${t.name}: ${t.status}`));

// ============================================================================
// 4. 等级排序
// ============================================================================

console.log('\n--- 4. 学生等级排序 ---');

interface Student {
  name: string;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  score: number;
}

const gradeOrder = { A: 0, B: 1, C: 2, D: 3, F: 4 };

const students: Student[] = [
  { name: 'Alice', grade: 'B', score: 85 },
  { name: 'Bob', grade: 'A', score: 95 },
  { name: 'Charlie', grade: 'C', score: 75 },
  { name: 'David', grade: 'B', score: 88 },
  { name: 'Eve', grade: 'A', score: 92 },
  { name: 'Frank', grade: 'C', score: 72 },
  { name: 'Grace', grade: 'B', score: 82 },
];

console.log('原始学生:');
students.forEach((s) => console.log(`  ${s.name}: ${s.grade} (${s.score})`));

const gradeCmp = (a: Student, b: Student) =>
  gradeOrder[a.grade] - gradeOrder[b.grade];
const sortedStudents = sort(students, gradeCmp);

console.log('\n按等级排序后:');
sortedStudents.forEach((s) => console.log(`  ${s.name}: ${s.grade} (${s.score})`));

// ============================================================================
// 5. 表格多列排序
// ============================================================================

console.log('\n--- 5. 表格多列排序 ---');

interface TableRow {
  id: number;
  department: string;
  level: 'junior' | 'mid' | 'senior';
  salary: number;
}

const levelOrder = { junior: 0, mid: 1, senior: 2 };

const employees: TableRow[] = [
  { id: 1, department: 'Dev', level: 'senior', salary: 100000 },
  { id: 2, department: 'HR', level: 'mid', salary: 60000 },
  { id: 3, department: 'Dev', level: 'junior', salary: 50000 },
  { id: 4, department: 'Dev', level: 'mid', salary: 70000 },
  { id: 5, department: 'HR', level: 'senior', salary: 80000 },
  { id: 6, department: 'Dev', level: 'junior', salary: 55000 },
];

console.log('原始员工:');
employees.forEach((e) =>
  console.log(`  ${e.department} | ${e.level} | $${e.salary}`)
);

// 先按部门，同部门按级别
const multiCmp = (a: TableRow, b: TableRow) => {
  const deptDiff = a.department.localeCompare(b.department);
  if (deptDiff !== 0) return deptDiff;
  return levelOrder[a.level] - levelOrder[b.level];
};

const sortedEmployees = sort(employees, multiCmp);
console.log('\n按部门、级别排序后:');
sortedEmployees.forEach((e) =>
  console.log(`  ${e.department} | ${e.level} | $${e.salary}`)
);

// ============================================================================
// 6. 性能对比：fewUnique 分布
// ============================================================================

console.log('\n--- 6. 性能测试：fewUnique 分布 ---');

// 生成只有 10 种不同值的数据
const fewUniqueData = generateNumbers(50000, 'fewUnique', { uniqueCount: 10 });
console.log(`数组大小: ${fewUniqueData.length}, 约 10 种不同值`);

console.time('  三路快排');
sort(fewUniqueData, numberAsc);
console.timeEnd('  三路快排');

console.log('\n⭐ 三路快排在重复多的数据上表现优异！');

console.log('\n===== Demo 完成 =====');

