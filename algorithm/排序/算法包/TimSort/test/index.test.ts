/**
 * TimSort - 测试用例
 */

import { sort, sortInPlace, sortWithStats, meta } from '../src/index';
import { verifySort, verifyStable } from '../../公共库/src/正确性校验';
import { numberAsc, byField } from '../../公共库/src/比较器';
import {
  generateNumbers,
  generateBoundaryTestCases,
} from '../../公共库/src/数据生成器';

console.log(`\n===== ${meta.name} 测试 =====\n`);

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`✅ ${name}`);
    passed++;
  } catch (error) {
    console.log(`❌ ${name}`);
    console.error(`   ${(error as Error).message}`);
    failed++;
  }
}

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

// ============================================================================
// 边界测试
// ============================================================================

console.log('--- 边界测试 ---');

const boundaryTests = generateBoundaryTestCases();

for (const { name, data } of boundaryTests) {
  test(`边界: ${name}`, () => {
    const original = [...data];
    const sorted = sort(data, numberAsc);
    const result = verifySort(original, sorted, numberAsc, true); // 检查稳定性
    assert(result.passed, result.error || '未知错误');
  });
}

// ============================================================================
// 各种数据分布测试
// ============================================================================

console.log('\n--- 数据分布测试 ---');

const distributions = ['random', 'sorted', 'reversed', 'nearlySorted', 'fewUnique'] as const;

for (const dist of distributions) {
  test(`分布: ${dist} (n=1000)`, () => {
    const data = generateNumbers(1000, dist);
    const original = [...data];
    const sorted = sort(data, numberAsc);
    const result = verifySort(original, sorted, numberAsc, true);
    assert(result.passed, result.error || '未知错误');
  });
}

// ============================================================================
// 稳定性测试 ⭐
// ============================================================================

console.log('\n--- 稳定性测试 ⭐ ---');

test('稳定性: 相等元素保持原顺序', () => {
  interface Item {
    value: number;
    originalIndex: number;
  }

  const items: Item[] = [
    { value: 3, originalIndex: 0 },
    { value: 1, originalIndex: 1 },
    { value: 3, originalIndex: 2 },
    { value: 2, originalIndex: 3 },
    { value: 3, originalIndex: 4 },
    { value: 1, originalIndex: 5 },
  ];

  const cmp = (a: Item, b: Item) => a.value - b.value;
  const sorted = sort(items, cmp);

  // 检查 value=3 的元素是否保持原顺序
  const threes = sorted.filter((item) => item.value === 3);
  assert(threes[0].originalIndex === 0, '第一个 3 应是 originalIndex=0');
  assert(threes[1].originalIndex === 2, '第二个 3 应是 originalIndex=2');
  assert(threes[2].originalIndex === 4, '第三个 3 应是 originalIndex=4');

  // 检查 value=1 的元素
  const ones = sorted.filter((item) => item.value === 1);
  assert(ones[0].originalIndex === 1, '第一个 1 应是 originalIndex=1');
  assert(ones[1].originalIndex === 5, '第二个 1 应是 originalIndex=5');
});

test('多列排序稳定性', () => {
  interface Employee {
    name: string;
    department: string;
    salary: number;
  }

  const employees: Employee[] = [
    { name: 'Alice', department: 'Dev', salary: 80000 },
    { name: 'Bob', department: 'HR', salary: 60000 },
    { name: 'Charlie', department: 'Dev', salary: 75000 },
    { name: 'David', department: 'HR', salary: 60000 },
  ];

  // 先按薪资排序
  let sorted = sort(employees, (a, b) => a.salary - b.salary);

  // 再按部门排序
  sorted = sort(sorted, (a, b) => a.department.localeCompare(b.department));

  // Dev 部门的员工应保持薪资顺序
  const devs = sorted.filter((e) => e.department === 'Dev');
  assert(devs[0].salary <= devs[1].salary, 'Dev 部门应保持薪资顺序');

  // HR 部门的同薪资员工应保持原顺序
  const hrs = sorted.filter((e) => e.department === 'HR');
  assert(hrs[0].name === 'Bob', 'HR 同薪资第一个应是 Bob');
  assert(hrs[1].name === 'David', 'HR 同薪资第二个应是 David');
});

// ============================================================================
// 近乎有序数据性能
// ============================================================================

console.log('\n--- 近乎有序数据性能 ---');

test('完全有序数据 run 数量最少', () => {
  const ordered = Array.from({ length: 1000 }, (_, i) => i);
  const { runs } = sortWithStats(ordered, numberAsc);

  // 完全有序应该只有 1 个 run（可能因 minrun 被分成几个）
  assert(runs <= 33, `完全有序的 run 数量应很少，实际 ${runs}`);
});

test('逆序数据能正确处理（反转 run）', () => {
  const reversed = Array.from({ length: 100 }, (_, i) => 100 - i);
  const original = [...reversed];
  const sorted = sort(reversed, numberAsc);
  const result = verifySort(original, sorted, numberAsc, true);
  assert(result.passed, result.error || '未知错误');
});

test('近乎有序数据性能优于随机', () => {
  const nearlySorted = generateNumbers(5000, 'nearlySorted', { swapPercent: 2 });
  const random = generateNumbers(5000, 'random');

  const nearlyStats = sortWithStats(nearlySorted, numberAsc);
  const randomStats = sortWithStats(random, numberAsc);

  console.log(`   近乎有序: runs=${nearlyStats.runs}, merges=${nearlyStats.merges}`);
  console.log(`   完全随机: runs=${randomStats.runs}, merges=${randomStats.merges}`);

  // 近乎有序应该有更少的 run
  assert(nearlyStats.runs < randomStats.runs, '近乎有序的 run 数量应更少');
});

// ============================================================================
// 原地排序测试
// ============================================================================

console.log('\n--- 原地排序测试 ---');

test('sortInPlace 修改原数组', () => {
  const arr = [5, 3, 1, 4, 2];
  const original = [...arr];
  const result = sortInPlace(arr, numberAsc);
  assert(result === arr, '应返回同一引用');
  const verification = verifySort(original, arr, numberAsc, true);
  assert(verification.passed, verification.error || '未知错误');
});

// ============================================================================
// 大规模测试
// ============================================================================

console.log('\n--- 大规模测试 ---');

test('大规模近乎有序数据', () => {
  const data = generateNumbers(10000, 'nearlySorted');
  const original = [...data];

  const start = performance.now();
  const sorted = sort(data, numberAsc);
  const time = performance.now() - start;

  const result = verifySort(original, sorted, numberAsc, true);
  assert(result.passed, result.error || '未知错误');

  console.log(`   10000 近乎有序元素: ${time.toFixed(2)}ms`);
});

test('大规模随机数据', () => {
  const data = generateNumbers(10000, 'random');
  const original = [...data];

  const start = performance.now();
  const sorted = sort(data, numberAsc);
  const time = performance.now() - start;

  const result = verifySort(original, sorted, numberAsc, true);
  assert(result.passed, result.error || '未知错误');

  console.log(`   10000 随机元素: ${time.toFixed(2)}ms`);
});

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assert(meta.stable === true, '应为稳定排序');
  assert(meta.inPlace === false, '不是原地排序（需要辅助空间）');
  assert(meta.timeComplexity.best === 'O(n)', '最好情况应为 O(n)');
  assert(meta.spaceComplexity === 'O(n)', '空间复杂度');
});

// ============================================================================
// 总结
// ============================================================================

console.log(`\n===== 测试完成 =====`);
console.log(`✅ 通过: ${passed}`);
console.log(`❌ 失败: ${failed}`);

if (failed > 0) {
  process.exit(1);
}

