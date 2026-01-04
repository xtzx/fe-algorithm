/**
 * 三路快速排序 - 测试用例
 */

import { sort, sortInPlace, sortOptimized, sortWithStats, meta } from '../src/index';
import { verifySort } from '../../公共库/src/正确性校验';
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
    const result = verifySort(original, sorted, numberAsc, false);
    assert(result.passed, result.error || '未知错误');
  });
}

// ============================================================================
// 大量重复元素测试
// ============================================================================

console.log('\n--- 大量重复元素测试 ---');

test('全相同元素', () => {
  const arr = Array(100).fill(42);
  const original = [...arr];
  const sorted = sort(arr, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

test('只有 2 种值', () => {
  const arr = Array.from({ length: 100 }, () => (Math.random() > 0.5 ? 1 : 0));
  const original = [...arr];
  const sorted = sort(arr, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

test('只有 3 种值 (荷兰国旗问题)', () => {
  const arr = Array.from({ length: 100 }, () => Math.floor(Math.random() * 3));
  const original = [...arr];
  const sorted = sort(arr, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

test('fewUnique 分布 (10 种值)', () => {
  const data = generateNumbers(1000, 'fewUnique', { uniqueCount: 10 });
  const original = [...data];
  const sorted = sort(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// 各种数据分布测试
// ============================================================================

console.log('\n--- 数据分布测试 ---');

const distributions = ['random', 'sorted', 'reversed', 'nearlySorted'] as const;

for (const dist of distributions) {
  test(`分布: ${dist} (n=1000)`, () => {
    const data = generateNumbers(1000, dist);
    const original = [...data];
    const sorted = sort(data, numberAsc);
    const result = verifySort(original, sorted, numberAsc, false);
    assert(result.passed, result.error || '未知错误');
  });
}

// ============================================================================
// 原地排序测试
// ============================================================================

console.log('\n--- 原地排序测试 ---');

test('sortInPlace 修改原数组', () => {
  const arr = [5, 3, 5, 1, 5, 2, 5, 4];
  const original = [...arr];
  const result = sortInPlace(arr, numberAsc);
  assert(result === arr, '应返回同一引用');
  const verification = verifySort(original, arr, numberAsc, false);
  assert(verification.passed, verification.error || '未知错误');
});

// ============================================================================
// 优化版测试
// ============================================================================

console.log('\n--- 优化版测试 ---');

test('sortOptimized 正确性', () => {
  const data = generateNumbers(1000, 'fewUnique');
  const original = [...data];
  const sorted = sortOptimized(data, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// 统计测试
// ============================================================================

console.log('\n--- 统计测试 ---');

test('sortWithStats 返回正确统计', () => {
  const data = [5, 3, 5, 1, 5, 2];
  const { result, comparisons, swaps, recursions } = sortWithStats(
    data,
    numberAsc
  );

  assert(
    JSON.stringify(result) === JSON.stringify([1, 2, 3, 5, 5, 5]),
    '排序结果错误'
  );
  assert(comparisons > 0, '比较次数应大于 0');
  assert(swaps >= 0, '交换次数应 >= 0');
  assert(recursions >= 0, '递归次数应 >= 0');
});

test('全相同元素的递归次数最少', () => {
  const allSame = Array(100).fill(5);
  const { recursions } = sortWithStats(allSame, numberAsc);

  // 全相同元素时，只需要 1 次递归（没有左右子问题）
  assert(recursions <= 2, `全相同元素递归次数应很少，实际 ${recursions}`);
});

// ============================================================================
// 对象排序测试
// ============================================================================

console.log('\n--- 对象排序测试 ---');

test('状态码排序', () => {
  type Status = 'pending' | 'completed' | 'failed';
  const statusOrder: Record<Status, number> = {
    pending: 0,
    completed: 1,
    failed: 2,
  };

  interface Task {
    id: number;
    status: Status;
  }

  const tasks: Task[] = [
    { id: 1, status: 'completed' },
    { id: 2, status: 'pending' },
    { id: 3, status: 'failed' },
    { id: 4, status: 'pending' },
    { id: 5, status: 'completed' },
  ];

  const cmp = (a: Task, b: Task) => statusOrder[a.status] - statusOrder[b.status];
  const sorted = sort(tasks, cmp);

  // 验证顺序
  for (let i = 1; i < sorted.length; i++) {
    assert(
      statusOrder[sorted[i - 1].status] <= statusOrder[sorted[i].status],
      '状态顺序错误'
    );
  }
});

// ============================================================================
// 性能测试：全相同 vs 随机
// ============================================================================

console.log('\n--- 性能对比：全相同 vs 随机 ---');

test('全相同元素比较次数是 O(n)', () => {
  const allSame = Array(1000).fill(5);
  const { comparisons: sameComparisons } = sortWithStats(allSame, numberAsc);

  const random = generateNumbers(1000, 'random');
  const { comparisons: randomComparisons } = sortWithStats(random, numberAsc);

  console.log(`   全相同比较次数: ${sameComparisons}`);
  console.log(`   随机数据比较次数: ${randomComparisons}`);

  // 全相同应该比随机少很多
  assert(
    sameComparisons < randomComparisons,
    '全相同元素比较次数应少于随机数据'
  );
});

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assert(meta.stable === false, '应为不稳定排序');
  assert(meta.inPlace === true, '应为原地排序');
  assert(meta.特点 === '全相同元素时 O(n)', '特点描述');
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

