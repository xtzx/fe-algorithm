/**
 * 希尔排序测试用例
 */

import {
  sort,
  sortInPlace,
  sortWithGaps,
  getKnuthGaps,
  getShellGaps,
  getHibbardGaps,
  getSedgewickGaps,
  meta
} from '../src/index';
import { assertSort } from '../../../公共库/src/正确性校验';
import { numberAsc, numberDesc } from '../../../公共库/src/比较器';
import {
  generateNumbers,
  generateBoundaryTestCases
} from '../../../公共库/src/数据生成器';

// ============================================================================
// 测试工具
// ============================================================================

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`✅ ${name}`);
    passed++;
  } catch (error) {
    console.error(`❌ ${name}`);
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
// 测试用例
// ============================================================================

console.log('=== 希尔排序测试 ===\n');

// ---------- 边界情况 ----------

test('空数组', () => {
  const arr: number[] = [];
  const sorted = sort(arr, numberAsc);
  assert(sorted.length === 0, '空数组排序后应为空');
});

test('单元素', () => {
  const arr = [42];
  const sorted = sort(arr, numberAsc);
  assert(sorted.length === 1 && sorted[0] === 42, '单元素数组应不变');
});

test('两元素-已序', () => {
  const arr = [1, 2];
  const sorted = sort(arr, numberAsc);
  assertSort([1, 2], sorted, numberAsc);
});

test('两元素-逆序', () => {
  const arr = [2, 1];
  const sorted = sort(arr, numberAsc);
  assertSort([2, 1], sorted, numberAsc);
});

// ---------- 标准情况 ----------

test('已排序数组', () => {
  const arr = [1, 2, 3, 4, 5];
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('逆序数组', () => {
  const arr = [5, 4, 3, 2, 1];
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('重复元素', () => {
  const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('全相同元素', () => {
  const arr = [7, 7, 7, 7, 7];
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('随机数组 (n=100)', () => {
  const arr = generateNumbers(100, 'random');
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('随机数组 (n=1000)', () => {
  const arr = generateNumbers(1000, 'random');
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('近乎有序数组', () => {
  const arr = generateNumbers(100, 'nearlySorted');
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

test('重复多的数组', () => {
  const arr = generateNumbers(100, 'fewUnique');
  const sorted = sort(arr, numberAsc);
  assertSort(arr, sorted, numberAsc);
});

// ---------- 降序排序 ----------

test('降序排序', () => {
  const arr = [1, 5, 2, 8, 3];
  const sorted = sort(arr, numberDesc);
  assertSort(arr, sorted, numberDesc);
});

// ---------- 原地排序 ----------

test('sortInPlace 修改原数组', () => {
  const arr = [3, 1, 2];
  const result = sortInPlace(arr, numberAsc);
  assert(arr === result, '应返回同一数组引用');
  assertSort([3, 1, 2], arr, numberAsc);
});

test('sort 不修改原数组', () => {
  const arr = [3, 1, 2];
  const original = [...arr];
  sort(arr, numberAsc);
  assert(
    arr.every((v, i) => v === original[i]),
    'sort 不应修改原数组'
  );
});

// ---------- 间隔序列测试 ----------

test('Knuth 间隔序列生成', () => {
  const gaps = getKnuthGaps(100);
  assert(gaps.length > 0, '应生成非空序列');
  assert(gaps[gaps.length - 1] === 1, '最后一个间隔应为 1');
  assert(gaps[0] > gaps[gaps.length - 1], '间隔应从大到小');
});

test('Shell 间隔序列生成', () => {
  const gaps = getShellGaps(100);
  assert(gaps.length > 0, '应生成非空序列');
  assert(gaps[gaps.length - 1] === 1, '最后一个间隔应为 1');
});

test('Hibbard 间隔序列生成', () => {
  const gaps = getHibbardGaps(100);
  assert(gaps.length > 0, '应生成非空序列');
  assert(gaps[gaps.length - 1] === 1, '最后一个间隔应为 1');
});

test('Sedgewick 间隔序列生成', () => {
  const gaps = getSedgewickGaps(100);
  assert(gaps.length > 0, '应生成非空序列');
  assert(gaps[gaps.length - 1] === 1, '最后一个间隔应为 1');
});

// ---------- 不同间隔序列排序 ----------

test('使用 Shell 序列排序', () => {
  const arr = generateNumbers(100, 'random');
  const gaps = getShellGaps(100);
  const sorted = sortWithGaps([...arr], numberAsc, gaps);
  assertSort(arr, sorted, numberAsc);
});

test('使用 Hibbard 序列排序', () => {
  const arr = generateNumbers(100, 'random');
  const gaps = getHibbardGaps(100);
  const sorted = sortWithGaps([...arr], numberAsc, gaps);
  assertSort(arr, sorted, numberAsc);
});

test('使用 Sedgewick 序列排序', () => {
  const arr = generateNumbers(100, 'random');
  const gaps = getSedgewickGaps(100);
  const sorted = sortWithGaps([...arr], numberAsc, gaps);
  assertSort(arr, sorted, numberAsc);
});

// ---------- 对象排序 ----------

test('对象数组排序', () => {
  interface User { name: string; age: number; }

  const users: User[] = [
    { name: 'Charlie', age: 30 },
    { name: 'Alice', age: 25 },
    { name: 'Bob', age: 35 },
  ];

  const cmp = (a: User, b: User) => a.age - b.age;
  const sorted = sort(users, cmp);

  assert(sorted[0].name === 'Alice', '最年轻的应该是 Alice');
  assert(sorted[2].name === 'Bob', '最年长的应该是 Bob');
});

// ---------- 元信息验证 ----------

test('元信息正确', () => {
  assert(meta.name === '希尔排序', 'name 应为希尔排序');
  assert(meta.stable === false, 'stable 应为 false');
  assert(meta.inPlace === true, 'inPlace 应为 true');
});

// ---------- 边界测试集 ----------

const boundaryCases = generateBoundaryTestCases();
for (const { name, data } of boundaryCases) {
  test(`边界: ${name}`, () => {
    const sorted = sort(data, numberAsc);
    assertSort(data, sorted, numberAsc);
  });
}

// ============================================================================
// 测试结果
// ============================================================================

console.log('\n' + '='.repeat(40));
console.log(`测试完成: ${passed} 通过, ${failed} 失败`);
if (failed > 0) {
  process.exit(1);
}

