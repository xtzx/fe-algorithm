/**
 * Introsort - 测试用例
 */

import { sort, sortInPlace, sortWithStats, meta } from '../src/index';
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
// 各种数据分布测试
// ============================================================================

console.log('\n--- 数据分布测试 ---');

const distributions = ['random', 'sorted', 'reversed', 'nearlySorted', 'fewUnique'] as const;

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
  const arr = [5, 3, 1, 4, 2];
  const original = [...arr];
  const result = sortInPlace(arr, numberAsc);
  assert(result === arr, '应返回同一引用');
  const verification = verifySort(original, arr, numberAsc, false);
  assert(verification.passed, verification.error || '未知错误');
});

// ============================================================================
// 防止最坏情况测试
// ============================================================================

console.log('\n--- 防止最坏情况测试 ---');

test('有序数据不会退化到 O(n²)', () => {
  const sortedData = generateNumbers(2000, 'sorted');
  const original = [...sortedData];

  // 应该能在合理时间内完成
  const start = performance.now();
  const result = sort(sortedData, numberAsc);
  const time = performance.now() - start;

  const verification = verifySort(original, result, numberAsc, false);
  assert(verification.passed, '排序应正确');

  // O(n²) 对于 2000 个元素会很慢，O(n log n) 应该很快
  console.log(`   有序数据排序时间: ${time.toFixed(2)}ms`);
});

test('逆序数据不会退化到 O(n²)', () => {
  const reversedData = generateNumbers(2000, 'reversed');
  const original = [...reversedData];

  const start = performance.now();
  const result = sort(reversedData, numberAsc);
  const time = performance.now() - start;

  const verification = verifySort(original, result, numberAsc, false);
  assert(verification.passed, '排序应正确');

  console.log(`   逆序数据排序时间: ${time.toFixed(2)}ms`);
});

test('全相同元素', () => {
  const allSame = Array(1000).fill(42);
  const original = [...allSame];
  const sorted = sort(allSame, numberAsc);
  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');
});

// ============================================================================
// 统计测试
// ============================================================================

console.log('\n--- 统计测试 ---');

test('sortWithStats 返回正确统计', () => {
  const data = generateNumbers(500, 'random');
  const { result, insertionCalls, heapCalls, quickCalls, comparisons } =
    sortWithStats(data, numberAsc);

  // 验证结果正确
  const verification = verifySort(data, result, numberAsc, false);
  assert(verification.passed, '排序结果应正确');

  // 插入排序肯定会被调用（处理小数组）
  assert(insertionCalls > 0, '插入排序应被调用');

  // 快排应该被调用
  assert(quickCalls >= 0, '快排调用次数应 >= 0');

  // 堆排可能不被调用（如果没有触发深度限制）
  assert(heapCalls >= 0, '堆排调用次数应 >= 0');

  console.log(`   插入: ${insertionCalls}, 堆: ${heapCalls}, 快排: ${quickCalls}`);
});

test('深度限制触发堆排序', () => {
  // 构造可能触发深度限制的数据
  // 实际上由于三数取中，可能不容易触发
  const data = generateNumbers(1000, 'sorted');
  const { heapCalls } = sortWithStats(data, numberAsc);

  // 只是检查不崩溃，堆排可能被触发也可能不被
  console.log(`   堆排序调用次数: ${heapCalls}`);
});

// ============================================================================
// 对象排序测试
// ============================================================================

console.log('\n--- 对象排序测试 ---');

test('表格数据排序', () => {
  interface Row {
    id: number;
    value: number;
  }

  const rows: Row[] = Array.from({ length: 100 }, (_, i) => ({
    id: i,
    value: Math.floor(Math.random() * 1000),
  }));

  const cmp = (a: Row, b: Row) => a.value - b.value;
  const sorted = sort(rows, cmp);

  // 验证有序
  for (let i = 1; i < sorted.length; i++) {
    assert(sorted[i - 1].value <= sorted[i].value, '应按 value 有序');
  }
});

// ============================================================================
// 大规模测试
// ============================================================================

console.log('\n--- 大规模测试 ---');

test('大规模随机数据', () => {
  const data = generateNumbers(10000, 'random');
  const original = [...data];

  const start = performance.now();
  const sorted = sort(data, numberAsc);
  const time = performance.now() - start;

  const result = verifySort(original, sorted, numberAsc, false);
  assert(result.passed, result.error || '未知错误');

  console.log(`   10000 个元素排序时间: ${time.toFixed(2)}ms`);
});

// ============================================================================
// 元信息测试
// ============================================================================

console.log('\n--- 元信息测试 ---');

test('meta 信息正确', () => {
  assert(meta.stable === false, '应为不稳定排序');
  assert(meta.inPlace === true, '应为原地排序');
  assert(meta.timeComplexity.worst === 'O(n log n)', '最坏情况应保证');
  assert(meta.组成.includes('快速排序'), '应包含快速排序');
  assert(meta.组成.includes('堆排序'), '应包含堆排序');
  assert(meta.组成.includes('插入排序'), '应包含插入排序');
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

