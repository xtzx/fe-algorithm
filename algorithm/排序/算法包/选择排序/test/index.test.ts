/**
 * 选择排序测试用例
 *
 * 使用公共库的正确性校验，纯手写断言
 */

import { sort, sortInPlace, sortBidirectional, sortWithStats, meta } from '../src/index';
import { numberAsc, numberDesc } from '../../公共库/src/比较器';
import {
  verifySorted,
  verifyPermutation
} from '../../公共库/src/正确性校验';
import {
  generateNumbers,
  generateBoundaryTestCases
} from '../../公共库/src/数据生成器';

// ============================================================================
// 断言工具
// ============================================================================

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string): void {
  if (condition) {
    passed++;
    console.log(`✅ ${message}`);
  } else {
    failed++;
    console.log(`❌ ${message}`);
  }
}

function assertEqual<T>(actual: T, expected: T, message: string): void {
  const equal = JSON.stringify(actual) === JSON.stringify(expected);
  assert(equal, `${message} (got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)})`);
}

// ============================================================================
// 基础测试
// ============================================================================

console.log('\n=== 基础测试 ===\n');

// 空数组
assertEqual(sort([], numberAsc), [], '空数组排序');

// 单元素
assertEqual(sort([1], numberAsc), [1], '单元素排序');

// 两元素 - 已排序
assertEqual(sort([1, 2], numberAsc), [1, 2], '两元素已排序');

// 两元素 - 逆序
assertEqual(sort([2, 1], numberAsc), [1, 2], '两元素逆序');

// 全相同
assertEqual(sort([5, 5, 5, 5], numberAsc), [5, 5, 5, 5], '全相同元素');

// 已排序
assertEqual(sort([1, 2, 3, 4, 5], numberAsc), [1, 2, 3, 4, 5], '已排序数组');

// 完全逆序
assertEqual(sort([5, 4, 3, 2, 1], numberAsc), [1, 2, 3, 4, 5], '完全逆序');

// 降序排序
assertEqual(sort([1, 2, 3, 4, 5], numberDesc), [5, 4, 3, 2, 1], '降序排序');

// ============================================================================
// 正确性校验测试
// ============================================================================

console.log('\n=== 正确性校验测试 ===\n');

// 随机数据
const randomData = generateNumbers(100, 'random');
const sortedRandom = sort(randomData, numberAsc);

const sortedResult = verifySorted(sortedRandom, numberAsc);
assert(sortedResult.passed, '随机数据排序 - 有序性');

const permResult = verifyPermutation(randomData, sortedRandom);
assert(permResult.passed, '随机数据排序 - 置换性');

// 重复多数据
const fewUniqueData = generateNumbers(100, 'fewUnique');
const sortedFewUnique = sort(fewUniqueData, numberAsc);

assert(verifySorted(sortedFewUnique, numberAsc).passed, '重复多数据 - 有序性');
assert(verifyPermutation(fewUniqueData, sortedFewUnique).passed, '重复多数据 - 置换性');

// 近乎有序数据
const nearlySortedData = generateNumbers(100, 'nearlySorted');
const sortedNearly = sort(nearlySortedData, numberAsc);

assert(verifySorted(sortedNearly, numberAsc).passed, '近乎有序数据 - 有序性');

// ============================================================================
// 不稳定性验证
// ============================================================================

console.log('\n=== 不稳定性验证 ===\n');

// 选择排序是不稳定的，这里验证元信息正确标记
assert(meta.stable === false, '元信息正确标记为不稳定');

// 演示不稳定性案例
interface Item { key: number; id: string; }
const unstableCase: Item[] = [
  { key: 3, id: 'a' },
  { key: 1, id: 'b' },
  { key: 3, id: 'c' },
];

const sortedUnstable = sort(unstableCase, (a, b) => a.key - b.key);

// 验证排序正确性（有序）
assert(
  sortedUnstable.every((item, i) => i === 0 || sortedUnstable[i-1].key <= item.key),
  '不稳定性测试 - 排序结果有序'
);

// 注：选择排序不保证稳定性，key=3 的两个元素顺序可能变化
console.log('  注：选择排序不保证稳定性，相等元素顺序可能改变');

// ============================================================================
// 交换次数测试
// ============================================================================

console.log('\n=== 交换次数测试 ===\n');

const swapTestData = generateNumbers(50, 'random');
const { result, swaps, comparisons } = sortWithStats(swapTestData, numberAsc);

assert(verifySorted(result, numberAsc).passed, '带统计排序 - 结果正确');
assert(swaps <= swapTestData.length - 1, `交换次数 <= n-1 (实际: ${swaps})`);
console.log(`  比较次数: ${comparisons}, 交换次数: ${swaps}`);

// 已排序数组应该零交换
const sortedData = [1, 2, 3, 4, 5];
const { swaps: zeroSwaps } = sortWithStats(sortedData, numberAsc);
assertEqual(zeroSwaps, 0, '已排序数组零交换');

// ============================================================================
// 双向选择排序测试
// ============================================================================

console.log('\n=== 双向选择排序测试 ===\n');

const bidirData = generateNumbers(50, 'random');
const bidirResult = sortBidirectional(bidirData, numberAsc);

assert(verifySorted(bidirResult, numberAsc).passed, '双向选择排序 - 有序性');
assert(verifyPermutation(bidirData, bidirResult).passed, '双向选择排序 - 置换性');

// ============================================================================
// 原地排序测试
// ============================================================================

console.log('\n=== 原地排序测试 ===\n');

const inPlaceArr = [5, 3, 8, 4, 2];
const originalRef = inPlaceArr;
sortInPlace(inPlaceArr, numberAsc);

assertEqual(inPlaceArr, [2, 3, 4, 5, 8], '原地排序结果正确');
assert(inPlaceArr === originalRef, '原地排序返回同一引用');

// ============================================================================
// 边界测试
// ============================================================================

console.log('\n=== 边界测试 ===\n');

const boundaryCases = generateBoundaryTestCases();
for (const { name, data } of boundaryCases) {
  const sorted = sort(data, numberAsc);
  const result = verifySorted(sorted, numberAsc);
  assert(result.passed, `边界测试 - ${name}`);
}

// ============================================================================
// 元信息验证
// ============================================================================

console.log('\n=== 元信息验证 ===\n');

assert(meta.stable === false, '元信息 - 不稳定');
assert(meta.inPlace === true, '元信息 - 原地排序');
assert(meta.timeComplexity.best === 'O(n²)', '元信息 - 最好时间复杂度');
assert(meta.timeComplexity.average === 'O(n²)', '元信息 - 平均时间复杂度');
assert(meta.spaceComplexity === 'O(1)', '元信息 - 空间复杂度');

// ============================================================================
// 测试结果汇总
// ============================================================================

console.log('\n=== 测试结果 ===\n');
console.log(`通过: ${passed}`);
console.log(`失败: ${failed}`);
console.log(`总计: ${passed + failed}`);

if (failed > 0) {
  console.log('\n❌ 存在失败的测试用例');
  process.exit(1);
} else {
  console.log('\n✅ 所有测试通过');
}
