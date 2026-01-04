/**
 * 冒泡排序测试用例
 *
 * 使用公共库的正确性校验，纯手写断言
 */

import { sort, sortInPlace, sortCocktail, meta } from '../src/index';
import { numberAsc, numberDesc } from '../../公共库/src/比较器';
import {
  verifySorted,
  verifyPermutation,
  verifyStable
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
// 稳定性测试
// ============================================================================

console.log('\n=== 稳定性测试 ===\n');

interface Item {
  key: number;
  id: string;
}

const stableTestData: Item[] = [
  { key: 3, id: 'a' },
  { key: 1, id: 'b' },
  { key: 2, id: 'c' },
  { key: 1, id: 'd' },
  { key: 3, id: 'e' },
  { key: 2, id: 'f' },
];

const cmpByKey = (a: Item, b: Item) => a.key - b.key;
const sortedStable = sort(stableTestData, cmpByKey);

// 验证稳定性
const stableResult = verifyStable(stableTestData, sortedStable, cmpByKey);
assert(stableResult.passed, '稳定性测试 - 相等元素保持原始顺序');

// 手动验证：key=1 的元素应该是 b, d 顺序
const key1Items = sortedStable.filter(item => item.key === 1);
assertEqual(key1Items.map(i => i.id), ['b', 'd'], 'key=1 的元素顺序正确');

// key=2 的元素应该是 c, f 顺序
const key2Items = sortedStable.filter(item => item.key === 2);
assertEqual(key2Items.map(i => i.id), ['c', 'f'], 'key=2 的元素顺序正确');

// key=3 的元素应该是 a, e 顺序
const key3Items = sortedStable.filter(item => item.key === 3);
assertEqual(key3Items.map(i => i.id), ['a', 'e'], 'key=3 的元素顺序正确');

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
// 鸡尾酒排序测试
// ============================================================================

console.log('\n=== 鸡尾酒排序测试 ===\n');

// 乌龟元素测试
const turtleCase = [2, 3, 4, 5, 1];
assertEqual(sortCocktail(turtleCase, numberAsc), [1, 2, 3, 4, 5], '鸡尾酒排序 - 乌龟元素');

// 随机数据
const cocktailRandom = generateNumbers(50, 'random');
const cocktailSorted = sortCocktail(cocktailRandom, numberAsc);
assert(verifySorted(cocktailSorted, numberAsc).passed, '鸡尾酒排序 - 随机数据');

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

assert(meta.stable === true, '元信息 - 稳定性正确');
assert(meta.inPlace === true, '元信息 - 原地排序正确');
assert(meta.timeComplexity.best === 'O(n)', '元信息 - 最好时间复杂度');
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
