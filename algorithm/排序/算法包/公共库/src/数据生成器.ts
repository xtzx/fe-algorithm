/**
 * 数据生成器模块
 *
 * 生成各种分布的测试数据，用于排序算法测试和基准测试。
 */

// ============================================================================
// 类型定义
// ============================================================================

/** 数据分布类型 */
export type Distribution =
  | 'random'        // 完全随机
  | 'sorted'        // 已排序（升序）
  | 'reversed'      // 完全逆序
  | 'nearlySorted'  // 近乎有序
  | 'fewUnique'     // 少量唯一值（重复多）
  | 'sawtooth';     // 锯齿形

/** 数字生成选项 */
export interface NumberOptions {
  min?: number;           // 最小值（默认 0）
  max?: number;           // 最大值（默认 10000）
  swapPercent?: number;   // nearlySorted 时交换比例（默认 5）
  uniqueCount?: number;   // fewUnique 时唯一值数量（默认 10）
  sawtoothPeriod?: number;// sawtooth 周期（默认 100）
}

/** 表格行数据 */
export interface TableRow {
  id: number;
  name: string;
  score: number;
  grade: string;
  timestamp: number;
}

/** 搜索结果数据 */
export interface SearchResult {
  id: string;
  title: string;
  relevance: number;
  publishTime: number;
  viewCount: number;
}

// ============================================================================
// 随机工具
// ============================================================================

/** 生成 [min, max) 范围的随机整数 */
function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min)) + min;
}

/** 生成随机字符串 */
function randomString(length: number): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars[randomInt(0, chars.length)];
  }
  return result;
}

/** Fisher-Yates 洗牌 */
function shuffle<T>(arr: T[]): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomInt(0, i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// ============================================================================
// 数字数组生成
// ============================================================================

/**
 * 生成指定分布的数字数组
 *
 * @param size 数组大小
 * @param distribution 分布类型
 * @param options 生成选项
 */
export function generateNumbers(
  size: number,
  distribution: Distribution,
  options: NumberOptions = {}
): number[] {
  const {
    min = 0,
    max = 10000,
    swapPercent = 5,
    uniqueCount = 10,
    sawtoothPeriod = 100,
  } = options;

  switch (distribution) {
    case 'random':
      return generateRandom(size, min, max);
    case 'sorted':
      return generateSorted(size, min, max);
    case 'reversed':
      return generateReversed(size, min, max);
    case 'nearlySorted':
      return generateNearlySorted(size, min, max, swapPercent);
    case 'fewUnique':
      return generateFewUnique(size, min, max, uniqueCount);
    case 'sawtooth':
      return generateSawtooth(size, min, max, sawtoothPeriod);
    default:
      throw new Error(`Unknown distribution: ${distribution}`);
  }
}

function generateRandom(size: number, min: number, max: number): number[] {
  return Array.from({ length: size }, () => randomInt(min, max));
}

function generateSorted(size: number, min: number, max: number): number[] {
  const step = (max - min) / size;
  return Array.from({ length: size }, (_, i) =>
    Math.floor(min + i * step + Math.random() * step)
  );
}

function generateReversed(size: number, min: number, max: number): number[] {
  return generateSorted(size, min, max).reverse();
}

function generateNearlySorted(
  size: number,
  min: number,
  max: number,
  swapPercent: number
): number[] {
  const arr = generateSorted(size, min, max);
  const swapCount = Math.floor(size * swapPercent / 100);

  for (let i = 0; i < swapCount; i++) {
    const idx1 = randomInt(0, size);
    const idx2 = randomInt(0, size);
    [arr[idx1], arr[idx2]] = [arr[idx2], arr[idx1]];
  }

  return arr;
}

function generateFewUnique(
  size: number,
  min: number,
  max: number,
  uniqueCount: number
): number[] {
  const uniqueValues = Array.from({ length: uniqueCount }, () =>
    randomInt(min, max)
  );

  return Array.from({ length: size }, () =>
    uniqueValues[randomInt(0, uniqueCount)]
  );
}

function generateSawtooth(
  size: number,
  min: number,
  max: number,
  period: number
): number[] {
  const step = (max - min) / period;
  return Array.from({ length: size }, (_, i) =>
    Math.floor(min + (i % period) * step)
  );
}

// ============================================================================
// 对象数组生成
// ============================================================================

const NAMES = [
  'Alice', 'Bob', 'Charlie', 'David', 'Eve',
  'Frank', 'Grace', 'Henry', 'Ivy', 'Jack',
  'Kate', 'Leo', 'Mia', 'Nick', 'Olivia',
];

const GRADES = ['A', 'B', 'C', 'D', 'F'];

/**
 * 生成表格行数据
 */
export function generateTableRows(size: number): TableRow[] {
  const now = Date.now();
  return Array.from({ length: size }, (_, i) => ({
    id: i + 1,
    name: NAMES[randomInt(0, NAMES.length)],
    score: randomInt(0, 101),
    grade: GRADES[randomInt(0, GRADES.length)],
    timestamp: now - randomInt(0, 86400000 * 365), // 过去一年内
  }));
}

/**
 * 生成搜索结果数据
 */
export function generateSearchResults(size: number): SearchResult[] {
  const now = Date.now();
  return Array.from({ length: size }, (_, i) => ({
    id: `result-${i + 1}`,
    title: `Article ${randomString(8)}`,
    relevance: Math.random() * 100,
    publishTime: now - randomInt(0, 86400000 * 30), // 过去 30 天内
    viewCount: randomInt(0, 100000),
  }));
}

/**
 * 生成带原始索引的对象（用于稳定性测试）
 */
export function generateIndexedItems<T>(
  items: T[]
): Array<{ value: T; originalIndex: number }> {
  return items.map((value, originalIndex) => ({ value, originalIndex }));
}

// ============================================================================
// 边界数据生成
// ============================================================================

/** 生成边界测试用例集 */
export function generateBoundaryTestCases(): Array<{
  name: string;
  data: number[];
}> {
  return [
    { name: '空数组', data: [] },
    { name: '单元素', data: [1] },
    { name: '两元素-已序', data: [1, 2] },
    { name: '两元素-逆序', data: [2, 1] },
    { name: '全相同', data: [5, 5, 5, 5, 5] },
    { name: '已排序', data: [1, 2, 3, 4, 5] },
    { name: '完全逆序', data: [5, 4, 3, 2, 1] },
    { name: '含负数', data: [-3, 1, -1, 2, 0] },
    { name: '含浮点', data: [1.5, 1.1, 1.9, 1.3] },
    { name: '含重复', data: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3] },
  ];
}

// ============================================================================
// 导出
// ============================================================================

export default {
  generateNumbers,
  generateTableRows,
  generateSearchResults,
  generateIndexedItems,
  generateBoundaryTestCases,
};

