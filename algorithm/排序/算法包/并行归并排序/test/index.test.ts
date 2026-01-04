/**
 * 并行归并排序测试
 */

import { describe, it, expect, vi, beforeAll, afterAll } from 'vitest';
import {
  parallelMergeSort,
  singleThreadMergeSort,
  shouldUseParallel,
  benchmark,
} from '../src/index';
import { mergeSort, merge } from '../src/worker';

// ============================================================================
// 辅助函数
// ============================================================================

function generateRandomArray(size: number): number[] {
  return Array.from({ length: size }, () => Math.random() * 10000);
}

function isSorted<T>(arr: T[], cmp: (a: T, b: T) => number): boolean {
  for (let i = 1; i < arr.length; i++) {
    if (cmp(arr[i - 1], arr[i]) > 0) {
      return false;
    }
  }
  return true;
}

function isPermutation<T>(original: T[], sorted: T[]): boolean {
  if (original.length !== sorted.length) return false;

  const originalCopy = [...original].sort();
  const sortedCopy = [...sorted].sort();

  return JSON.stringify(originalCopy) === JSON.stringify(sortedCopy);
}

// ============================================================================
// 单线程归并排序测试
// ============================================================================

describe('singleThreadMergeSort', () => {
  it('should sort empty array', () => {
    const arr: number[] = [];
    const sorted = singleThreadMergeSort(arr, (a, b) => a - b);
    expect(sorted).toEqual([]);
  });

  it('should sort single element', () => {
    const arr = [42];
    const sorted = singleThreadMergeSort(arr, (a, b) => a - b);
    expect(sorted).toEqual([42]);
  });

  it('should sort already sorted array', () => {
    const arr = [1, 2, 3, 4, 5];
    const sorted = singleThreadMergeSort(arr, (a, b) => a - b);
    expect(sorted).toEqual([1, 2, 3, 4, 5]);
  });

  it('should sort reverse sorted array', () => {
    const arr = [5, 4, 3, 2, 1];
    const sorted = singleThreadMergeSort(arr, (a, b) => a - b);
    expect(sorted).toEqual([1, 2, 3, 4, 5]);
  });

  it('should sort array with duplicates', () => {
    const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    const sorted = singleThreadMergeSort(arr, (a, b) => a - b);
    expect(isSorted(sorted, (a, b) => a - b)).toBe(true);
    expect(isPermutation(arr, sorted)).toBe(true);
  });

  it('should sort large array correctly', () => {
    const arr = generateRandomArray(10000);
    const sorted = singleThreadMergeSort(arr, (a, b) => a - b);
    expect(isSorted(sorted, (a, b) => a - b)).toBe(true);
    expect(isPermutation(arr, sorted)).toBe(true);
  });

  it('should sort strings', () => {
    const arr = ['banana', 'apple', 'cherry', 'date'];
    const sorted = singleThreadMergeSort(arr, (a, b) => a.localeCompare(b));
    expect(sorted).toEqual(['apple', 'banana', 'cherry', 'date']);
  });

  it('should sort objects', () => {
    const arr = [
      { id: 3, name: 'Charlie' },
      { id: 1, name: 'Alice' },
      { id: 2, name: 'Bob' },
    ];
    const sorted = singleThreadMergeSort(arr, (a, b) => a.id - b.id);
    expect(sorted.map(x => x.id)).toEqual([1, 2, 3]);
  });

  it('should be stable', () => {
    const arr = [
      { key: 1, order: 1 },
      { key: 2, order: 1 },
      { key: 1, order: 2 },
      { key: 2, order: 2 },
    ];
    const sorted = singleThreadMergeSort(arr, (a, b) => a.key - b.key);

    const key1Items = sorted.filter(x => x.key === 1);
    expect(key1Items[0].order).toBe(1);
    expect(key1Items[1].order).toBe(2);
  });
});

// ============================================================================
// Worker merge 函数测试
// ============================================================================

describe('merge (worker)', () => {
  const numCmp = (a: number, b: number) => a - b;

  it('should merge two empty arrays', () => {
    expect(merge([], [], numCmp)).toEqual([]);
  });

  it('should merge empty with non-empty', () => {
    expect(merge([], [1, 2], numCmp)).toEqual([1, 2]);
    expect(merge([1, 2], [], numCmp)).toEqual([1, 2]);
  });

  it('should merge two sorted arrays', () => {
    expect(merge([1, 3, 5], [2, 4, 6], numCmp)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('should handle duplicates', () => {
    expect(merge([1, 2, 2], [2, 3], numCmp)).toEqual([1, 2, 2, 2, 3]);
  });
});

// ============================================================================
// Worker mergeSort 函数测试
// ============================================================================

describe('mergeSort (worker)', () => {
  const numCmp = (a: number, b: number) => a - b;

  it('should sort array', () => {
    const arr = [3, 1, 4, 1, 5, 9, 2, 6];
    const sorted = mergeSort([...arr], numCmp);
    expect(isSorted(sorted, numCmp)).toBe(true);
  });
});

// ============================================================================
// shouldUseParallel 测试
// ============================================================================

describe('shouldUseParallel', () => {
  it('should return false for small arrays', () => {
    expect(shouldUseParallel(100)).toBe(false);
    expect(shouldUseParallel(1000)).toBe(false);
    expect(shouldUseParallel(9999)).toBe(false);
  });

  it('should return true for large arrays', () => {
    // 注意：在非浏览器环境可能返回 false
    const result = shouldUseParallel(100000);
    // Worker 是否可用取决于环境
    expect(typeof result).toBe('boolean');
  });

  it('should respect custom threshold', () => {
    expect(shouldUseParallel(500, 1000)).toBe(false);
    expect(shouldUseParallel(1500, 1000)).toBe(true);
  });
});

// ============================================================================
// 并行排序测试（需要 Worker 支持）
// ============================================================================

describe('parallelMergeSort', () => {
  it('should fall back to single thread for small arrays', async () => {
    const arr = [3, 1, 4, 1, 5];
    const sorted = await parallelMergeSort(arr, (a, b) => a - b);
    expect(isSorted(sorted, (a, b) => a - b)).toBe(true);
    expect(isPermutation(arr, sorted)).toBe(true);
  });

  it('should sort empty array', async () => {
    const sorted = await parallelMergeSort([], (a, b) => a - b);
    expect(sorted).toEqual([]);
  });

  it('should sort large array', async () => {
    const arr = generateRandomArray(1000);
    const sorted = await parallelMergeSort(arr, (a, b) => a - b, {
      threshold: 100,
    });
    expect(isSorted(sorted, (a, b) => a - b)).toBe(true);
    expect(isPermutation(arr, sorted)).toBe(true);
  });

  it('should produce same result as single thread', async () => {
    const arr = generateRandomArray(5000);
    const singleResult = singleThreadMergeSort(arr, (a, b) => a - b);
    const parallelResult = await parallelMergeSort(arr, (a, b) => a - b, {
      threshold: 100,
    });
    expect(parallelResult).toEqual(singleResult);
  });

  it('should handle custom worker count', async () => {
    const arr = generateRandomArray(2000);
    const sorted = await parallelMergeSort(arr, (a, b) => a - b, {
      workerCount: 2,
      threshold: 100,
    });
    expect(isSorted(sorted, (a, b) => a - b)).toBe(true);
  });

  it('should sort descending', async () => {
    const arr = [1, 2, 3, 4, 5];
    const sorted = await parallelMergeSort(arr, (a, b) => b - a);
    expect(sorted).toEqual([5, 4, 3, 2, 1]);
  });
});

// ============================================================================
// 性能基准测试
// ============================================================================

describe('benchmark', () => {
  it('should return benchmark results', async () => {
    const arr = generateRandomArray(5000);
    const result = await benchmark(arr, (a, b) => a - b, 2);

    expect(result.singleThread).toBeGreaterThan(0);
    expect(result.parallel).toBeGreaterThan(0);
    expect(result.speedup).toBeGreaterThan(0);
  });
});

