/**
 * 计数排序测试
 */

import { describe, it, expect } from 'vitest';
import {
  countingSort,
  countingSortAuto,
  countingSortBy,
  countingSortByAuto,
  meta,
} from '../src/index';
import {
  verifySorted,
  verifyPermutation,
  verifyStable,
  verifySort,
} from '../../../公共库/src/正确性校验';
import { numberAsc } from '../../../公共库/src/比较器';

describe('计数排序', () => {
  // ========================================================================
  // 基础功能测试
  // ========================================================================

  describe('基础功能', () => {
    it('应正确排序随机数组', () => {
      const arr = [4, 2, 8, 5, 2, 3, 9, 1];
      const sorted = countingSort(arr, 0, 10);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
    });

    it('应正确排序已排序数组', () => {
      const arr = [1, 2, 3, 4, 5];
      const sorted = countingSort(arr, 0, 10);

      expect(sorted).toEqual([1, 2, 3, 4, 5]);
    });

    it('应正确排序逆序数组', () => {
      const arr = [5, 4, 3, 2, 1];
      const sorted = countingSort(arr, 0, 10);

      expect(sorted).toEqual([1, 2, 3, 4, 5]);
    });

    it('应正确排序含重复元素的数组', () => {
      const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
      const sorted = countingSort(arr, 0, 10);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
    });
  });

  // ========================================================================
  // 边界条件测试
  // ========================================================================

  describe('边界条件', () => {
    it('应处理空数组', () => {
      expect(countingSort([], 0, 10)).toEqual([]);
    });

    it('应处理单元素数组', () => {
      expect(countingSort([5], 0, 10)).toEqual([5]);
    });

    it('应处理两元素数组', () => {
      expect(countingSort([2, 1], 0, 10)).toEqual([1, 2]);
      expect(countingSort([1, 2], 0, 10)).toEqual([1, 2]);
    });

    it('应处理全相同元素', () => {
      const arr = [5, 5, 5, 5, 5];
      const sorted = countingSort(arr, 0, 10);

      expect(sorted).toEqual([5, 5, 5, 5, 5]);
    });

    it('应处理负数范围', () => {
      const arr = [-3, -1, -5, -2];
      const sorted = countingSort(arr, -10, 0);

      expect(sorted).toEqual([-5, -3, -2, -1]);
    });

    it('应处理包含 0 的数组', () => {
      const arr = [3, 0, 2, 0, 1];
      const sorted = countingSort(arr, 0, 10);

      expect(sorted).toEqual([0, 0, 1, 2, 3]);
    });
  });

  // ========================================================================
  // 稳定性测试
  // ========================================================================

  describe('稳定性', () => {
    it('应保持相同元素的原始顺序', () => {
      interface Item {
        value: number;
        id: string;
      }

      const items: Item[] = [
        { value: 3, id: 'a' },
        { value: 1, id: 'b' },
        { value: 3, id: 'c' }, // 与 a 相同
        { value: 2, id: 'd' },
        { value: 1, id: 'e' }, // 与 b 相同
      ];

      const sorted = countingSortBy(items, i => i.value, 0, 5);

      // value=1: b 应在 e 前面
      // value=3: a 应在 c 前面
      const value1 = sorted.filter(i => i.value === 1);
      const value3 = sorted.filter(i => i.value === 3);

      expect(value1[0].id).toBe('b');
      expect(value1[1].id).toBe('e');
      expect(value3[0].id).toBe('a');
      expect(value3[1].id).toBe('c');
    });

    it('应通过稳定性验证', () => {
      // 使用带原始索引的数据测试
      const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
      const sorted = countingSort(arr, 0, 10);

      // 数值级稳定性验证
      expect(verifyStable(arr, sorted, numberAsc).passed).toBe(true);
    });
  });

  // ========================================================================
  // 自动范围检测测试
  // ========================================================================

  describe('自动范围检测', () => {
    it('countingSortAuto 应自动检测范围', () => {
      const arr = [4, 2, 8, 5, 2, 3, 9, 1];
      const sorted = countingSortAuto(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
    });

    it('countingSortAuto 应处理负数', () => {
      const arr = [-5, 3, -2, 8, -1];
      const sorted = countingSortAuto(arr);

      expect(sorted).toEqual([-5, -2, -1, 3, 8]);
    });

    it('countingSortByAuto 应自动检测范围', () => {
      interface Item { key: number; name: string }
      const items: Item[] = [
        { key: 5, name: 'a' },
        { key: 2, name: 'b' },
        { key: 8, name: 'c' },
      ];

      const sorted = countingSortByAuto(items, i => i.key);
      expect(sorted.map(i => i.key)).toEqual([2, 5, 8]);
    });
  });

  // ========================================================================
  // 对象排序测试
  // ========================================================================

  describe('对象排序', () => {
    interface Student {
      name: string;
      score: number;
    }

    const students: Student[] = [
      { name: 'Alice', score: 85 },
      { name: 'Bob', score: 92 },
      { name: 'Charlie', score: 85 },
      { name: 'David', score: 78 },
    ];

    it('应按字段正确排序', () => {
      const sorted = countingSortBy(students, s => s.score, 0, 100);

      expect(sorted[0].name).toBe('David'); // 78
      expect(sorted[3].name).toBe('Bob');   // 92
    });

    it('应保持相同分数学生的顺序（稳定性）', () => {
      const sorted = countingSortBy(students, s => s.score, 0, 100);

      // Alice(85) 应在 Charlie(85) 前面
      const score85 = sorted.filter(s => s.score === 85);
      expect(score85[0].name).toBe('Alice');
      expect(score85[1].name).toBe('Charlie');
    });
  });

  // ========================================================================
  // 错误处理测试
  // ========================================================================

  describe('错误处理', () => {
    it('应拒绝无效范围（min > max）', () => {
      expect(() => countingSort([1, 2, 3], 10, 5)).toThrow('无效范围');
    });

    it('应拒绝超出范围的值', () => {
      expect(() => countingSort([1, 2, 15], 0, 10)).toThrow('超出范围');
    });

    it('应拒绝浮点数', () => {
      expect(() => countingSort([1, 2.5, 3], 0, 10)).toThrow('整数');
    });

    it('countingSortBy 应拒绝非整数 key', () => {
      interface Item { key: number }
      const items: Item[] = [{ key: 1.5 }];

      expect(() =>
        countingSortBy(items, i => i.key, 0, 10)
      ).toThrow('整数');
    });
  });

  // ========================================================================
  // 性能特征测试
  // ========================================================================

  describe('性能特征', () => {
    it('应高效处理大量重复数据', () => {
      // 少量唯一值，大量重复
      const arr = Array.from({ length: 10000 }, () =>
        Math.floor(Math.random() * 10)
      );

      const start = performance.now();
      const sorted = countingSort(arr, 0, 10);
      const duration = performance.now() - start;

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      // 应该非常快（毫秒级）
      expect(duration).toBeLessThan(100);
    });

    it('应高效处理大范围稀疏数据', () => {
      // 这是计数排序的弱点场景，但仍应正确工作
      const arr = [1, 500, 999];
      const sorted = countingSort(arr, 0, 1000);

      expect(sorted).toEqual([1, 500, 999]);
    });
  });

  // ========================================================================
  // 元数据测试
  // ========================================================================

  describe('元数据', () => {
    it('应有正确的元数据', () => {
      expect(meta.name).toBe('计数排序');
      expect(meta.stable).toBe(true);
      expect(meta.inPlace).toBe(false);
      expect(meta.timeComplexity.average).toBe('O(n + k)');
    });
  });
});

