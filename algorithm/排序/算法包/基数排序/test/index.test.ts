/**
 * 基数排序测试
 */

import { describe, it, expect } from 'vitest';
import {
  radixSort,
  radixSortBy,
  radixSortWithNegative,
  radixSortByWithNegative,
  meta,
} from '../src/index';
import {
  verifySorted,
  verifyPermutation,
  verifyStable,
} from '../../../公共库/src/正确性校验';
import { numberAsc } from '../../../公共库/src/比较器';

describe('基数排序', () => {
  // ========================================================================
  // 基础功能测试
  // ========================================================================

  describe('基础功能', () => {
    it('应正确排序随机数组', () => {
      const arr = [170, 45, 75, 90, 802, 24, 2, 66];
      const sorted = radixSort(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
      expect(sorted).toEqual([2, 24, 45, 66, 75, 90, 170, 802]);
    });

    it('应正确排序已排序数组', () => {
      const arr = [1, 2, 3, 4, 5];
      const sorted = radixSort(arr);

      expect(sorted).toEqual([1, 2, 3, 4, 5]);
    });

    it('应正确排序逆序数组', () => {
      const arr = [999, 888, 777, 666, 555];
      const sorted = radixSort(arr);

      expect(sorted).toEqual([555, 666, 777, 888, 999]);
    });

    it('应正确排序含重复元素的数组', () => {
      const arr = [50, 30, 50, 20, 30, 50];
      const sorted = radixSort(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(sorted.filter(x => x === 50).length).toBe(3);
    });
  });

  // ========================================================================
  // 边界条件测试
  // ========================================================================

  describe('边界条件', () => {
    it('应处理空数组', () => {
      expect(radixSort([])).toEqual([]);
    });

    it('应处理单元素数组', () => {
      expect(radixSort([42])).toEqual([42]);
    });

    it('应处理两元素数组', () => {
      expect(radixSort([100, 10])).toEqual([10, 100]);
    });

    it('应处理全相同元素', () => {
      const arr = [123, 123, 123];
      expect(radixSort(arr)).toEqual([123, 123, 123]);
    });

    it('应处理包含 0 的数组', () => {
      const arr = [100, 0, 50, 0, 25];
      const sorted = radixSort(arr);

      expect(sorted).toEqual([0, 0, 25, 50, 100]);
    });

    it('应处理大数值', () => {
      const arr = [999999999, 1, 123456789];
      const sorted = radixSort(arr);

      expect(sorted).toEqual([1, 123456789, 999999999]);
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
        { value: 50, id: 'a' },
        { value: 20, id: 'b' },
        { value: 50, id: 'c' },
        { value: 30, id: 'd' },
        { value: 50, id: 'e' },
      ];

      const sorted = radixSortBy(items, i => i.value);

      // value=50 的元素应保持 a, c, e 顺序
      const value50 = sorted.filter(i => i.value === 50);
      expect(value50.map(i => i.id)).toEqual(['a', 'c', 'e']);
    });

    it('应通过稳定性验证', () => {
      const arr = [30, 10, 40, 10, 50, 30, 20];
      const sorted = radixSort(arr);

      expect(verifyStable(arr, sorted, numberAsc).passed).toBe(true);
    });
  });

  // ========================================================================
  // 负数支持测试
  // ========================================================================

  describe('负数支持', () => {
    it('radixSortWithNegative 应正确排序负数', () => {
      const arr = [-5, 3, -2, 8, -1, 0, 5, -3];
      const sorted = radixSortWithNegative(arr);

      expect(sorted).toEqual([-5, -3, -2, -1, 0, 3, 5, 8]);
    });

    it('应处理全是负数的数组', () => {
      const arr = [-10, -5, -20, -1];
      const sorted = radixSortWithNegative(arr);

      expect(sorted).toEqual([-20, -10, -5, -1]);
    });

    it('应处理只有一个负数', () => {
      const arr = [5, -1, 3];
      const sorted = radixSortWithNegative(arr);

      expect(sorted).toEqual([-1, 3, 5]);
    });

    it('radixSortByWithNegative 应正确排序对象', () => {
      interface Item { value: number; name: string }
      const items: Item[] = [
        { value: -5, name: 'a' },
        { value: 3, name: 'b' },
        { value: -2, name: 'c' },
      ];

      const sorted = radixSortByWithNegative(items, i => i.value);
      expect(sorted.map(i => i.value)).toEqual([-5, -2, 3]);
    });
  });

  // ========================================================================
  // 对象排序测试
  // ========================================================================

  describe('对象排序', () => {
    interface Order {
      id: number;
      amount: number;
    }

    const orders: Order[] = [
      { id: 1003, amount: 500 },
      { id: 1001, amount: 300 },
      { id: 1005, amount: 800 },
      { id: 1002, amount: 300 },
    ];

    it('应按整数字段正确排序', () => {
      const sorted = radixSortBy(orders, o => o.id);

      expect(sorted.map(o => o.id)).toEqual([1001, 1002, 1003, 1005]);
    });

    it('应保持相同 key 的原始顺序（稳定性）', () => {
      const sorted = radixSortBy(orders, o => o.amount);

      // amount=300: id=1001 应在 id=1002 前面
      const amount300 = sorted.filter(o => o.amount === 300);
      expect(amount300[0].id).toBe(1001);
      expect(amount300[1].id).toBe(1002);
    });
  });

  // ========================================================================
  // 错误处理测试
  // ========================================================================

  describe('错误处理', () => {
    it('应拒绝负数（radixSort）', () => {
      expect(() => radixSort([1, -2, 3])).toThrow('不支持负数');
    });

    it('应拒绝浮点数', () => {
      expect(() => radixSort([1, 2.5, 3])).toThrow('整数');
    });

    it('radixSortBy 应拒绝负数 key', () => {
      interface Item { key: number }
      const items: Item[] = [{ key: -1 }];

      expect(() => radixSortBy(items, i => i.key)).toThrow('不能为负数');
    });

    it('radixSortBy 应拒绝非整数 key', () => {
      interface Item { key: number }
      const items: Item[] = [{ key: 1.5 }];

      expect(() => radixSortBy(items, i => i.key)).toThrow('整数');
    });
  });

  // ========================================================================
  // 不同基数测试
  // ========================================================================

  describe('不同基数', () => {
    it('基数 2 应正确工作', () => {
      const arr = [7, 3, 5, 1, 6, 2, 4];
      const sorted = radixSort(arr, 2);

      expect(sorted).toEqual([1, 2, 3, 4, 5, 6, 7]);
    });

    it('基数 16 应正确工作', () => {
      const arr = [255, 16, 128, 64, 1];
      const sorted = radixSort(arr, 16);

      expect(sorted).toEqual([1, 16, 64, 128, 255]);
    });

    it('基数 256 应正确工作', () => {
      const arr = [1000000, 1, 999999];
      const sorted = radixSort(arr, 256);

      expect(sorted).toEqual([1, 999999, 1000000]);
    });
  });

  // ========================================================================
  // 性能特征测试
  // ========================================================================

  describe('性能特征', () => {
    it('应高效处理大量数据', () => {
      const arr = Array.from({ length: 10000 }, () =>
        Math.floor(Math.random() * 1000000)
      );

      const start = performance.now();
      const sorted = radixSort(arr);
      const duration = performance.now() - start;

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(duration).toBeLessThan(100);
    });

    it('应高效处理固定位数数据', () => {
      // 模拟手机号
      const phones = Array.from({ length: 10000 }, () =>
        13000000000 + Math.floor(Math.random() * 1000000000)
      );

      const start = performance.now();
      const sorted = radixSort(phones);
      const duration = performance.now() - start;

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(duration).toBeLessThan(100);
    });
  });

  // ========================================================================
  // 元数据测试
  // ========================================================================

  describe('元数据', () => {
    it('应有正确的元数据', () => {
      expect(meta.name).toBe('基数排序');
      expect(meta.stable).toBe(true);
      expect(meta.inPlace).toBe(false);
      expect(meta.timeComplexity.average).toBe('O(d·(n+k))');
    });
  });
});

