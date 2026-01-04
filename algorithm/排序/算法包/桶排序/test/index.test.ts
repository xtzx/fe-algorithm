/**
 * 桶排序测试
 */

import { describe, it, expect } from 'vitest';
import {
  bucketSort,
  bucketSortGeneric,
  bucketSortStable,
  createRangeBucketSort,
  meta,
} from '../src/index';
import {
  verifySorted,
  verifyPermutation,
  verifyStable,
} from '../../../公共库/src/正确性校验';
import { numberAsc } from '../../../公共库/src/比较器';

describe('桶排序', () => {
  // ========================================================================
  // 基础浮点数排序测试
  // ========================================================================

  describe('基础浮点数排序', () => {
    it('应正确排序 [0, 1) 范围的浮点数', () => {
      const arr = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12];
      const sorted = bucketSort(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
    });

    it('应处理接近边界的值', () => {
      const arr = [0.001, 0.999, 0.5, 0.0, 0.99];
      const sorted = bucketSort(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
    });

    it('应处理重复值', () => {
      const arr = [0.5, 0.3, 0.5, 0.3, 0.5];
      const sorted = bucketSort(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(sorted.filter(x => x === 0.5).length).toBe(3);
    });
  });

  // ========================================================================
  // 边界条件测试
  // ========================================================================

  describe('边界条件', () => {
    it('应处理空数组', () => {
      expect(bucketSort([])).toEqual([]);
    });

    it('应处理单元素数组', () => {
      expect(bucketSort([0.5])).toEqual([0.5]);
    });

    it('应处理两元素数组', () => {
      expect(bucketSort([0.8, 0.2])).toEqual([0.2, 0.8]);
    });

    it('应拒绝超出范围的值', () => {
      expect(() => bucketSort([0.5, 1.0])).toThrow('不在 [0, 1) 范围内');
      expect(() => bucketSort([0.5, -0.1])).toThrow('不在 [0, 1) 范围内');
    });
  });

  // ========================================================================
  // 通用桶排序测试
  // ========================================================================

  describe('通用桶排序', () => {
    interface Product {
      name: string;
      price: number;
    }

    const products: Product[] = [
      { name: 'A', price: 150 },
      { name: 'B', price: 50 },
      { name: 'C', price: 250 },
      { name: 'D', price: 80 },
    ];

    it('应按自定义字段排序', () => {
      const sorted = bucketSortGeneric(
        products,
        3, // 3 个桶：0-99, 100-199, 200-299
        p => Math.min(2, Math.floor(p.price / 100)),
        (a, b) => a.price - b.price
      );

      expect(sorted.map(p => p.price)).toEqual([50, 80, 150, 250]);
    });

    it('应拒绝无效桶索引', () => {
      expect(() =>
        bucketSortGeneric(
          products,
          3,
          () => 5, // 超出范围
          (a, b) => a.price - b.price
        )
      ).toThrow('超出范围');
    });

    it('应拒绝无效桶数量', () => {
      expect(() =>
        bucketSortGeneric(products, 0, () => 0, (a, b) => a.price - b.price)
      ).toThrow('桶数量必须为正数');
    });
  });

  // ========================================================================
  // 稳定版本测试
  // ========================================================================

  describe('稳定版本', () => {
    interface Item {
      value: number;
      id: string;
    }

    it('应保持相同值元素的顺序', () => {
      const items: Item[] = [
        { value: 50, id: 'a' },
        { value: 50, id: 'b' },
        { value: 20, id: 'c' },
        { value: 50, id: 'd' },
      ];

      const sorted = bucketSortStable(
        items,
        2,
        i => (i.value < 50 ? 0 : 1),
        (a, b) => a.value - b.value
      );

      // value=50 的元素应保持 a, b, d 顺序
      const value50 = sorted.filter(i => i.value === 50);
      expect(value50.map(i => i.id)).toEqual(['a', 'b', 'd']);
    });

    it('应通过稳定性验证', () => {
      const arr = [0.55, 0.53, 0.57, 0.51, 0.59];
      const sorted = bucketSortStable(
        arr,
        10,
        n => Math.min(9, Math.floor(n * 10)),
        numberAsc
      );

      // 所有元素都在同一个桶（桶5），应保持原顺序
      expect(verifyStable(arr, sorted, numberAsc).passed).toBe(true);
    });
  });

  // ========================================================================
  // 工厂函数测试
  // ========================================================================

  describe('工厂函数', () => {
    interface Order {
      id: number;
      amount: number;
    }

    it('createRangeBucketSort 应正确工作', () => {
      const sortByAmount = createRangeBucketSort<Order>(
        0,
        1000,
        10,
        o => o.amount,
        (a, b) => a.amount - b.amount
      );

      const orders: Order[] = [
        { id: 1, amount: 500 },
        { id: 2, amount: 100 },
        { id: 3, amount: 800 },
      ];

      const sorted = sortByAmount(orders);
      expect(sorted.map(o => o.amount)).toEqual([100, 500, 800]);
    });

    it('应拒绝超出范围的值', () => {
      const sortByAmount = createRangeBucketSort<Order>(
        0,
        1000,
        10,
        o => o.amount,
        (a, b) => a.amount - b.amount
      );

      const orders: Order[] = [{ id: 1, amount: 1500 }];
      expect(() => sortByAmount(orders)).toThrow('超出范围');
    });
  });

  // ========================================================================
  // 分布敏感性测试
  // ========================================================================

  describe('分布敏感性', () => {
    it('均匀分布时应高效', () => {
      // 生成均匀分布的数据
      const arr = Array.from({ length: 1000 }, () => Math.random());

      const start = performance.now();
      const sorted = bucketSort(arr);
      const duration = performance.now() - start;

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(duration).toBeLessThan(100);
    });

    it('极端分布时仍应正确（但可能较慢）', () => {
      // 所有元素都在同一个桶
      const arr = Array.from({ length: 100 }, () => 0.5 + Math.random() * 0.01);
      const sorted = bucketSort(arr);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
    });
  });

  // ========================================================================
  // 元数据测试
  // ========================================================================

  describe('元数据', () => {
    it('应有正确的元数据', () => {
      expect(meta.name).toBe('桶排序');
      expect(meta.stable).toBe(false);
      expect(meta.stableVariant).toBe('bucketSortStable');
      expect(meta.inPlace).toBe(false);
    });
  });
});

