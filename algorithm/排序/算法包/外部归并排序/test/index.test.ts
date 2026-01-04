/**
 * 外部归并排序测试
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  externalMergeSort,
  kWayMerge,
  twoWayMerge,
  iterativeMerge,
  createExternalSorter,
  StreamSorter,
  meta,
} from '../src/index';
import {
  verifySorted,
  verifyPermutation,
} from '../../../公共库/src/正确性校验';
import { numberAsc, byField } from '../../../公共库/src/比较器';

describe('外部归并排序', () => {
  // ========================================================================
  // 基础功能测试
  // ========================================================================

  describe('externalMergeSort', () => {
    it('应正确排序随机数组', () => {
      const arr = [5, 2, 8, 1, 9, 3, 7, 4, 6];
      const sorted = externalMergeSort(arr, 3, numberAsc);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
    });

    it('应正确处理能一次装入内存的数组', () => {
      const arr = [3, 1, 2];
      const sorted = externalMergeSort(arr, 10, numberAsc);

      expect(sorted).toEqual([1, 2, 3]);
    });

    it('应正确处理需要多块的数组', () => {
      const arr = Array.from({ length: 100 }, () =>
        Math.floor(Math.random() * 1000)
      );
      const sorted = externalMergeSort(arr, 10, numberAsc);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(verifyPermutation(arr, sorted).passed).toBe(true);
    });

    it('应正确排序对象数组', () => {
      interface Item { id: number; value: number }
      const items: Item[] = [
        { id: 1, value: 50 },
        { id: 2, value: 30 },
        { id: 3, value: 70 },
        { id: 4, value: 10 },
        { id: 5, value: 40 },
      ];

      const sorted = externalMergeSort(
        items,
        2,
        byField('value', numberAsc)
      );

      expect(sorted.map(i => i.value)).toEqual([10, 30, 40, 50, 70]);
    });
  });

  // ========================================================================
  // 边界条件测试
  // ========================================================================

  describe('边界条件', () => {
    it('应处理空数组', () => {
      expect(externalMergeSort([], 10, numberAsc)).toEqual([]);
    });

    it('应处理单元素数组', () => {
      expect(externalMergeSort([42], 10, numberAsc)).toEqual([42]);
    });

    it('应处理块大小为 1 的情况', () => {
      const arr = [3, 1, 2];
      const sorted = externalMergeSort(arr, 1, numberAsc);

      expect(sorted).toEqual([1, 2, 3]);
    });

    it('应处理元素数量刚好是块大小倍数', () => {
      const arr = [6, 5, 4, 3, 2, 1];
      const sorted = externalMergeSort(arr, 2, numberAsc);

      expect(sorted).toEqual([1, 2, 3, 4, 5, 6]);
    });
  });

  // ========================================================================
  // K 路归并测试
  // ========================================================================

  describe('kWayMerge', () => {
    it('应正确合并多个有序数组', () => {
      const chunks = [[1, 4, 7], [2, 5, 8], [3, 6, 9]];
      const merged = kWayMerge(chunks, numberAsc);

      expect(merged).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it('应处理空输入', () => {
      expect(kWayMerge([], numberAsc)).toEqual([]);
    });

    it('应处理单个数组', () => {
      expect(kWayMerge([[1, 2, 3]], numberAsc)).toEqual([1, 2, 3]);
    });

    it('应处理包含空数组的输入', () => {
      const chunks = [[1, 3], [], [2, 4]];
      const merged = kWayMerge(chunks, numberAsc);

      expect(merged).toEqual([1, 2, 3, 4]);
    });

    it('应处理不等长的数组', () => {
      const chunks = [[1], [2, 3, 4], [5, 6]];
      const merged = kWayMerge(chunks, numberAsc);

      expect(merged).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('应处理重复元素', () => {
      const chunks = [[1, 2, 2], [2, 3], [1, 4]];
      const merged = kWayMerge(chunks, numberAsc);

      expect(merged).toEqual([1, 1, 2, 2, 2, 3, 4]);
    });
  });

  // ========================================================================
  // 两路归并测试
  // ========================================================================

  describe('twoWayMerge', () => {
    it('应正确合并两个有序数组', () => {
      const merged = twoWayMerge([1, 3, 5], [2, 4, 6], numberAsc);

      expect(merged).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('应处理一个空数组', () => {
      expect(twoWayMerge([1, 2, 3], [], numberAsc)).toEqual([1, 2, 3]);
      expect(twoWayMerge([], [1, 2, 3], numberAsc)).toEqual([1, 2, 3]);
    });

    it('应处理两个空数组', () => {
      expect(twoWayMerge([], [], numberAsc)).toEqual([]);
    });
  });

  // ========================================================================
  // 迭代归并测试
  // ========================================================================

  describe('iterativeMerge', () => {
    it('应正确迭代归并多个数组', () => {
      const chunks = [[1, 5], [2, 6], [3, 7], [4, 8]];
      const merged = iterativeMerge(chunks, numberAsc);

      expect(merged).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('应处理奇数个数组', () => {
      const chunks = [[1], [2], [3]];
      const merged = iterativeMerge(chunks, numberAsc);

      expect(merged).toEqual([1, 2, 3]);
    });

    it('应处理空输入', () => {
      expect(iterativeMerge([], numberAsc)).toEqual([]);
    });
  });

  // ========================================================================
  // 可配置排序器测试
  // ========================================================================

  describe('createExternalSorter', () => {
    it('应使用自定义排序和归并函数', () => {
      let sortCalled = 0;
      let mergeCalled = 0;

      const sorter = createExternalSorter<number>({
        chunkSize: 3,
        sortChunk: (chunk, cmp) => {
          sortCalled++;
          return chunk.sort(cmp);
        },
        mergeChunks: (chunks, cmp) => {
          mergeCalled++;
          return kWayMerge(chunks, cmp);
        },
      });

      const arr = [5, 2, 8, 1, 9, 3, 7, 4, 6];
      const sorted = sorter(arr, numberAsc);

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(sortCalled).toBe(3); // 9 / 3 = 3 块
      expect(mergeCalled).toBe(1);
    });
  });

  // ========================================================================
  // 流式排序器测试
  // ========================================================================

  describe('StreamSorter', () => {
    let sorter: StreamSorter<number>;

    beforeEach(() => {
      sorter = new StreamSorter<number>(5, numberAsc);
    });

    it('应正确处理流式数据', () => {
      sorter.add(5);
      sorter.add(2);
      sorter.add(8);
      sorter.add(1);
      sorter.add(9);

      const result = sorter.getResult();
      expect(result).toEqual([1, 2, 5, 8, 9]);
    });

    it('应正确处理批量添加', () => {
      sorter.addBatch([5, 2, 8, 1, 9, 3, 7, 4, 6]);

      const result = sorter.getResult();
      expect(verifySorted(result, numberAsc).passed).toBe(true);
    });

    it('应在缓冲区满时自动刷新', () => {
      // chunkSize = 5，添加 6 个元素后应该有 2 个块
      sorter.addBatch([1, 2, 3, 4, 5, 6]);

      expect(sorter.getChunkCount()).toBe(2); // 1 个完整块 + 1 个缓冲区
    });

    it('reset 应清空状态', () => {
      sorter.addBatch([1, 2, 3]);
      sorter.reset();

      expect(sorter.getResult()).toEqual([]);
    });

    it('多次 getResult 应返回相同结果', () => {
      sorter.addBatch([3, 1, 2]);

      const result1 = sorter.getResult();
      const result2 = sorter.getResult();

      expect(result1).toEqual(result2);
    });
  });

  // ========================================================================
  // 稳定性测试
  // ========================================================================

  describe('稳定性', () => {
    it('kWayMerge 应保持稳定性', () => {
      interface Item { value: number; id: string }

      const chunks: Item[][] = [
        [{ value: 1, id: 'a' }, { value: 3, id: 'c' }],
        [{ value: 1, id: 'b' }, { value: 2, id: 'd' }],
      ];

      const merged = kWayMerge(
        chunks,
        (a, b) => a.value - b.value
      );

      // value=1 的元素应保持原块顺序：a 在 b 前面
      const value1 = merged.filter(i => i.value === 1);
      expect(value1[0].id).toBe('a');
      expect(value1[1].id).toBe('b');
    });
  });

  // ========================================================================
  // 性能测试
  // ========================================================================

  describe('性能', () => {
    it('应高效处理大量数据', () => {
      const arr = Array.from({ length: 10000 }, () =>
        Math.floor(Math.random() * 100000)
      );

      const start = performance.now();
      const sorted = externalMergeSort(arr, 1000, numberAsc);
      const duration = performance.now() - start;

      expect(verifySorted(sorted, numberAsc).passed).toBe(true);
      expect(duration).toBeLessThan(500);
    });

    it('K 路归并应高效处理多个块', () => {
      const chunks = Array.from({ length: 100 }, () =>
        Array.from({ length: 100 }, () =>
          Math.floor(Math.random() * 10000)
        ).sort((a, b) => a - b)
      );

      const start = performance.now();
      const merged = kWayMerge(chunks, numberAsc);
      const duration = performance.now() - start;

      expect(verifySorted(merged, numberAsc).passed).toBe(true);
      expect(duration).toBeLessThan(200);
    });
  });

  // ========================================================================
  // 元数据测试
  // ========================================================================

  describe('元数据', () => {
    it('应有正确的元数据', () => {
      expect(meta.name).toBe('外部归并排序');
      expect(meta.stable).toBe(true);
      expect(meta.inPlace).toBe(false);
      expect(meta.timeComplexity.average).toBe('O(n log n)');
    });
  });
});

