/**
 * 表格排序测试
 */

import { describe, it, expect } from 'vitest';
import { SAMPLE_DATA, type TableRow, type SortColumn } from '../src/数据模型';
import {
  sortByMultipleColumns,
  sortByColumn,
  handleMultiColumnSort,
} from '../src/多列稳定排序';
import {
  createTableComparator,
  sortByComparator,
  TableComparatorBuilder,
} from '../src/比较器组合器';
import { verifyStable } from '../../../算法包/公共库/src/正确性校验';
import { numberAsc } from '../../../算法包/公共库/src/比较器';

describe('表格排序', () => {
  // ========================================================================
  // 多列稳定排序测试
  // ========================================================================

  describe('sortByMultipleColumns', () => {
    it('应按单列正确排序', () => {
      const sorted = sortByMultipleColumns(SAMPLE_DATA, [
        { field: 'score', order: 'asc', type: 'number' },
      ]);

      // 验证升序
      for (let i = 1; i < sorted.length; i++) {
        expect(sorted[i].score).toBeGreaterThanOrEqual(sorted[i - 1].score);
      }
    });

    it('应按多列正确排序', () => {
      const sorted = sortByMultipleColumns(SAMPLE_DATA, [
        { field: 'department', order: 'asc', type: 'string' },
        { field: 'score', order: 'desc', type: 'number' },
      ]);

      // 验证：按部门分组，每组内分数降序
      let prevDept = '';
      let prevScore = Infinity;

      for (const row of sorted) {
        if (row.department !== prevDept) {
          prevDept = row.department;
          prevScore = Infinity;
        }
        expect(row.score).toBeLessThanOrEqual(prevScore);
        prevScore = row.score;
      }
    });

    it('应保持稳定性', () => {
      // 只按 score 排序
      const sorted = sortByMultipleColumns(SAMPLE_DATA, [
        { field: 'score', order: 'asc', type: 'number' },
      ]);

      // 验证相同 score 的行保持原顺序
      const cmp = (a: TableRow, b: TableRow) => a.score - b.score;
      expect(verifyStable(SAMPLE_DATA, sorted, cmp).passed).toBe(true);
    });

    it('应处理空数组', () => {
      expect(sortByMultipleColumns([], [{ field: 'score', order: 'asc' }])).toEqual([]);
    });

    it('应处理无排序列', () => {
      const result = sortByMultipleColumns(SAMPLE_DATA, []);
      expect(result).toHaveLength(SAMPLE_DATA.length);
    });
  });

  // ========================================================================
  // 比较器组合测试
  // ========================================================================

  describe('sortByComparator', () => {
    it('应与 sortByMultipleColumns 产生相同结果', () => {
      const columns: SortColumn<TableRow>[] = [
        { field: 'department', order: 'asc', type: 'string' },
        { field: 'score', order: 'desc', type: 'number' },
      ];

      const resultA = sortByMultipleColumns(SAMPLE_DATA, columns);
      const resultB = sortByComparator(SAMPLE_DATA, columns);

      // 两种方式的排序结果应该相同
      expect(resultA.map(r => r.id)).toEqual(resultB.map(r => r.id));
    });

    it('应正确处理日期排序', () => {
      const sorted = sortByComparator(SAMPLE_DATA, [
        { field: 'joinDate', order: 'asc', type: 'date' },
      ]);

      // 验证日期升序
      for (let i = 1; i < sorted.length; i++) {
        const prevDate = new Date(sorted[i - 1].joinDate);
        const currDate = new Date(sorted[i].joinDate);
        expect(currDate.getTime()).toBeGreaterThanOrEqual(prevDate.getTime());
      }
    });

    it('应正确处理降序', () => {
      const sorted = sortByComparator(SAMPLE_DATA, [
        { field: 'salary', order: 'desc', type: 'number' },
      ]);

      for (let i = 1; i < sorted.length; i++) {
        expect(sorted[i].salary).toBeLessThanOrEqual(sorted[i - 1].salary);
      }
    });
  });

  // ========================================================================
  // 构建器测试
  // ========================================================================

  describe('TableComparatorBuilder', () => {
    it('应正确构建比较器', () => {
      const builder = new TableComparatorBuilder<TableRow>()
        .addColumn('department', 'asc', 'string')
        .addColumn('score', 'desc', 'number');

      const sorted = builder.sort(SAMPLE_DATA);
      const columns = builder.getColumns();

      expect(columns).toHaveLength(2);
      expect(columns[0].field).toBe('department');
      expect(sorted.length).toBe(SAMPLE_DATA.length);
    });

    it('toggleOrder 应切换排序方向', () => {
      const builder = new TableComparatorBuilder<TableRow>()
        .addColumn('score', 'asc', 'number');

      expect(builder.getColumns()[0].order).toBe('asc');

      builder.toggleOrder('score');
      expect(builder.getColumns()[0].order).toBe('desc');
    });

    it('removeColumn 应移除列', () => {
      const builder = new TableComparatorBuilder<TableRow>()
        .addColumn('score', 'asc', 'number')
        .addColumn('salary', 'desc', 'number');

      builder.removeColumn('score');
      expect(builder.getColumns()).toHaveLength(1);
      expect(builder.getColumns()[0].field).toBe('salary');
    });

    it('clear 应清空所有列', () => {
      const builder = new TableComparatorBuilder<TableRow>()
        .addColumn('score', 'asc', 'number')
        .clear();

      expect(builder.getColumns()).toHaveLength(0);
    });
  });

  // ========================================================================
  // 交互测试
  // ========================================================================

  describe('handleMultiColumnSort', () => {
    it('首次点击应创建升序排序', () => {
      const result = handleMultiColumnSort(SAMPLE_DATA, [], 'score', 'number');

      expect(result.columns).toHaveLength(1);
      expect(result.columns[0]).toEqual({
        field: 'score',
        order: 'asc',
        type: 'number',
      });
    });

    it('再次点击应切换为降序', () => {
      const state1 = handleMultiColumnSort(SAMPLE_DATA, [], 'score', 'number');
      const state2 = handleMultiColumnSort(state1.data, state1.columns, 'score', 'number');

      expect(state2.columns[0].order).toBe('desc');
    });

    it('第三次点击应移除排序', () => {
      const state1 = handleMultiColumnSort(SAMPLE_DATA, [], 'score', 'number');
      const state2 = handleMultiColumnSort(state1.data, state1.columns, 'score', 'number');
      const state3 = handleMultiColumnSort(state2.data, state2.columns, 'score', 'number');

      expect(state3.columns).toHaveLength(0);
    });

    it('Shift+点击应添加次级排序', () => {
      const state1 = handleMultiColumnSort(SAMPLE_DATA, [], 'score', 'number');
      const state2 = handleMultiColumnSort(
        state1.data,
        state1.columns,
        'department',
        'string',
        true // addToExisting
      );

      expect(state2.columns).toHaveLength(2);
      expect(state2.columns[0].field).toBe('score');
      expect(state2.columns[1].field).toBe('department');
    });
  });

  // ========================================================================
  // 边界条件测试
  // ========================================================================

  describe('边界条件', () => {
    it('应处理只有一条数据', () => {
      const single = [SAMPLE_DATA[0]];
      const sorted = sortByMultipleColumns(single, [
        { field: 'score', order: 'asc', type: 'number' },
      ]);

      expect(sorted).toHaveLength(1);
      expect(sorted[0]).toEqual(single[0]);
    });

    it('应处理全相同值', () => {
      const same = SAMPLE_DATA.map(r => ({ ...r, score: 100 }));
      const sorted = sortByMultipleColumns(same, [
        { field: 'score', order: 'asc', type: 'number' },
      ]);

      // 应保持原顺序（稳定性）
      expect(sorted.map(r => r.id)).toEqual(same.map(r => r.id));
    });
  });

  // ========================================================================
  // 性能测试
  // ========================================================================

  describe('性能', () => {
    it('应在合理时间内完成大数据排序', () => {
      const largeData = Array.from({ length: 10000 }, (_, i) => ({
        ...SAMPLE_DATA[i % SAMPLE_DATA.length],
        id: i,
      }));

      const start = performance.now();
      sortByMultipleColumns(largeData, [
        { field: 'department', order: 'asc', type: 'string' },
        { field: 'score', order: 'desc', type: 'number' },
      ]);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(500);
    });
  });
});

