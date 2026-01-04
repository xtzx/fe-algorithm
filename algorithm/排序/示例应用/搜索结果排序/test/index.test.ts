/**
 * 搜索结果排序测试
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { generateSearchResults, type SearchResult } from '../src/数据模型';
import {
  MinHeap,
  topKByHeap,
  topKBySort,
  topKWithFilter,
} from '../src/TopK小顶堆';
import {
  TopKTracker,
  SlidingWindowTopK,
  Leaderboard,
} from '../src/增量更新排序';
import {
  paginateByOffset,
  paginateWithCursor,
  PaginationManager,
} from '../src/分页与游标';
import { numberAsc } from '../../../算法包/公共库/src/比较器';

// ============================================================================
// TopK 测试
// ============================================================================

describe('TopK小顶堆', () => {
  const cmp = (a: SearchResult, b: SearchResult) => a.relevance - b.relevance;

  describe('MinHeap', () => {
    it('应正确维护最小堆性质', () => {
      const heap = new MinHeap<number>(numberAsc);

      heap.push(5);
      heap.push(3);
      heap.push(7);
      heap.push(1);

      expect(heap.peek()).toBe(1);
      expect(heap.pop()).toBe(1);
      expect(heap.pop()).toBe(3);
      expect(heap.pop()).toBe(5);
      expect(heap.pop()).toBe(7);
    });

    it('replaceTop 应正确工作', () => {
      const heap = new MinHeap<number>(numberAsc);

      heap.push(1);
      heap.push(2);
      heap.push(3);

      heap.replaceTop(4);
      expect(heap.peek()).toBe(2);
    });
  });

  describe('topKByHeap', () => {
    it('应返回最大的 K 个元素', () => {
      const results = generateSearchResults(100);
      const top10 = topKByHeap(results, 10, cmp);

      expect(top10).toHaveLength(10);

      // 验证是最大的 10 个
      const sortedAll = [...results].sort((a, b) => b.relevance - a.relevance);
      const expectedIds = sortedAll.slice(0, 10).map(r => r.id);

      expect(top10.map(r => r.id).sort()).toEqual(expectedIds.sort());
    });

    it('应返回降序排列', () => {
      const results = generateSearchResults(50);
      const top10 = topKByHeap(results, 10, cmp);

      for (let i = 1; i < top10.length; i++) {
        expect(top10[i].relevance).toBeLessThanOrEqual(top10[i - 1].relevance);
      }
    });

    it('k > n 时应返回所有元素', () => {
      const results = generateSearchResults(5);
      const top10 = topKByHeap(results, 10, cmp);

      expect(top10).toHaveLength(5);
    });

    it('k = 0 应返回空数组', () => {
      const results = generateSearchResults(10);
      expect(topKByHeap(results, 0, cmp)).toEqual([]);
    });
  });

  describe('topKByHeap vs topKBySort', () => {
    it('两种方法应返回相同结果', () => {
      const results = generateSearchResults(100);

      const heap = topKByHeap(results, 10, cmp);
      const sort = topKBySort(results, 10, cmp);

      expect(heap.map(r => r.id).sort()).toEqual(sort.map(r => r.id).sort());
    });
  });

  describe('topKWithFilter', () => {
    it('应只返回满足条件的元素', () => {
      const results = generateSearchResults(100);

      const filtered = topKWithFilter(
        results,
        10,
        r => r.relevance > 50,
        cmp
      );

      filtered.forEach(r => {
        expect(r.relevance).toBeGreaterThan(50);
      });
    });
  });
});

// ============================================================================
// 增量更新测试
// ============================================================================

describe('增量更新排序', () => {
  const cmp = (a: SearchResult, b: SearchResult) => a.relevance - b.relevance;

  describe('TopKTracker', () => {
    let tracker: TopKTracker<SearchResult>;

    beforeEach(() => {
      tracker = new TopKTracker(5, cmp);
    });

    it('应正确追踪 TopK', () => {
      const results = generateSearchResults(20);
      tracker.addBatch(results);

      const topK = tracker.getTopK();
      expect(topK).toHaveLength(5);

      // 验证是最大的 5 个
      const sortedAll = [...results].sort((a, b) => b.relevance - a.relevance);
      expect(topK.map(r => r.id).sort()).toEqual(
        sortedAll.slice(0, 5).map(r => r.id).sort()
      );
    });

    it('add 应返回是否进入 TopK', () => {
      // 添加低分数据
      for (let i = 0; i < 5; i++) {
        const result: SearchResult = {
          id: `low-${i}`,
          title: '',
          description: '',
          relevance: 10,
          publishTime: Date.now(),
          viewCount: 0,
          likeCount: 0,
          category: '',
        };
        expect(tracker.add(result)).toBe(true);
      }

      // 添加更低分数据，应该不进入
      const lowResult: SearchResult = {
        id: 'very-low',
        title: '',
        description: '',
        relevance: 5,
        publishTime: Date.now(),
        viewCount: 0,
        likeCount: 0,
        category: '',
      };
      expect(tracker.add(lowResult)).toBe(false);

      // 添加高分数据，应该进入
      const highResult: SearchResult = {
        id: 'high',
        title: '',
        description: '',
        relevance: 99,
        publishTime: Date.now(),
        viewCount: 0,
        likeCount: 0,
        category: '',
      };
      expect(tracker.add(highResult)).toBe(true);
    });

    it('getThreshold 应返回门槛值', () => {
      const results = generateSearchResults(10);
      tracker.addBatch(results);

      const threshold = tracker.getThreshold();
      const topK = tracker.getTopK();

      // 门槛值应该是 TopK 中最小的
      expect(threshold?.relevance).toBe(topK[topK.length - 1].relevance);
    });

    it('getStats 应返回正确统计', () => {
      tracker.addBatch(generateSearchResults(10));
      const stats = tracker.getStats();

      expect(stats.k).toBe(5);
      expect(stats.currentSize).toBe(5);
      expect(stats.totalAdded).toBe(10);
    });
  });

  describe('SlidingWindowTopK', () => {
    it('应只保留窗口内数据', () => {
      const sliding = new SlidingWindowTopK<number>(5, 2, numberAsc);

      for (let i = 0; i < 10; i++) {
        sliding.add(i);
      }

      // 窗口内是 5-9，Top 2 应该是 8, 9
      expect(sliding.size()).toBe(5);
      expect(sliding.getTopK()).toEqual([9, 8]);
    });
  });

  describe('Leaderboard', () => {
    interface Player { id: string; score: number }

    it('应正确维护排行榜', () => {
      const board = new Leaderboard<Player>(
        3,
        p => p.id,
        (a, b) => a.score - b.score
      );

      board.upsert({ id: 'a', score: 100 });
      board.upsert({ id: 'b', score: 200 });
      board.upsert({ id: 'c', score: 150 });

      const top3 = board.getLeaderboard();
      expect(top3.map(p => p.id)).toEqual(['b', 'c', 'a']);
    });

    it('upsert 应更新已有元素', () => {
      const board = new Leaderboard<Player>(
        3,
        p => p.id,
        (a, b) => a.score - b.score
      );

      board.upsert({ id: 'a', score: 100 });
      board.upsert({ id: 'a', score: 200 });

      expect(board.size()).toBe(1);
      expect(board.getLeaderboard()[0].score).toBe(200);
    });

    it('getRank 应返回正确排名', () => {
      const board = new Leaderboard<Player>(
        5,
        p => p.id,
        (a, b) => a.score - b.score
      );

      board.upsert({ id: 'a', score: 100 });
      board.upsert({ id: 'b', score: 200 });
      board.upsert({ id: 'c', score: 150 });

      expect(board.getRank('b')).toBe(1);
      expect(board.getRank('c')).toBe(2);
      expect(board.getRank('a')).toBe(3);
    });
  });
});

// ============================================================================
// 分页测试
// ============================================================================

describe('分页与游标', () => {
  const data = Array.from({ length: 100 }, (_, i) => ({ id: `item-${i}`, value: i }));

  describe('paginateByOffset', () => {
    it('应返回正确的页面数据', () => {
      const page1 = paginateByOffset(data, 1, 10);

      expect(page1.items).toHaveLength(10);
      expect(page1.items[0].id).toBe('item-0');
      expect(page1.page).toBe(1);
      expect(page1.hasNext).toBe(true);
      expect(page1.hasPrev).toBe(false);
    });

    it('最后一页应正确处理', () => {
      const lastPage = paginateByOffset(data, 10, 10);

      expect(lastPage.items).toHaveLength(10);
      expect(lastPage.hasNext).toBe(false);
      expect(lastPage.hasPrev).toBe(true);
    });

    it('页码超出范围应返回有效页', () => {
      const overPage = paginateByOffset(data, 999, 10);

      expect(overPage.page).toBe(10);
    });
  });

  describe('paginateWithCursor', () => {
    it('首页应正确返回', () => {
      const page1 = paginateWithCursor(data, null, 10, d => d.id);

      expect(page1.items).toHaveLength(10);
      expect(page1.items[0].id).toBe('item-0');
      expect(page1.nextCursor).toBe('item-9');
      expect(page1.hasMore).toBe(true);
    });

    it('使用游标应返回下一页', () => {
      const page2 = paginateWithCursor(data, 'item-9', 10, d => d.id);

      expect(page2.items[0].id).toBe('item-10');
      expect(page2.items).toHaveLength(10);
    });

    it('游标失效应从头开始', () => {
      const page = paginateWithCursor(data, 'invalid-cursor', 10, d => d.id);

      expect(page.items[0].id).toBe('item-0');
    });
  });

  describe('PaginationManager', () => {
    let manager: PaginationManager<typeof data[0]>;

    beforeEach(() => {
      manager = new PaginationManager(10);
      manager.setData(data);
    });

    it('getCurrentPage 应返回当前页', () => {
      const page = manager.getCurrentPage();

      expect(page.page).toBe(1);
      expect(page.items).toHaveLength(10);
    });

    it('nextPage 应前进', () => {
      manager.nextPage();
      const page = manager.getCurrentPage();

      expect(page.page).toBe(2);
    });

    it('prevPage 在第一页应返回 null', () => {
      expect(manager.prevPage()).toBeNull();
    });

    it('goToPage 应跳转', () => {
      manager.goToPage(5);
      const page = manager.getCurrentPage();

      expect(page.page).toBe(5);
    });
  });
});

// ============================================================================
// 综合测试
// ============================================================================

describe('综合场景', () => {
  it('TopK 增量更新不应破坏排序不变量', () => {
    const cmp = (a: SearchResult, b: SearchResult) => a.relevance - b.relevance;
    const tracker = new TopKTracker<SearchResult>(10, cmp);

    // 添加多批数据
    for (let batch = 0; batch < 10; batch++) {
      const results = generateSearchResults(100);
      tracker.addBatch(results);

      // 每次都验证 TopK 是降序的
      const topK = tracker.getTopK();
      for (let i = 1; i < topK.length; i++) {
        expect(topK[i].relevance).toBeLessThanOrEqual(topK[i - 1].relevance);
      }
    }
  });

  it('分页数据应与 TopK 结果一致', () => {
    const results = generateSearchResults(100);
    const cmp = (a: SearchResult, b: SearchResult) => a.relevance - b.relevance;

    // 获取 Top 50
    const top50 = topKByHeap(results, 50, cmp);

    // 分页第 1 页
    const page1 = paginateByOffset(top50, 1, 10);

    // 应该是 relevance 最高的 10 个
    expect(page1.items[0].relevance).toBe(top50[0].relevance);
  });
});

