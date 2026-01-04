/**
 * 搜索结果数据模型
 */

// ============================================================================
// 类型定义
// ============================================================================

/** 搜索结果 */
export interface SearchResult {
  id: string;
  title: string;
  description: string;
  relevance: number;      // 相关度分数 0-100
  publishTime: number;    // 发布时间戳
  viewCount: number;      // 浏览量
  likeCount: number;      // 点赞数
  category: string;
}

/** 排序维度 */
export type SortDimension = 'relevance' | 'publishTime' | 'viewCount' | 'likeCount';

/** 排序配置 */
export interface SortConfig {
  dimension: SortDimension;
  order: 'asc' | 'desc';
}

/** 分页结果 */
export interface PageResult<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrev: boolean;
}

/** 游标分页结果 */
export interface CursorPageResult<T> {
  items: T[];
  nextCursor: string | null;
  prevCursor: string | null;
  hasMore: boolean;
}

// ============================================================================
// 示例数据生成
// ============================================================================

const CATEGORIES = ['技术', '产品', '设计', '运营', '市场'];

const TITLES = [
  'JavaScript 性能优化技巧',
  'React Hooks 最佳实践',
  'TypeScript 高级类型',
  'CSS Grid 布局详解',
  'Node.js 异步编程',
  'Vue 3 组合式 API',
  'Webpack 构建优化',
  'Git 工作流程',
  'Docker 容器化部署',
  '微服务架构设计',
];

/**
 * 生成搜索结果
 */
export function generateSearchResults(count: number): SearchResult[] {
  const now = Date.now();
  const oneMonth = 30 * 24 * 60 * 60 * 1000;

  return Array.from({ length: count }, (_, i) => ({
    id: `result-${i + 1}`,
    title: `${TITLES[i % TITLES.length]} (${i + 1})`,
    description: `这是第 ${i + 1} 条搜索结果的描述...`,
    relevance: Math.round(Math.random() * 100),
    publishTime: now - Math.floor(Math.random() * oneMonth),
    viewCount: Math.floor(Math.random() * 100000),
    likeCount: Math.floor(Math.random() * 10000),
    category: CATEGORIES[i % CATEGORIES.length],
  }));
}

/**
 * 生成带特定相关度分布的数据
 */
export function generateWithRelevanceDistribution(
  count: number,
  distribution: 'uniform' | 'normal' | 'biased'
): SearchResult[] {
  const results = generateSearchResults(count);

  switch (distribution) {
    case 'uniform':
      // 均匀分布
      results.forEach((r, i) => {
        r.relevance = Math.round((i / count) * 100);
      });
      break;

    case 'normal':
      // 正态分布（集中在 50 附近）
      results.forEach(r => {
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        r.relevance = Math.round(Math.max(0, Math.min(100, 50 + z * 15)));
      });
      break;

    case 'biased':
      // 偏斜分布（少量高分）
      results.forEach(r => {
        r.relevance = Math.round(Math.pow(Math.random(), 2) * 100);
      });
      break;
  }

  return results;
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 格式化时间戳
 */
export function formatTime(timestamp: number): string {
  return new Date(timestamp).toISOString().split('T')[0];
}

/**
 * 打印搜索结果
 */
export function printResults(
  results: SearchResult[],
  columns: (keyof SearchResult)[] = ['id', 'relevance', 'viewCount']
): void {
  const header = columns.map(c => String(c).padEnd(15)).join('| ');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const r of results) {
    const row = columns.map(c => {
      const val = r[c];
      if (c === 'publishTime') {
        return formatTime(val as number).padEnd(15);
      }
      return String(val).padEnd(15);
    }).join('| ');
    console.log(row);
  }
}

