/**
 * ============================================================
 * 📚 数组与双指针 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示双指针、滑动窗口、前缀和在前端实际业务中的应用
 */

// ============================================================
// 1. 滑动窗口 - 实时统计最近 N 秒的请求数（限流/熔断）
// ============================================================

/**
 * 📝 业务场景：前端限流器
 *
 * 场景描述：
 * - 监控 API 调用频率
 * - 如果最近 N 秒内请求数超过阈值，触发限流
 * - 用于前端防抖之外的二层保护
 */
class SlidingWindowRateLimiter {
  private windowSize: number; // 窗口大小（毫秒）
  private maxRequests: number; // 窗口内最大请求数
  private requests: number[] = []; // 请求时间戳队列

  constructor(windowSizeMs: number, maxRequests: number) {
    this.windowSize = windowSizeMs;
    this.maxRequests = maxRequests;
  }

  /**
   * 检查是否允许请求
   * 滑动窗口思想：维护一个时间窗口，只保留窗口内的请求
   */
  allowRequest(): boolean {
    const now = Date.now();
    const windowStart = now - this.windowSize;

    // 滑动窗口：移除过期的请求（left 指针右移）
    while (this.requests.length > 0 && this.requests[0] < windowStart) {
      this.requests.shift();
    }

    // 检查是否超过限制
    if (this.requests.length >= this.maxRequests) {
      return false;
    }

    // 记录当前请求（right 指针右移）
    this.requests.push(now);
    return true;
  }

  /**
   * 获取当前窗口内的请求数
   */
  getCurrentCount(): number {
    const now = Date.now();
    const windowStart = now - this.windowSize;

    // 清理过期请求
    while (this.requests.length > 0 && this.requests[0] < windowStart) {
      this.requests.shift();
    }

    return this.requests.length;
  }
}

// 使用示例
const rateLimiter = new SlidingWindowRateLimiter(1000, 10); // 1秒内最多10次

async function fetchWithRateLimit(url: string): Promise<Response | null> {
  if (!rateLimiter.allowRequest()) {
    console.warn('请求被限流，请稍后再试');
    return null;
  }
  return fetch(url);
}

// ============================================================
// 2. 滑动窗口 - 移动平均值计算（图表/数据分析）
// ============================================================

/**
 * 📝 业务场景：股票K线图的移动平均线
 *
 * 场景描述：
 * - 计算最近 N 个数据点的平均值
 * - 常用于平滑曲线、趋势分析
 */
class MovingAverage {
  private windowSize: number;
  private window: number[] = [];
  private sum = 0;

  constructor(size: number) {
    this.windowSize = size;
  }

  /**
   * 添加新数据点，返回当前移动平均值
   * 滑动窗口：O(1) 时间维护窗口和
   */
  next(val: number): number {
    // 扩张：加入新元素
    this.window.push(val);
    this.sum += val;

    // 收缩：移除超出窗口的元素
    if (this.window.length > this.windowSize) {
      this.sum -= this.window.shift()!;
    }

    return this.sum / this.window.length;
  }
}

// 使用示例：计算5日移动平均
const ma5 = new MovingAverage(5);
const stockPrices = [100, 102, 105, 103, 108, 110, 112];
const maLine = stockPrices.map((price) => ({
  price,
  ma5: ma5.next(price).toFixed(2),
}));
// console.log(maLine);

// ============================================================
// 3. 前缀和 - 表格区间求和（Excel 式快速计算）
// ============================================================

/**
 * 📝 业务场景：报表数据区间汇总
 *
 * 场景描述：
 * - 用户可以选择任意时间范围查看数据汇总
 * - 需要快速计算任意区间的总和
 */
class RangeSum {
  private prefixSum: number[] = [0];

  constructor(nums: number[]) {
    // 预处理：构建前缀和数组
    for (const num of nums) {
      this.prefixSum.push(this.prefixSum[this.prefixSum.length - 1] + num);
    }
  }

  /**
   * O(1) 时间查询区间 [left, right] 的和
   */
  query(left: number, right: number): number {
    return this.prefixSum[right + 1] - this.prefixSum[left];
  }
}

// 使用示例：月度销售数据快速汇总
const monthlySales = [120, 150, 180, 200, 160, 220, 190, 210, 230, 180, 250, 300];
const salesRangeSum = new RangeSum(monthlySales);

// 快速查询任意季度的销售总额
const q1Sales = salesRangeSum.query(0, 2); // 1-3月
const q2Sales = salesRangeSum.query(3, 5); // 4-6月
const h1Sales = salesRangeSum.query(0, 5); // 上半年

// console.log({ q1Sales, q2Sales, h1Sales });

// ============================================================
// 4. 双指针 - 合并有序日志列表
// ============================================================

/**
 * 📝 业务场景：合并多个来源的日志
 *
 * 场景描述：
 * - 前端有多个日志来源（用户操作日志、网络请求日志、错误日志）
 * - 每个日志源按时间排序
 * - 需要合并成统一的时间线展示
 */
interface LogEntry {
  timestamp: number;
  type: string;
  message: string;
}

function mergeSortedLogs(logs1: LogEntry[], logs2: LogEntry[]): LogEntry[] {
  const result: LogEntry[] = [];
  let i = 0;
  let j = 0;

  // 双指针合并两个有序数组
  while (i < logs1.length && j < logs2.length) {
    if (logs1[i].timestamp <= logs2[j].timestamp) {
      result.push(logs1[i]);
      i++;
    } else {
      result.push(logs2[j]);
      j++;
    }
  }

  // 处理剩余元素
  while (i < logs1.length) {
    result.push(logs1[i]);
    i++;
  }
  while (j < logs2.length) {
    result.push(logs2[j]);
    j++;
  }

  return result;
}

// 合并多个日志源
function mergeMultipleLogs(...logSources: LogEntry[][]): LogEntry[] {
  return logSources.reduce((merged, current) => mergeSortedLogs(merged, current), []);
}

// ============================================================
// 5. 快慢指针 - 去重保留最新 N 条记录
// ============================================================

/**
 * 📝 业务场景：搜索历史记录去重
 *
 * 场景描述：
 * - 用户搜索历史可能有重复
 * - 需要去重，保留最新的搜索记录
 */
function deduplicateSearchHistory(history: string[]): string[] {
  // 反转，让最新的在前面
  const reversed = [...history].reverse();
  const seen = new Set<string>();
  const result: string[] = [];

  // 快慢指针思想的变体：只保留第一次出现的
  for (const item of reversed) {
    if (!seen.has(item)) {
      seen.add(item);
      result.push(item);
    }
  }

  // 反转回来，最新的在最后
  return result.reverse();
}

// 使用示例
const searchHistory = ['React', 'Vue', 'React', 'Angular', 'Vue', 'React'];
const deduped = deduplicateSearchHistory(searchHistory);
// console.log(deduped); // ['Angular', 'Vue', 'React'] - 每个词只保留最后一次出现

// ============================================================
// 6. 滑动窗口 - 虚拟滚动可视区域计算
// ============================================================

/**
 * 📝 业务场景：长列表虚拟滚动
 *
 * 场景描述：
 * - 渲染超长列表（如 10000 条数据）
 * - 只渲染可视区域内的元素
 * - 滚动时动态更新可视范围
 */
interface VirtualListConfig {
  itemHeight: number; // 每个项目的高度
  containerHeight: number; // 容器高度
  buffer: number; // 上下缓冲区项目数
}

interface VisibleRange {
  startIndex: number;
  endIndex: number;
  offsetTop: number;
}

function calculateVisibleRange(
  scrollTop: number,
  totalItems: number,
  config: VirtualListConfig
): VisibleRange {
  const { itemHeight, containerHeight, buffer } = config;

  // 计算可视区域的起始和结束索引（双指针思想）
  const visibleStart = Math.floor(scrollTop / itemHeight);
  const visibleEnd = Math.ceil((scrollTop + containerHeight) / itemHeight);

  // 加上缓冲区
  const startIndex = Math.max(0, visibleStart - buffer);
  const endIndex = Math.min(totalItems - 1, visibleEnd + buffer);

  // 计算偏移量（用于 transform）
  const offsetTop = startIndex * itemHeight;

  return { startIndex, endIndex, offsetTop };
}

// React 伪代码示例
/*
function VirtualList({ items, itemHeight, containerHeight }) {
  const [scrollTop, setScrollTop] = useState(0);

  const { startIndex, endIndex, offsetTop } = calculateVisibleRange(
    scrollTop,
    items.length,
    { itemHeight, containerHeight, buffer: 5 }
  );

  const visibleItems = items.slice(startIndex, endIndex + 1);

  return (
    <div
      style={{ height: containerHeight, overflow: 'auto' }}
      onScroll={e => setScrollTop(e.target.scrollTop)}
    >
      <div style={{ height: items.length * itemHeight }}>
        <div style={{ transform: `translateY(${offsetTop}px)` }}>
          {visibleItems.map((item, i) => (
            <div key={startIndex + i} style={{ height: itemHeight }}>
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
*/

// ============================================================
// 7. 滑动窗口 - 性能指标统计（最近 N 次的平均响应时间）
// ============================================================

/**
 * 📝 业务场景：前端性能监控
 *
 * 场景描述：
 * - 监控 API 响应时间
 * - 计算最近 N 次请求的平均响应时间
 * - 计算 p95/p99 等百分位数
 */
class PerformanceMonitor {
  private windowSize: number;
  private responseTimes: number[] = [];

  constructor(windowSize: number) {
    this.windowSize = windowSize;
  }

  record(responseTime: number): void {
    this.responseTimes.push(responseTime);

    // 滑动窗口：只保留最近 N 次记录
    if (this.responseTimes.length > this.windowSize) {
      this.responseTimes.shift();
    }
  }

  getAverageResponseTime(): number {
    if (this.responseTimes.length === 0) return 0;
    const sum = this.responseTimes.reduce((a, b) => a + b, 0);
    return sum / this.responseTimes.length;
  }

  getPercentile(percentile: number): number {
    if (this.responseTimes.length === 0) return 0;

    const sorted = [...this.responseTimes].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  getP95(): number {
    return this.getPercentile(95);
  }

  getP99(): number {
    return this.getPercentile(99);
  }
}

// 使用示例
const perfMonitor = new PerformanceMonitor(100); // 最近100次请求

// 模拟记录响应时间
// [50, 60, 45, 200, 55, ...].forEach(time => perfMonitor.record(time));

// console.log({
//   avg: perfMonitor.getAverageResponseTime(),
//   p95: perfMonitor.getP95(),
//   p99: perfMonitor.getP99()
// });

// ============================================================
// 8. 前缀和 - 进度条计算
// ============================================================

/**
 * 📝 业务场景：多步骤表单进度计算
 *
 * 场景描述：
 * - 每个步骤有不同的权重
 * - 需要计算当前完成的进度百分比
 */
interface FormStep {
  id: string;
  name: string;
  weight: number; // 权重
  completed: boolean;
}

function calculateProgress(steps: FormStep[]): number {
  // 构建权重前缀和
  const weights = steps.map((s) => s.weight);
  const prefixSum = [0];
  for (const w of weights) {
    prefixSum.push(prefixSum[prefixSum.length - 1] + w);
  }

  const totalWeight = prefixSum[prefixSum.length - 1];

  // 计算已完成的权重
  let completedWeight = 0;
  for (let i = 0; i < steps.length; i++) {
    if (steps[i].completed) {
      completedWeight += steps[i].weight;
    }
  }

  return totalWeight > 0 ? (completedWeight / totalWeight) * 100 : 0;
}

// 使用示例
const formSteps: FormStep[] = [
  { id: '1', name: '基本信息', weight: 20, completed: true },
  { id: '2', name: '详细资料', weight: 30, completed: true },
  { id: '3', name: '上传文件', weight: 30, completed: false },
  { id: '4', name: '确认提交', weight: 20, completed: false },
];

const progress = calculateProgress(formSteps);
// console.log(`当前进度: ${progress}%`); // 50%

// ============================================================
// 导出
// ============================================================

export {
  SlidingWindowRateLimiter,
  MovingAverage,
  RangeSum,
  mergeSortedLogs,
  mergeMultipleLogs,
  deduplicateSearchHistory,
  calculateVisibleRange,
  PerformanceMonitor,
  calculateProgress,
};

