/**
 * ============================================================
 * 📚 动态规划 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示动态规划在前端实际业务中的应用
 */

// ============================================================
// 1. 编辑距离应用 - 拼写检查
// ============================================================

/**
 * 📝 业务场景：拼写检查与建议
 *
 * 场景描述：
 * - 用户输入单词时提供拼写建议
 * - 找出与输入最相似的正确单词
 */
class SpellChecker {
  private dictionary: string[];

  constructor(dictionary: string[]) {
    this.dictionary = dictionary;
  }

  /**
   * 计算编辑距离
   */
  private editDistance(word1: string, word2: string): number {
    const m = word1.length;
    const n = word2.length;

    // dp[j] = word1 前 i 个字符到 word2 前 j 个字符的编辑距离
    let dp: number[] = Array.from({ length: n + 1 }, (_, i) => i);

    for (let i = 1; i <= m; i++) {
      let prev = dp[0];
      dp[0] = i;

      for (let j = 1; j <= n; j++) {
        const temp = dp[j];
        if (word1[i - 1] === word2[j - 1]) {
          dp[j] = prev;
        } else {
          dp[j] = Math.min(prev, dp[j], dp[j - 1]) + 1;
        }
        prev = temp;
      }
    }

    return dp[n];
  }

  /**
   * 获取拼写建议
   */
  getSuggestions(input: string, maxDistance: number = 2): string[] {
    const suggestions: { word: string; distance: number }[] = [];

    for (const word of this.dictionary) {
      const distance = this.editDistance(input.toLowerCase(), word.toLowerCase());
      if (distance <= maxDistance) {
        suggestions.push({ word, distance });
      }
    }

    return suggestions
      .sort((a, b) => a.distance - b.distance)
      .map((s) => s.word);
  }

  /**
   * 检查单词是否正确
   */
  isCorrect(word: string): boolean {
    return this.dictionary.some(
      (w) => w.toLowerCase() === word.toLowerCase()
    );
  }
}

// ============================================================
// 2. LCS 应用 - 简易 Diff 算法
// ============================================================

/**
 * 📝 业务场景：文本差异对比
 *
 * 场景描述：
 * - 比较两个版本的文本差异
 * - 类似 Git diff 的简化实现
 */
interface DiffResult {
  type: 'added' | 'removed' | 'unchanged';
  content: string;
}

class TextDiff {
  /**
   * 计算 LCS（最长公共子序列）
   */
  private getLCS(arr1: string[], arr2: string[]): string[] {
    const m = arr1.length;
    const n = arr2.length;

    // dp[i][j] = arr1 前 i 个和 arr2 前 j 个的 LCS 长度
    const dp: number[][] = Array.from({ length: m + 1 }, () =>
      Array(n + 1).fill(0)
    );

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (arr1[i - 1] === arr2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    // 回溯找出 LCS
    const lcs: string[] = [];
    let i = m,
      j = n;
    while (i > 0 && j > 0) {
      if (arr1[i - 1] === arr2[j - 1]) {
        lcs.unshift(arr1[i - 1]);
        i--;
        j--;
      } else if (dp[i - 1][j] > dp[i][j - 1]) {
        i--;
      } else {
        j--;
      }
    }

    return lcs;
  }

  /**
   * 计算差异
   */
  diff(oldText: string, newText: string): DiffResult[] {
    const oldLines = oldText.split('\n');
    const newLines = newText.split('\n');
    const lcs = this.getLCS(oldLines, newLines);

    const result: DiffResult[] = [];
    let oldIndex = 0;
    let newIndex = 0;
    let lcsIndex = 0;

    while (oldIndex < oldLines.length || newIndex < newLines.length) {
      if (lcsIndex < lcs.length) {
        // 输出旧文本中被删除的行
        while (oldIndex < oldLines.length && oldLines[oldIndex] !== lcs[lcsIndex]) {
          result.push({ type: 'removed', content: oldLines[oldIndex] });
          oldIndex++;
        }

        // 输出新文本中被添加的行
        while (newIndex < newLines.length && newLines[newIndex] !== lcs[lcsIndex]) {
          result.push({ type: 'added', content: newLines[newIndex] });
          newIndex++;
        }

        // 输出未改变的行
        if (lcsIndex < lcs.length) {
          result.push({ type: 'unchanged', content: lcs[lcsIndex] });
          oldIndex++;
          newIndex++;
          lcsIndex++;
        }
      } else {
        // 处理剩余的行
        while (oldIndex < oldLines.length) {
          result.push({ type: 'removed', content: oldLines[oldIndex] });
          oldIndex++;
        }
        while (newIndex < newLines.length) {
          result.push({ type: 'added', content: newLines[newIndex] });
          newIndex++;
        }
      }
    }

    return result;
  }

  /**
   * 格式化输出（类似 Git diff）
   */
  formatDiff(diffs: DiffResult[]): string {
    return diffs
      .map((d) => {
        switch (d.type) {
          case 'added':
            return `+ ${d.content}`;
          case 'removed':
            return `- ${d.content}`;
          default:
            return `  ${d.content}`;
        }
      })
      .join('\n');
  }
}

// ============================================================
// 3. 文本换行（Word Wrap）
// ============================================================

/**
 * 📝 业务场景：最优文本换行
 *
 * 场景描述：
 * - 给定一行的最大宽度，将文本分成多行
 * - 最小化每行末尾空白的平方和（更均匀）
 */
class OptimalWordWrap {
  private lineWidth: number;

  constructor(lineWidth: number) {
    this.lineWidth = lineWidth;
  }

  /**
   * 计算一行的代价（空白的立方）
   */
  private lineCost(words: string[], i: number, j: number): number {
    let length = -1; // 第一个单词前不加空格
    for (let k = i; k <= j; k++) {
      length += words[k].length + 1;
    }

    if (length > this.lineWidth) {
      return Infinity;
    }

    const spaces = this.lineWidth - length;
    return spaces * spaces * spaces;
  }

  /**
   * 最优换行
   */
  wrap(text: string): string[] {
    const words = text.split(/\s+/).filter(Boolean);
    const n = words.length;

    if (n === 0) return [];

    // dp[i] = 从第 i 个单词到末尾的最小代价
    const dp: number[] = new Array(n + 1).fill(0);
    const breaks: number[] = new Array(n + 1).fill(0);

    // 从后往前计算
    for (let i = n - 1; i >= 0; i--) {
      dp[i] = Infinity;

      for (let j = i; j < n; j++) {
        const cost = this.lineCost(words, i, j);

        if (cost === Infinity) break;

        // 最后一行不计代价
        const totalCost = j === n - 1 ? 0 : cost + dp[j + 1];

        if (totalCost < dp[i]) {
          dp[i] = totalCost;
          breaks[i] = j + 1;
        }
      }
    }

    // 构建结果
    const lines: string[] = [];
    let i = 0;
    while (i < n) {
      const j = breaks[i];
      lines.push(words.slice(i, j).join(' '));
      i = j;
    }

    return lines;
  }
}

// ============================================================
// 4. 股票分析 - 最佳买卖时机
// ============================================================

/**
 * 📝 业务场景：股票交易分析
 *
 * 场景描述：
 * - 分析历史价格，找出最佳买卖策略
 * - 支持多种交易规则
 */
interface TradeResult {
  buyDay: number;
  sellDay: number;
  profit: number;
}

class StockAnalyzer {
  /**
   * 只能交易一次的最大利润
   */
  maxProfitOnce(prices: number[]): TradeResult | null {
    if (prices.length < 2) return null;

    let minPrice = prices[0];
    let minDay = 0;
    let maxProfit = 0;
    let result: TradeResult | null = null;

    for (let i = 1; i < prices.length; i++) {
      const profit = prices[i] - minPrice;
      if (profit > maxProfit) {
        maxProfit = profit;
        result = {
          buyDay: minDay,
          sellDay: i,
          profit: maxProfit,
        };
      }

      if (prices[i] < minPrice) {
        minPrice = prices[i];
        minDay = i;
      }
    }

    return result;
  }

  /**
   * 可以交易多次的最大利润
   */
  maxProfitUnlimited(prices: number[]): TradeResult[] {
    const trades: TradeResult[] = [];
    let buyDay = 0;
    let inPosition = false;

    for (let i = 1; i < prices.length; i++) {
      if (!inPosition && prices[i] > prices[i - 1]) {
        buyDay = i - 1;
        inPosition = true;
      } else if (inPosition && prices[i] < prices[i - 1]) {
        trades.push({
          buyDay,
          sellDay: i - 1,
          profit: prices[i - 1] - prices[buyDay],
        });
        inPosition = false;
      }
    }

    // 最后一段上涨
    if (inPosition) {
      trades.push({
        buyDay,
        sellDay: prices.length - 1,
        profit: prices[prices.length - 1] - prices[buyDay],
      });
    }

    return trades;
  }

  /**
   * 带冷冻期的最大利润
   */
  maxProfitWithCooldown(prices: number[]): number {
    const n = prices.length;
    if (n < 2) return 0;

    // hold: 持有股票
    // sold: 刚卖出（冷冻期）
    // rest: 不持有且不在冷冻期
    let hold = -prices[0];
    let sold = 0;
    let rest = 0;

    for (let i = 1; i < n; i++) {
      const prevHold = hold;
      const prevSold = sold;
      const prevRest = rest;

      hold = Math.max(prevHold, prevRest - prices[i]);
      sold = prevHold + prices[i];
      rest = Math.max(prevRest, prevSold);
    }

    return Math.max(sold, rest);
  }
}

// ============================================================
// 5. 瀑布流布局优化
// ============================================================

/**
 * 📝 业务场景：瀑布流布局
 *
 * 场景描述：
 * - 将不同高度的元素分配到多列
 * - 使各列高度尽量均匀
 */
interface LayoutItem {
  id: string;
  height: number;
}

interface ColumnLayout {
  items: LayoutItem[];
  totalHeight: number;
}

class WaterfallLayout {
  private columns: number;

  constructor(columns: number) {
    this.columns = columns;
  }

  /**
   * 贪心算法：每次放入最短的列
   */
  layoutGreedy(items: LayoutItem[]): ColumnLayout[] {
    const columns: ColumnLayout[] = Array.from({ length: this.columns }, () => ({
      items: [],
      totalHeight: 0,
    }));

    for (const item of items) {
      // 找最短的列
      let minColumn = 0;
      for (let i = 1; i < this.columns; i++) {
        if (columns[i].totalHeight < columns[minColumn].totalHeight) {
          minColumn = i;
        }
      }

      columns[minColumn].items.push(item);
      columns[minColumn].totalHeight += item.height;
    }

    return columns;
  }

  /**
   * DP 优化：使高度差最小（简化版，适用于列数较少的情况）
   * 类似于分割等和子集问题
   */
  layoutOptimal(items: LayoutItem[]): ColumnLayout[] {
    // 对于两列的情况，转化为"分割等和子集"问题
    if (this.columns === 2) {
      return this.layoutTwoColumns(items);
    }

    // 多列情况使用贪心（DP 复杂度过高）
    return this.layoutGreedy(items);
  }

  private layoutTwoColumns(items: LayoutItem[]): ColumnLayout[] {
    const totalHeight = items.reduce((sum, item) => sum + item.height, 0);
    const target = Math.floor(totalHeight / 2);

    // dp[j] = 是否可以恰好达到高度 j
    const dp: boolean[] = new Array(target + 1).fill(false);
    dp[0] = true;

    // 记录选择
    const choices: number[][] = Array.from({ length: target + 1 }, () => []);

    for (let i = 0; i < items.length; i++) {
      const h = items[i].height;
      for (let j = target; j >= h; j--) {
        if (dp[j - h] && !dp[j]) {
          dp[j] = true;
          choices[j] = [...choices[j - h], i];
        }
      }
    }

    // 找到最接近 target 的可达高度
    let bestHeight = 0;
    for (let j = target; j >= 0; j--) {
      if (dp[j]) {
        bestHeight = j;
        break;
      }
    }

    // 构建两列
    const inColumn1 = new Set(choices[bestHeight]);
    const columns: ColumnLayout[] = [
      { items: [], totalHeight: 0 },
      { items: [], totalHeight: 0 },
    ];

    for (let i = 0; i < items.length; i++) {
      const col = inColumn1.has(i) ? 0 : 1;
      columns[col].items.push(items[i]);
      columns[col].totalHeight += items[i].height;
    }

    return columns;
  }
}

// ============================================================
// 6. 正则表达式匹配（简化版）
// ============================================================

/**
 * 📝 业务场景：通配符匹配
 *
 * 场景描述：
 * - 支持 * (匹配任意) 和 ? (匹配单个) 的简单匹配
 */
class WildcardMatcher {
  /**
   * 通配符匹配
   * ? 匹配任意单个字符
   * * 匹配任意字符串（包括空串）
   */
  isMatch(str: string, pattern: string): boolean {
    const m = str.length;
    const n = pattern.length;

    // dp[i][j] = str 前 i 个字符是否匹配 pattern 前 j 个字符
    const dp: boolean[][] = Array.from({ length: m + 1 }, () =>
      new Array(n + 1).fill(false)
    );

    dp[0][0] = true;

    // 初始化：pattern 开头的 * 可以匹配空串
    for (let j = 1; j <= n; j++) {
      if (pattern[j - 1] === '*') {
        dp[0][j] = dp[0][j - 1];
      }
    }

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (pattern[j - 1] === '*') {
          // * 匹配空或任意字符
          dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
        } else if (pattern[j - 1] === '?' || str[i - 1] === pattern[j - 1]) {
          // ? 或完全匹配
          dp[i][j] = dp[i - 1][j - 1];
        }
      }
    }

    return dp[m][n];
  }
}

// ============================================================
// 7. 任务调度优化（背包变体）
// ============================================================

/**
 * 📝 业务场景：任务调度
 *
 * 场景描述：
 * - 在有限时间内选择最有价值的任务
 * - 类似于 0-1 背包问题
 */
interface Task {
  id: string;
  duration: number; // 耗时
  priority: number; // 优先级/价值
}

class TaskScheduler {
  /**
   * 在给定时间内选择最优任务组合
   */
  selectTasks(tasks: Task[], availableTime: number): Task[] {
    const n = tasks.length;

    // dp[j] = 时间 j 内的最大价值
    const dp: number[] = new Array(availableTime + 1).fill(0);

    // 记录选择
    const selections: Task[][] = Array.from(
      { length: availableTime + 1 },
      () => []
    );

    for (const task of tasks) {
      // 0-1 背包：从后往前
      for (let j = availableTime; j >= task.duration; j--) {
        const newValue = dp[j - task.duration] + task.priority;
        if (newValue > dp[j]) {
          dp[j] = newValue;
          selections[j] = [...selections[j - task.duration], task];
        }
      }
    }

    return selections[availableTime];
  }

  /**
   * 计算完成所有任务的最短时间（可并行）
   */
  minTimeParallel(tasks: Task[], workers: number): number {
    // 二分搜索最小时间
    const totalTime = tasks.reduce((sum, t) => sum + t.duration, 0);
    let left = Math.max(...tasks.map((t) => t.duration));
    let right = totalTime;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (this.canFinishInTime(tasks, workers, mid)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    return left;
  }

  private canFinishInTime(
    tasks: Task[],
    workers: number,
    timeLimit: number
  ): boolean {
    // 贪心检查
    const sorted = [...tasks].sort((a, b) => b.duration - a.duration);
    const workerTimes = new Array(workers).fill(0);

    for (const task of sorted) {
      // 找最空闲的 worker
      const minIdx = workerTimes.indexOf(Math.min(...workerTimes));
      workerTimes[minIdx] += task.duration;
      if (workerTimes[minIdx] > timeLimit) return false;
    }

    return true;
  }
}

// ============================================================
// 8. 最长递增子序列应用 - 趋势分析
// ============================================================

/**
 * 📝 业务场景：趋势分析
 *
 * 场景描述：
 * - 分析数据趋势
 * - 找出最长的递增/递减趋势
 */
interface TrendResult {
  startIndex: number;
  endIndex: number;
  values: number[];
  length: number;
}

class TrendAnalyzer {
  /**
   * 最长递增趋势（LIS）- O(n log n) 优化版
   */
  longestIncreasingTrend(data: number[]): TrendResult {
    const n = data.length;
    if (n === 0) return { startIndex: 0, endIndex: 0, values: [], length: 0 };

    // tails[i] = 长度为 i+1 的 LIS 的最小结尾值
    const tails: number[] = [];
    // 记录每个元素在 LIS 中的位置
    const positions: number[] = new Array(n);
    // 记录前驱
    const predecessors: number[] = new Array(n).fill(-1);

    for (let i = 0; i < n; i++) {
      // 二分搜索
      let left = 0,
        right = tails.length;
      while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (tails[mid] < data[i]) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }

      positions[i] = left;

      if (left > 0) {
        // 找前驱
        for (let j = i - 1; j >= 0; j--) {
          if (positions[j] === left - 1 && data[j] < data[i]) {
            predecessors[i] = j;
            break;
          }
        }
      }

      if (left === tails.length) {
        tails.push(data[i]);
      } else {
        tails[left] = data[i];
      }
    }

    // 回溯找出 LIS
    const lisLength = tails.length;
    let currentPos = lisLength - 1;
    let endIndex = -1;

    // 找到最后一个位置
    for (let i = n - 1; i >= 0; i--) {
      if (positions[i] === currentPos) {
        endIndex = i;
        break;
      }
    }

    // 回溯
    const values: number[] = [];
    let idx = endIndex;
    while (idx !== -1) {
      values.unshift(data[idx]);
      idx = predecessors[idx];
    }

    const startIndex = endIndex - values.length + 1;

    return {
      startIndex,
      endIndex,
      values,
      length: lisLength,
    };
  }
}

// ============================================================
// 导出
// ============================================================

export {
  SpellChecker,
  TextDiff,
  OptimalWordWrap,
  StockAnalyzer,
  WaterfallLayout,
  WildcardMatcher,
  TaskScheduler,
  TrendAnalyzer,
};

export type {
  DiffResult,
  TradeResult,
  LayoutItem,
  ColumnLayout,
  Task,
  TrendResult,
};

