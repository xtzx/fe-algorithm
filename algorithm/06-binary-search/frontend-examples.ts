/**
 * ============================================================
 * 📚 二分查找 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示二分查找在前端实际业务中的应用
 */

// ============================================================
// 1. 虚拟列表 - 定高
// ============================================================

/**
 * 📝 业务场景：定高虚拟列表
 *
 * 场景描述：
 * - 列表有成千上万项
 * - 每项高度固定
 * - 只渲染可视区域的项
 */
class FixedHeightVirtualList {
  private itemHeight: number;
  private containerHeight: number;
  private totalItems: number;

  constructor(itemHeight: number, containerHeight: number, totalItems: number) {
    this.itemHeight = itemHeight;
    this.containerHeight = containerHeight;
    this.totalItems = totalItems;
  }

  /**
   * 根据滚动位置计算可见区域
   * 定高的情况可以直接计算，但这里展示二分的思路
   */
  getVisibleRange(scrollTop: number): { start: number; end: number } {
    const start = Math.floor(scrollTop / this.itemHeight);
    const visibleCount = Math.ceil(this.containerHeight / this.itemHeight);
    const end = Math.min(start + visibleCount + 1, this.totalItems);

    return { start, end };
  }
}

// ============================================================
// 2. 虚拟列表 - 变高
// ============================================================

/**
 * 📝 业务场景：变高虚拟列表
 *
 * 场景描述：
 * - 每项高度不固定
 * - 需要预先计算累加高度
 * - 二分查找第一个可见项
 */
class DynamicHeightVirtualList {
  private heights: number[] = [];
  private prefixHeights: number[] = []; // 累加高度

  constructor(heights: number[]) {
    this.heights = heights;
    this.buildPrefixHeights();
  }

  private buildPrefixHeights(): void {
    this.prefixHeights = [0];
    for (let i = 0; i < this.heights.length; i++) {
      this.prefixHeights.push(
        this.prefixHeights[i] + this.heights[i]
      );
    }
  }

  /**
   * 二分查找第一个可见项的索引
   * 找第一个 prefixHeights[i] > scrollTop 的 i
   */
  getStartIndex(scrollTop: number): number {
    let left = 0;
    let right = this.prefixHeights.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.prefixHeights[mid] > scrollTop) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    return Math.max(0, left - 1);
  }

  /**
   * 获取可见区域
   */
  getVisibleRange(
    scrollTop: number,
    containerHeight: number
  ): { start: number; end: number } {
    const start = this.getStartIndex(scrollTop);

    // 找最后一个可见项
    let end = start;
    let currentHeight = this.prefixHeights[start];
    const endHeight = scrollTop + containerHeight;

    while (end < this.heights.length && currentHeight < endHeight) {
      currentHeight += this.heights[end];
      end++;
    }

    return { start, end };
  }

  /**
   * 获取某项的偏移量
   */
  getItemOffset(index: number): number {
    return this.prefixHeights[index] || 0;
  }

  /**
   * 获取总高度
   */
  getTotalHeight(): number {
    return this.prefixHeights[this.prefixHeights.length - 1];
  }
}

// ============================================================
// 3. 图表数据点查找
// ============================================================

/**
 * 📝 业务场景：图表鼠标交互
 *
 * 场景描述：
 * - 鼠标悬停时显示最近的数据点
 * - 数据按 x 坐标排序
 */
interface DataPoint {
  x: number;
  y: number;
  label?: string;
}

class ChartInteraction {
  private dataPoints: DataPoint[];

  constructor(dataPoints: DataPoint[]) {
    this.dataPoints = dataPoints.sort((a, b) => a.x - b.x);
  }

  /**
   * 找到距离鼠标 x 坐标最近的数据点
   */
  findNearestPoint(mouseX: number): DataPoint | null {
    if (this.dataPoints.length === 0) return null;

    // 二分找第一个 >= mouseX 的位置
    let left = 0;
    let right = this.dataPoints.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.dataPoints[mid].x >= mouseX) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    // 比较左右两个点
    const rightPoint = this.dataPoints[left];
    const leftPoint = this.dataPoints[left - 1];

    if (!leftPoint) return rightPoint;
    if (!rightPoint) return leftPoint;

    // 返回更近的那个
    return Math.abs(leftPoint.x - mouseX) <= Math.abs(rightPoint.x - mouseX)
      ? leftPoint
      : rightPoint;
  }

  /**
   * 找一个 x 范围内的所有点
   */
  findPointsInRange(minX: number, maxX: number): DataPoint[] {
    // 找左边界
    let left = 0;
    let right = this.dataPoints.length;
    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.dataPoints[mid].x >= minX) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const startIndex = left;

    // 找右边界
    left = 0;
    right = this.dataPoints.length;
    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.dataPoints[mid].x > maxX) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const endIndex = left;

    return this.dataPoints.slice(startIndex, endIndex);
  }
}

// ============================================================
// 4. Git Bisect 模拟
// ============================================================

/**
 * 📝 业务场景：查找引入 bug 的提交
 *
 * 场景描述：
 * - 某个版本开始出现 bug
 * - 之前的版本都是好的
 * - 二分定位第一个坏版本
 */
class GitBisect {
  private commits: string[] = [];
  private isBuggy: (commit: string) => boolean;

  constructor(commits: string[], isBuggy: (commit: string) => boolean) {
    this.commits = commits;
    this.isBuggy = isBuggy;
  }

  /**
   * 找到第一个有 bug 的提交
   */
  findFirstBuggyCommit(): { commit: string; index: number } | null {
    let left = 0;
    let right = this.commits.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      console.log(`Testing commit ${mid}: ${this.commits[mid]}`);

      if (this.isBuggy(this.commits[mid])) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    if (left >= this.commits.length) {
      return null;
    }

    return { commit: this.commits[left], index: left };
  }
}

// ============================================================
// 5. IP 地址定位
// ============================================================

/**
 * 📝 业务场景：IP 归属地查询
 *
 * 场景描述：
 * - IP 范围表（起始IP, 结束IP, 地区）
 * - 查询某个 IP 属于哪个地区
 */
interface IPRange {
  start: number;
  end: number;
  region: string;
}

class IPGeolocation {
  private ranges: IPRange[] = [];

  constructor(ranges: IPRange[]) {
    this.ranges = ranges.sort((a, b) => a.start - b.start);
  }

  /**
   * IP 字符串转整数
   */
  static ipToNumber(ip: string): number {
    const parts = ip.split('.').map(Number);
    return (
      (parts[0] << 24) +
      (parts[1] << 16) +
      (parts[2] << 8) +
      parts[3]
    ) >>> 0;
  }

  /**
   * 查找 IP 所属地区
   */
  findRegion(ip: string): string | null {
    const ipNum = IPGeolocation.ipToNumber(ip);

    // 二分找最后一个 start <= ipNum 的范围
    let left = 0;
    let right = this.ranges.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.ranges[mid].start > ipNum) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    const index = left - 1;
    if (index < 0) return null;

    const range = this.ranges[index];
    if (ipNum >= range.start && ipNum <= range.end) {
      return range.region;
    }

    return null;
  }
}

// ============================================================
// 6. 时间刻度定位
// ============================================================

/**
 * 📝 业务场景：时间选择器
 *
 * 场景描述：
 * - 时间轴上有标记点
 * - 拖动时吸附到最近的标记
 */
class TimelineSnap {
  private timestamps: number[] = [];

  constructor(timestamps: number[]) {
    this.timestamps = timestamps.sort((a, b) => a - b);
  }

  /**
   * 找最近的时间点
   */
  snapToNearest(time: number): number {
    if (this.timestamps.length === 0) return time;

    let left = 0;
    let right = this.timestamps.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.timestamps[mid] >= time) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    const rightTime = this.timestamps[left];
    const leftTime = this.timestamps[left - 1];

    if (leftTime === undefined) return rightTime;
    if (rightTime === undefined) return leftTime;

    return Math.abs(time - leftTime) <= Math.abs(time - rightTime)
      ? leftTime
      : rightTime;
  }

  /**
   * 找时间范围内的所有标记
   */
  getMarksInRange(start: number, end: number): number[] {
    // 找左边界
    let left = 0;
    let right = this.timestamps.length;
    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.timestamps[mid] >= start) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const startIdx = left;

    // 找右边界
    left = 0;
    right = this.timestamps.length;
    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.timestamps[mid] > end) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const endIdx = left;

    return this.timestamps.slice(startIdx, endIdx);
  }
}

// ============================================================
// 7. 任务调度优化
// ============================================================

/**
 * 📝 业务场景：并发任务调度
 *
 * 场景描述：
 * - 有 n 个任务，每个任务耗时不同
 * - 有多个 worker
 * - 找最少需要多少 worker 能在时限内完成
 */
class TaskScheduler {
  private taskDurations: number[];

  constructor(taskDurations: number[]) {
    this.taskDurations = taskDurations;
  }

  /**
   * 检查 k 个 worker 能否在 timeLimit 内完成所有任务
   */
  private canFinish(workers: number, timeLimit: number): boolean {
    let currentWorker = 0;
    let currentTime = 0;

    for (const duration of this.taskDurations) {
      if (duration > timeLimit) return false;

      if (currentTime + duration <= timeLimit) {
        currentTime += duration;
      } else {
        currentWorker++;
        if (currentWorker >= workers) return false;
        currentTime = duration;
      }
    }

    return true;
  }

  /**
   * 找最少需要多少 worker
   */
  findMinWorkers(timeLimit: number): number {
    let left = 1;
    let right = this.taskDurations.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.canFinish(mid, timeLimit)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    return left;
  }

  /**
   * 找最短完成时间
   */
  findMinTime(workers: number): number {
    const total = this.taskDurations.reduce((a, b) => a + b, 0);
    const maxTask = Math.max(...this.taskDurations);

    let left = maxTask;
    let right = total;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.canFinish(workers, mid)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    return left;
  }
}

// ============================================================
// 8. 搜索建议/自动补全
// ============================================================

/**
 * 📝 业务场景：搜索建议
 *
 * 场景描述：
 * - 有一个有序词典
 * - 根据前缀快速找匹配的词
 */
class SearchSuggestion {
  private dictionary: string[];

  constructor(words: string[]) {
    this.dictionary = words.sort();
  }

  /**
   * 找所有以 prefix 开头的词
   */
  suggest(prefix: string, limit = 10): string[] {
    // 找第一个 >= prefix 的位置
    let left = 0;
    let right = this.dictionary.length;

    while (left < right) {
      const mid = (left + right) >> 1;
      if (this.dictionary[mid] >= prefix) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    // 从该位置开始收集匹配的词
    const results: string[] = [];
    for (
      let i = left;
      i < this.dictionary.length && results.length < limit;
      i++
    ) {
      if (this.dictionary[i].startsWith(prefix)) {
        results.push(this.dictionary[i]);
      } else {
        break;
      }
    }

    return results;
  }
}

// ============================================================
// 导出
// ============================================================

export {
  FixedHeightVirtualList,
  DynamicHeightVirtualList,
  ChartInteraction,
  GitBisect,
  IPGeolocation,
  TimelineSnap,
  TaskScheduler,
  SearchSuggestion,
};

export type { DataPoint, IPRange };

