/**
 * 📝 题目：前 K 个高频元素
 * 🔗 链接：https://leetcode.cn/problems/top-k-frequent-elements/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：数组、哈希表、分治、桶排序、计数、快速选择、排序、堆
 *
 * 📋 题目描述：
 * 给你一个整数数组 nums 和一个整数 k，请你返回其中出现频率前 k 高的元素。
 * 可以按任意顺序返回答案。
 *
 * 示例：
 * 输入：nums = [1,1,1,2,2,3], k = 2
 * 输出：[1,2]
 *
 * 输入：nums = [1], k = 1
 * 输出：[1]
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 两步：1. 统计频率 2. 找前 K 大
//
// 1. 哈希表 + 排序：
//    - 统计频率：O(n)
//    - 按频率排序：O(m log m)，m 是不同元素数量
//    - 取前 K 个
//
// 2. 哈希表 + 小顶堆：
//    - 维护大小为 K 的小顶堆
//    - 时间 O(n log k)
//
// 3. 哈希表 + 桶排序（最优）：
//    - 频率范围是 [1, n]
//    - 用桶存储：bucket[freq] = [元素列表]
//    - 从高频到低频遍历桶，取前 K 个
//    - 时间 O(n)

// ============================================================
// 解法一：哈希表 + 排序
// ============================================================
// ⏱️ 时间复杂度：O(n log n) | 空间复杂度：O(n)

function topKFrequent_v1(nums: number[], k: number): number[] {
  // 统计频率
  const freqMap = new Map<number, number>();
  for (const num of nums) {
    freqMap.set(num, (freqMap.get(num) || 0) + 1);
  }

  // 按频率排序
  const sorted = [...freqMap.entries()].sort((a, b) => b[1] - a[1]);

  // 取前 K 个
  return sorted.slice(0, k).map(([num]) => num);
}

// ============================================================
// 解法二：桶排序（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)

/**
 * 📊 桶排序图解：
 *
 * nums = [1,1,1,2,2,3], k = 2
 *
 * Step 1: 统计频率
 *         freqMap = {1: 3, 2: 2, 3: 1}
 *
 * Step 2: 创建桶，bucket[freq] = [元素列表]
 *         bucket[1] = [3]      // 频率为 1 的元素
 *         bucket[2] = [2]      // 频率为 2 的元素
 *         bucket[3] = [1]      // 频率为 3 的元素
 *
 * Step 3: 从高频到低频遍历桶
 *         freq=6: 空
 *         freq=5: 空
 *         freq=4: 空
 *         freq=3: [1] → 结果加入 1
 *         freq=2: [2] → 结果加入 2
 *         已经有 k=2 个元素，返回 [1, 2]
 */
function topKFrequent_v2(nums: number[], k: number): number[] {
  // 统计频率
  const freqMap = new Map<number, number>();
  for (const num of nums) {
    freqMap.set(num, (freqMap.get(num) || 0) + 1);
  }

  // 创建桶：bucket[freq] = [元素列表]
  // 频率最大为 nums.length
  const bucket: number[][] = new Array(nums.length + 1).fill(null).map(() => []);

  for (const [num, freq] of freqMap) {
    bucket[freq].push(num);
  }

  // 从高频到低频遍历桶，取前 K 个
  const result: number[] = [];

  for (let freq = bucket.length - 1; freq >= 0 && result.length < k; freq--) {
    if (bucket[freq].length > 0) {
      result.push(...bucket[freq]);
    }
  }

  return result.slice(0, k);
}

// ============================================================
// 解法三：快速选择（类似快排）
// ============================================================
// ⏱️ 时间复杂度：平均 O(n)，最坏 O(n²) | 空间复杂度：O(n)
// 📝 基于快排的 partition 思想

function topKFrequent_v3(nums: number[], k: number): number[] {
  // 统计频率
  const freqMap = new Map<number, number>();
  for (const num of nums) {
    freqMap.set(num, (freqMap.get(num) || 0) + 1);
  }

  const unique = [...freqMap.keys()];

  // 快速选择：找到第 k 大的元素
  function quickSelect(left: number, right: number, targetIndex: number): void {
    if (left >= right) return;

    const pivotIndex = partition(left, right);

    if (pivotIndex === targetIndex) {
      return;
    } else if (pivotIndex < targetIndex) {
      quickSelect(pivotIndex + 1, right, targetIndex);
    } else {
      quickSelect(left, pivotIndex - 1, targetIndex);
    }
  }

  function partition(left: number, right: number): number {
    const pivotFreq = freqMap.get(unique[right])!;
    let i = left;

    for (let j = left; j < right; j++) {
      // 按频率降序排列
      if (freqMap.get(unique[j])! > pivotFreq) {
        [unique[i], unique[j]] = [unique[j], unique[i]];
        i++;
      }
    }

    [unique[i], unique[right]] = [unique[right], unique[i]];
    return i;
  }

  quickSelect(0, unique.length - 1, k - 1);

  return unique.slice(0, k);
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法         | 时间           | 空间  | 特点                     |
 * |-------------|----------------|-------|-------------------------|
 * | 排序        | O(n log n)     | O(n)  | 简单直观                 |
 * | 桶排序      | O(n)           | O(n)  | 推荐，最优               |
 * | 快速选择    | 平均 O(n)       | O(n)  | 不稳定                   |
 * | 小顶堆      | O(n log k)     | O(k)  | k 很小时效率高           |
 *
 * 面试推荐：桶排序（O(n) 时间，思路清晰）
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 桶的大小：
 *    - 频率范围是 [1, n]
 *    - 桶大小应该是 n+1
 *
 * 2. 桶可能有多个元素：
 *    - 多个元素可能有相同频率
 *
 * 3. result 可能超过 k 个：
 *    - 最后需要 slice(0, k)
 *
 * 4. 快速选择的边界处理
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 前 K 个高频单词 → 同样思路 + 字典序排序
 * - 数组中的第 K 个最大元素 → 快速选择 / 堆
 * - 最小的 K 个数 → 快速选择 / 堆
 *
 * 共同模式：哈希计数 + 排序/选择
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 热门搜索：统计搜索词频率，展示 Top K
 * 2. 用户行为：找出最常访问的页面
 * 3. 错误分析：找出出现最多的错误类型
 * 4. 标签统计：找出最热门的标签
 */

// 导出主解法
export { topKFrequent_v1, topKFrequent_v2, topKFrequent_v3 };
export default topKFrequent_v2;

