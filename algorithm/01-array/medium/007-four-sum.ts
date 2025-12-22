/**
 * 📝 题目：四数之和
 * 🔗 链接：https://leetcode.cn/problems/4sum/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：数组、双指针、排序
 *
 * 📋 题目描述：
 * 给你一个由 n 个整数组成的数组 nums，和一个目标值 target。
 * 请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]]：
 * - 0 <= a, b, c, d < n
 * - a、b、c 和 d 互不相同
 * - nums[a] + nums[b] + nums[c] + nums[d] == target
 *
 * 你可以按任意顺序返回答案。
 *
 * 示例：
 * 输入：nums = [1,0,-1,0,-2,2], target = 0
 * 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
 */

// ============================================================
// 💡 思路分析
// ============================================================
//
// 这道题是三数之和的扩展：
// - 三数之和：固定 1 个数，两数之和用双指针 → O(n²)
// - 四数之和：固定 2 个数，两数之和用双指针 → O(n³)
//
// 核心套路：
// 1. 排序
// 2. 双重循环固定前两个数
// 3. 对撞指针找后两个数
// 4. 多处去重（四处！）
//
// 剪枝优化：
// - 当前最小四数和 > target，后面不可能有解
// - 当前最大四数和 < target，当前这轮不可能有解

// ============================================================
// 解法：排序 + 双指针
// ============================================================
// ⏱️ 时间复杂度：O(n³) | 空间复杂度：O(log n) 排序所需

/**
 * 📊 执行过程图解：
 *
 * nums = [1, 0, -1, 0, -2, 2], target = 0
 *
 * Step 1: 排序
 *         [-2, -1, 0, 0, 1, 2]
 *
 * Step 2: 双重循环固定 i, j
 *
 * i = 0 (nums[i] = -2):
 *   j = 1 (nums[j] = -1):
 *     left = 2, right = 5
 *     target for two sum = 0 - (-2) - (-1) = 3
 *
 *     [-2, -1, 0, 0, 1, 2]
 *       ↑   ↑  ↑        ↑
 *       i   j left    right
 *
 *     sum = 0 + 2 = 2 < 3, left++
 *     sum = 0 + 2 = 2 < 3, left++
 *     sum = 1 + 2 = 3 = target ✓ 找到 [-2, -1, 1, 2]
 *
 *   j = 2 (nums[j] = 0):
 *     target for two sum = 0 - (-2) - 0 = 2
 *     sum = 0 + 2 = 2 ✓ 找到 [-2, 0, 0, 2]
 *
 * i = 1 (nums[i] = -1):
 *   j = 2 (nums[j] = 0):
 *     target for two sum = 0 - (-1) - 0 = 1
 *     sum = 0 + 2 = 2 > 1, right--
 *     sum = 0 + 1 = 1 ✓ 找到 [-1, 0, 0, 1]
 *
 * 结果: [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
 */
function fourSum(nums: number[], target: number): number[][] {
  const result: number[][] = [];
  const n = nums.length;

  if (n < 4) return result;

  // 1. 排序
  nums.sort((a, b) => a - b);

  // 2. 第一层循环：固定第一个数
  for (let i = 0; i < n - 3; i++) {
    // 去重：跳过重复的第一个数
    if (i > 0 && nums[i] === nums[i - 1]) continue;

    // 剪枝1：当前最小四数和 > target，后面不可能有解
    if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;

    // 剪枝2：当前最大四数和 < target，这一轮不可能有解，跳过
    if (nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target) continue;

    // 3. 第二层循环：固定第二个数
    for (let j = i + 1; j < n - 2; j++) {
      // 去重：跳过重复的第二个数
      if (j > i + 1 && nums[j] === nums[j - 1]) continue;

      // 剪枝3：当前最小四数和 > target
      if (nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) break;

      // 剪枝4：当前最大四数和 < target
      if (nums[i] + nums[j] + nums[n - 2] + nums[n - 1] < target) continue;

      // 4. 双指针找剩下两个数
      let left = j + 1;
      let right = n - 1;
      const twoSumTarget = target - nums[i] - nums[j];

      while (left < right) {
        const sum = nums[left] + nums[right];

        if (sum === twoSumTarget) {
          result.push([nums[i], nums[j], nums[left], nums[right]]);

          // 去重：跳过重复的 left
          while (left < right && nums[left] === nums[left + 1]) left++;
          // 去重：跳过重复的 right
          while (left < right && nums[right] === nums[right - 1]) right--;

          left++;
          right--;
        } else if (sum < twoSumTarget) {
          left++;
        } else {
          right--;
        }
      }
    }
  }

  return result;
}

// ============================================================
// 📊 通用 N 数之和模板
// ============================================================

/**
 * 通用 N 数之和解法
 *
 * 思路：递归降维
 * - N 数之和 = 固定 1 个数 + (N-1) 数之和
 * - 直到 N = 2，用双指针解决
 */
function nSum(nums: number[], target: number, n: number, start: number): number[][] {
  const result: number[][] = [];
  const len = nums.length;

  // 边界条件
  if (n < 2 || len < n) return result;

  // 基础情况：两数之和
  if (n === 2) {
    let left = start;
    let right = len - 1;

    while (left < right) {
      const sum = nums[left] + nums[right];
      const leftVal = nums[left];
      const rightVal = nums[right];

      if (sum === target) {
        result.push([leftVal, rightVal]);
        // 去重
        while (left < right && nums[left] === leftVal) left++;
        while (left < right && nums[right] === rightVal) right--;
      } else if (sum < target) {
        while (left < right && nums[left] === leftVal) left++;
      } else {
        while (left < right && nums[right] === rightVal) right--;
      }
    }
  } else {
    // 递归情况：固定一个数，递归求 (n-1) 数之和
    for (let i = start; i < len - n + 1; i++) {
      // 去重
      if (i > start && nums[i] === nums[i - 1]) continue;

      // 剪枝：最小和 > target
      let minSum = 0;
      for (let j = 0; j < n; j++) {
        minSum += nums[i + j];
      }
      if (minSum > target) break;

      // 剪枝：最大和 < target
      let maxSum = nums[i];
      for (let j = 1; j < n; j++) {
        maxSum += nums[len - j];
      }
      if (maxSum < target) continue;

      // 递归
      const subResult = nSum(nums, target - nums[i], n - 1, i + 1);
      for (const sub of subResult) {
        result.push([nums[i], ...sub]);
      }
    }
  }

  return result;
}

// 使用通用模板
function fourSumGeneric(nums: number[], target: number): number[][] {
  nums.sort((a, b) => a - b);
  return nSum(nums, target, 4, 0);
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法      | 时间    | 空间      | 特点                    |
 * |----------|---------|----------|-------------------------|
 * | 双循环    | O(n³)   | O(log n) | 直接，容易理解            |
 * | 通用模板  | O(n³)   | O(n)     | 可扩展到任意 N 数之和     |
 *
 * 面试建议：
 * 1. 先写出直接的四数之和解法
 * 2. 如果面试官问"能不能扩展"，再展示通用模板
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 去重有四处！
 *    - i：第一个数
 *    - j：第二个数
 *    - left：第三个数
 *    - right：第四个数
 *
 * 2. 去重的条件：
 *    - i: i > 0 && nums[i] === nums[i-1]
 *    - j: j > i+1 && nums[j] === nums[j-1]
 *    注意 j 的条件是 j > i+1，不是 j > 0
 *
 * 3. 剪枝很重要：
 *    虽然不加剪枝也能通过，但面试时展示剪枝思想是加分项
 *
 * 4. 整数溢出（某些语言需要注意）：
 *    四个大数相加可能溢出 int32
 *    TypeScript/JavaScript 用 number 类型通常没这个问题
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 两数之和 → 哈希表 O(n)
 * - 三数之和 → 排序 + 双指针 O(n²)
 * - 四数之和 → 排序 + 双重循环 + 双指针 O(n³)
 * - 最接近的三数之和 → 排序 + 双指针
 * - 四数之和 II → 哈希表分组 O(n²)（不同，因为是四个数组）
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 多维度筛选：找出满足多个条件组合的数据
 * 2. 预算分配：在多个选项中找到刚好用完预算的组合
 * 3. 资源调度：多个资源组合满足特定需求
 */

export { fourSum, fourSumGeneric, nSum };
export default fourSum;

