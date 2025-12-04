/**
 * 📝 题目：零钱兑换
 * 🔗 链接：https://leetcode.cn/problems/coin-change/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：广度优先搜索、数组、动态规划
 *
 * 📋 题目描述：
 * 给你一个整数数组 coins，表示不同面额的硬币；以及一个整数 amount，表示总金额。
 *
 * 计算并返回可以凑成总金额所需的 最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
 *
 * 你可以认为每种硬币的数量是无限的。
 *
 * 示例：
 * 输入：coins = [1, 2, 5], amount = 11
 * 输出：3
 * 解释：11 = 5 + 5 + 1
 *
 * 输入：coins = [2], amount = 3
 * 输出：-1
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 典型的完全背包问题！
// - 物品：硬币（可以无限使用）
// - 背包容量：amount
// - 求：最少物品数
//
// dp[i] = 凑成金额 i 的最少硬币数
//
// 状态转移：
// 对于每种硬币 coin：
//   dp[i] = min(dp[i], dp[i-coin] + 1)

// ============================================================
// 解法一：完全背包
// ============================================================
// ⏱️ 时间复杂度：O(amount × n) | 空间复杂度：O(amount)

/**
 * 📊 执行过程图解：
 *
 * coins = [1, 2, 5], amount = 11
 *
 * 初始: dp = [0, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]
 *
 * 处理 coin=1:
 * dp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 *
 * 处理 coin=2:
 * dp = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
 *
 * 处理 coin=5:
 * dp = [0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 3]
 *
 * 答案：dp[11] = 3
 */
function coinChange(coins: number[], amount: number): number {
  // dp[i] = 凑成金额 i 的最少硬币数
  const dp: number[] = Array(amount + 1).fill(Infinity);
  dp[0] = 0; // 凑成 0 不需要硬币

  for (const coin of coins) {
    // 完全背包：从前往后遍历（可以重复使用）
    for (let i = coin; i <= amount; i++) {
      dp[i] = Math.min(dp[i], dp[i - coin] + 1);
    }
  }

  return dp[amount] === Infinity ? -1 : dp[amount];
}

// ============================================================
// 解法二：另一种遍历顺序
// ============================================================
// 先遍历金额，再遍历硬币
function coinChange_v2(coins: number[], amount: number): number {
  const dp: number[] = Array(amount + 1).fill(Infinity);
  dp[0] = 0;

  for (let i = 1; i <= amount; i++) {
    for (const coin of coins) {
      if (coin <= i && dp[i - coin] !== Infinity) {
        dp[i] = Math.min(dp[i], dp[i - coin] + 1);
      }
    }
  }

  return dp[amount] === Infinity ? -1 : dp[amount];
}

// ============================================================
// 解法三：记忆化递归
// ============================================================
function coinChange_memo(coins: number[], amount: number): number {
  const memo: Map<number, number> = new Map();

  function dfs(remaining: number): number {
    if (remaining < 0) return -1;
    if (remaining === 0) return 0;
    if (memo.has(remaining)) return memo.get(remaining)!;

    let minCoins = Infinity;
    for (const coin of coins) {
      const result = dfs(remaining - coin);
      if (result >= 0) {
        minCoins = Math.min(minCoins, result + 1);
      }
    }

    const answer = minCoins === Infinity ? -1 : minCoins;
    memo.set(remaining, answer);
    return answer;
  }

  return dfs(amount);
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法         | 时间           | 空间      | 特点          |
 * |--------------|----------------|-----------|---------------|
 * | 完全背包     | O(amount × n)  | O(amount) | 推荐          |
 * | 记忆化递归   | O(amount × n)  | O(amount) | 好理解        |
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 初始值：
 *    - dp[0] = 0（凑成 0 不需要硬币）
 *    - 其他初始化为 Infinity（表示不可达）
 *
 * 2. 完全背包的遍历顺序：
 *    - 从前往后（可以重复使用同一硬币）
 *    - 与 0-1 背包相反
 *
 * 3. 返回值：
 *    - 如果 dp[amount] === Infinity，返回 -1
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 零钱兑换 II → 求方案数
 * - 完全平方数 → 类似思路
 * - 组合总和 IV → 求排列数
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 购物找零：最少纸币/硬币
 * 2. 资源分配：最少资源单位组合
 */

export { coinChange, coinChange_v2, coinChange_memo };
export default coinChange;

