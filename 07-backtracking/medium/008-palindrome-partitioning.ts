/**
 * 📝 题目：分割回文串
 * 🔗 链接：https://leetcode.cn/problems/palindrome-partitioning/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：字符串、动态规划、回溯
 *
 * 📋 题目描述：
 * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串。
 * 返回 s 所有可能的分割方案。
 *
 * 示例：
 * 输入：s = "aab"
 * 输出：[["a","a","b"],["aa","b"]]
 *
 * 输入：s = "a"
 * 输出：[["a"]]
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 分割问题的本质：在每个位置决定是否"切一刀"
//
// 回溯思路：
// - 从位置 start 开始，尝试切出所有可能的子串
// - 如果子串是回文，递归处理剩余部分
// - 如果子串不是回文，跳过
//
// 优化：可以预处理回文判断（用 DP）

// ============================================================
// 解法一：回溯
// ============================================================
// ⏱️ 时间复杂度：O(n × 2^n) | 空间复杂度：O(n²)

/**
 * 📊 执行过程图解：
 *
 * s = "aab"
 *
 *                    ""
 *          /         |         \
 *      ["a"]      ["aa"]      ["aab"] ✗不是回文
 *       / \          |
 *   ["a","a"]  ["aa","b"] ✓
 *      |
 * ["a","a","b"] ✓
 *
 * 结果：[["a","a","b"], ["aa","b"]]
 */
function partition(s: string): string[][] {
  const result: string[][] = [];
  const path: string[] = [];

  function backtrack(start: number) {
    // 结束条件：已经处理完整个字符串
    if (start === s.length) {
      result.push([...path]);
      return;
    }

    // 尝试从 start 开始切出不同长度的子串
    for (let end = start; end < s.length; end++) {
      const substring = s.slice(start, end + 1);

      // 如果是回文，继续递归
      if (isPalindrome(substring)) {
        path.push(substring);
        backtrack(end + 1);
        path.pop();
      }
      // 不是回文，跳过（剪枝）
    }
  }

  backtrack(0);
  return result;
}

// 判断回文
function isPalindrome(s: string): boolean {
  let left = 0;
  let right = s.length - 1;
  while (left < right) {
    if (s[left] !== s[right]) return false;
    left++;
    right--;
  }
  return true;
}

// ============================================================
// 解法二：回溯 + DP 预处理回文
// ============================================================
// ⏱️ 时间复杂度：O(n × 2^n) | 空间复杂度：O(n²)

/**
 * 优化：预处理出所有子串是否为回文
 * dp[i][j] = true 表示 s[i..j] 是回文
 */
function partition_optimized(s: string): string[][] {
  const n = s.length;
  const result: string[][] = [];
  const path: string[] = [];

  // 预处理回文 DP
  // dp[i][j] = s[i..j] 是否是回文
  const dp: boolean[][] = Array.from({ length: n }, () => Array(n).fill(false));

  // 初始化
  for (let i = 0; i < n; i++) {
    dp[i][i] = true; // 单个字符
  }

  // 填表：从短到长
  for (let len = 2; len <= n; len++) {
    for (let i = 0; i <= n - len; i++) {
      const j = i + len - 1;
      if (len === 2) {
        dp[i][j] = s[i] === s[j];
      } else {
        dp[i][j] = s[i] === s[j] && dp[i + 1][j - 1];
      }
    }
  }

  function backtrack(start: number) {
    if (start === n) {
      result.push([...path]);
      return;
    }

    for (let end = start; end < n; end++) {
      // 用 DP 表直接判断回文，O(1)
      if (dp[start][end]) {
        path.push(s.slice(start, end + 1));
        backtrack(end + 1);
        path.pop();
      }
    }
  }

  backtrack(0);
  return result;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法       | 时间       | 空间    | 回文判断  |
 * |------------|------------|---------|-----------|
 * | 基础回溯   | O(n × 2^n) | O(n)    | O(n)      |
 * | DP 预处理  | O(n × 2^n) | O(n²)   | O(1)      |
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 子串范围：
 *    - s.slice(start, end+1)，注意 slice 是左闭右开
 *
 * 2. 递归起点：
 *    - backtrack(end + 1)，不是 backtrack(start + 1)
 *
 * 3. DP 预处理的填表顺序：
 *    - 从短到长，因为长的依赖短的
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 分割回文串 II → 最少切割次数（DP）
 * - 复原 IP 地址 → 分割 + 验证
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 文本编辑：自动分词
 * 2. URL 路由：路径分割和验证
 */

export { partition, partition_optimized };
export default partition;

