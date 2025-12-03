/**
 * 📝 题目：无重复字符的最长子串
 * 🔗 链接：https://leetcode.cn/problems/longest-substring-without-repeating-characters/
 * 🏷️ 难度：Medium
 * 🏷️ 标签：哈希表、字符串、滑动窗口
 *
 * 📋 题目描述：
 * 给定一个字符串 s，请你找出其中不含有重复字符的 最长子串 的长度。
 *
 * 示例：
 * 输入：s = "abcabcbb"
 * 输出：3
 * 解释：最长子串是 "abc"，长度为 3
 *
 * 输入：s = "bbbbb"
 * 输出：1
 * 解释：最长子串是 "b"，长度为 1
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 关键词：「连续子串」「最长」「不重复」→ 滑动窗口！
//
// 1. 暴力思路：枚举所有子串，检查是否有重复 → O(n³)
//
// 2. 优化思考：
//    - 用滑动窗口维护一个"无重复字符"的区间
//    - 右边界扩张：加入新字符
//    - 左边界收缩：当出现重复时，收缩直到无重复
//
// 3. 如何判断重复？
//    - 用 Set：记录窗口内的字符
//    - 用 Map：记录字符的出现次数或位置
//
// 📊 滑动窗口示意图：
//
//    s = "abcabcbb"
//
//    [a] b c a b c b b      窗口={a}, 长度=1
//     ↑
//    L,R
//
//    [a  b] c a b c b b     窗口={a,b}, 长度=2
//     ↑  ↑
//     L  R
//
//    [a  b  c] a b c b b    窗口={a,b,c}, 长度=3 ✓
//     ↑     ↑
//     L     R
//
//    [a  b  c  a] b c b b   发现重复'a'!
//     ↑        ↑
//     L        R
//
//     a [b  c  a] b c b b   收缩左边界，移除'a'
//        ↑     ↑
//        L     R

// ============================================================
// 解法一：滑动窗口 + Set（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(min(n, m))，m 是字符集大小

/**
 * 📊 执行过程图解：
 *
 * s = "abcabcbb"
 *
 *   right=0: 'a' 不在窗口中, 加入, 窗口="a", maxLen=1
 *   right=1: 'b' 不在窗口中, 加入, 窗口="ab", maxLen=2
 *   right=2: 'c' 不在窗口中, 加入, 窗口="abc", maxLen=3
 *   right=3: 'a' 在窗口中! 收缩: 移除'a', left=1, 窗口="bc"
 *            再加入'a', 窗口="bca", maxLen=3
 *   right=4: 'b' 在窗口中! 收缩: 移除'b', left=2, 窗口="ca"
 *            再加入'b', 窗口="cab", maxLen=3
 *   right=5: 'c' 在窗口中! 收缩: 移除'c', left=3, 窗口="ab"
 *            再加入'c', 窗口="abc", maxLen=3
 *   right=6: 'b' 在窗口中! 收缩: 移除'a','b', left=5, 窗口="c"
 *            再加入'b', 窗口="cb", maxLen=3
 *   right=7: 'b' 在窗口中! 收缩: 移除'c','b', left=7, 窗口=""
 *            再加入'b', 窗口="b", maxLen=3
 *
 * 最终：maxLen = 3
 */
function lengthOfLongestSubstring_v1(s: string): number {
  const window = new Set<string>();
  let left = 0;
  let maxLen = 0;

  for (let right = 0; right < s.length; right++) {
    const char = s[right];

    // 如果字符已在窗口中，收缩左边界直到移除该字符
    while (window.has(char)) {
      window.delete(s[left]);
      left++;
    }

    // 将当前字符加入窗口
    window.add(char);

    // 更新最大长度
    maxLen = Math.max(maxLen, right - left + 1);
  }

  return maxLen;
}

// ============================================================
// 解法二：滑动窗口 + Map（记录字符位置，更优）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(min(n, m))
// ✅ 优化：直接跳到重复字符的下一个位置，不需要一个个移除

/**
 * 📊 执行过程图解：
 *
 * s = "abcabcbb"
 * map 记录每个字符最后出现的位置
 *
 *   right=0: 'a' at 0, map={a:0}, maxLen=1
 *   right=1: 'b' at 1, map={a:0,b:1}, maxLen=2
 *   right=2: 'c' at 2, map={a:0,b:1,c:2}, maxLen=3
 *   right=3: 'a' 重复! map['a']=0, left=max(0, 0+1)=1
 *            map={a:3,b:1,c:2}, maxLen=3
 *   right=4: 'b' 重复! map['b']=1, left=max(1, 1+1)=2
 *            map={a:3,b:4,c:2}, maxLen=3
 *   ...
 *
 * 关键优化：left = max(left, map[char] + 1)
 *          直接跳到重复字符的下一个位置
 */
function lengthOfLongestSubstring_v2(s: string): number {
  const map = new Map<string, number>(); // 记录字符最后出现的位置
  let left = 0;
  let maxLen = 0;

  for (let right = 0; right < s.length; right++) {
    const char = s[right];

    // 如果字符已存在，直接跳到其后面
    if (map.has(char)) {
      // 注意：要取 max，因为 map 中的位置可能在 left 之前
      left = Math.max(left, map.get(char)! + 1);
    }

    // 更新字符位置
    map.set(char, right);

    // 更新最大长度
    maxLen = Math.max(maxLen, right - left + 1);
  }

  return maxLen;
}

// ============================================================
// 解法三：滑动窗口 + 数组（ASCII 优化）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(128)
// 📝 如果字符集是 ASCII，可以用数组代替 Map
function lengthOfLongestSubstring_v3(s: string): number {
  const index = new Array(128).fill(-1); // ASCII 字符的位置
  let left = 0;
  let maxLen = 0;

  for (let right = 0; right < s.length; right++) {
    const charCode = s.charCodeAt(right);

    // 如果字符之前出现过，更新 left
    if (index[charCode] >= left) {
      left = index[charCode] + 1;
    }

    // 更新字符位置
    index[charCode] = right;

    // 更新最大长度
    maxLen = Math.max(maxLen, right - left + 1);
  }

  return maxLen;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法         | 时间  | 空间          | 特点                    |
 * |-------------|-------|---------------|------------------------|
 * | Set         | O(n)  | O(min(n,m))   | 简单直观                |
 * | Map 记录位置 | O(n)  | O(min(n,m))   | 优化，跳过无效收缩        |
 * | 数组        | O(n)  | O(128)        | ASCII 场景最优          |
 *
 * m 是字符集大小
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. Map 解法中 left 的更新：
 *    left = Math.max(left, map.get(char)! + 1)
 *    为什么要取 max？
 *    → 因为 map 中可能有 left 之前的旧数据
 *    → 例如 "abba"，处理到第二个 'a' 时，map['a']=0，但 left 已经是 2 了
 *
 * 2. 窗口长度计算：right - left + 1
 *    不是 right - left
 *
 * 3. 空字符串：返回 0
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 最小覆盖子串 → 滑动窗口（收缩时更新）
 * - 找到字符串中所有字母异位词 → 滑动窗口 + 哈希
 * - 长度最小的子数组 → 滑动窗口
 * - 替换后的最长重复字符 → 滑动窗口
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 密码校验：检查密码中是否有连续重复字符
 * 2. URL 去重：提取 URL 中不重复的路径段
 * 3. 输入法：找出最长的不重复输入序列
 * 4. 日志分析：找出最长的不重复操作序列
 */

// 导出主解法
export {
  lengthOfLongestSubstring_v1,
  lengthOfLongestSubstring_v2,
  lengthOfLongestSubstring_v3,
};
export default lengthOfLongestSubstring_v2;

