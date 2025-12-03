/**
 * 📝 题目：串联所有单词的子串
 * 🔗 链接：https://leetcode.cn/problems/substring-with-concatenation-of-all-words/
 * 🏷️ 难度：Hard
 * 🏷️ 标签：哈希表、字符串、滑动窗口
 *
 * 📋 题目描述：
 * 给定一个字符串 s 和一个字符串数组 words。words 中所有字符串 长度相同。
 * s 中的 串联子串 是指一个包含 words 中所有字符串以任意顺序排列连接起来的子串。
 * 返回所有串联子串在 s 中的开始索引。
 *
 * 示例：
 * 输入：s = "barfoothefoobarman", words = ["foo","bar"]
 * 输出：[0,9]
 * 解释：
 * - 从索引 0 开始的子串是 "barfoo"，是 ["bar","foo"] 的串联
 * - 从索引 9 开始的子串是 "foobar"，是 ["foo","bar"] 的串联
 *
 * 输入：s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
 * 输出：[]
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 关键观察：
// - 所有单词长度相同，设为 wordLen
// - 需要找的子串长度 = wordLen × words.length
//
// 1. 暴力思路：
//    - 枚举每个起始位置
//    - 检查是否是 words 的全排列
//    - 时间复杂度高
//
// 2. 滑动窗口优化：
//    - 窗口大小固定为 wordLen × words.length
//    - 每次移动 wordLen 个字符
//    - 用哈希表记录单词频率
//
// 关键：因为单词长度相同，我们可以把字符串按 wordLen 分组
//       起始位置可以是 0, 1, 2, ..., wordLen-1

// ============================================================
// 解法一：滑动窗口 + 哈希表（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n × wordLen) | 空间复杂度：O(m × wordLen)

/**
 * 📊 执行过程图解：
 *
 * s = "barfoothefoobarman", words = ["foo","bar"]
 * wordLen = 3, wordCount = 2, totalLen = 6
 *
 * 对于起始位置 offset = 0:
 *   把 s 按 wordLen=3 分割: ["bar", "foo", "the", "foo", "bar", "man"]
 *
 *   滑动窗口（每次移动一个单词）:
 *
 *   [bar foo] the foo bar man    window={bar:1,foo:1}, 匹配! index=0
 *    bar [foo the] foo bar man   window={foo:1,the:1}, 不匹配
 *    bar foo [the foo] bar man   window={the:1,foo:1}, 不匹配
 *    bar foo the [foo bar] man   window={foo:1,bar:1}, 匹配! index=9
 *    bar foo the foo [bar man]   window={bar:1,man:1}, 不匹配
 *
 * 对于起始位置 offset = 1, 2:
 *   类似处理...
 *
 * 结果: [0, 9]
 */
function findSubstring_v1(s: string, words: string[]): number[] {
  const result: number[] = [];
  if (words.length === 0 || s.length === 0) return result;

  const wordLen = words[0].length;
  const wordCount = words.length;
  const totalLen = wordLen * wordCount;

  if (s.length < totalLen) return result;

  // 统计 words 中每个单词的频率
  const need = new Map<string, number>();
  for (const word of words) {
    need.set(word, (need.get(word) || 0) + 1);
  }

  // 枚举起始偏移量 0 到 wordLen-1
  for (let offset = 0; offset < wordLen; offset++) {
    const window = new Map<string, number>();
    let left = offset;
    let right = offset;
    let valid = 0; // 满足条件的单词种类数

    while (right + wordLen <= s.length) {
      // 扩张窗口：加入一个单词
      const word = s.substring(right, right + wordLen);
      right += wordLen;

      if (need.has(word)) {
        window.set(word, (window.get(word) || 0) + 1);
        if (window.get(word) === need.get(word)) {
          valid++;
        }
      }

      // 当窗口大小达到 totalLen 时，判断是否匹配
      while (right - left >= totalLen) {
        // 检查是否匹配
        if (valid === need.size) {
          result.push(left);
        }

        // 收缩窗口：移除一个单词
        const leftWord = s.substring(left, left + wordLen);
        left += wordLen;

        if (need.has(leftWord)) {
          if (window.get(leftWord) === need.get(leftWord)) {
            valid--;
          }
          window.set(leftWord, window.get(leftWord)! - 1);
        }
      }
    }
  }

  return result;
}

// ============================================================
// 解法二：暴力检查每个位置
// ============================================================
// ⏱️ 时间复杂度：O(n × m × wordLen) | 空间复杂度：O(m × wordLen)
// 📝 思路简单，但效率较低

function findSubstring_v2(s: string, words: string[]): number[] {
  const result: number[] = [];
  if (words.length === 0 || s.length === 0) return result;

  const wordLen = words[0].length;
  const wordCount = words.length;
  const totalLen = wordLen * wordCount;

  if (s.length < totalLen) return result;

  // 统计 words 的频率
  const need = new Map<string, number>();
  for (const word of words) {
    need.set(word, (need.get(word) || 0) + 1);
  }

  // 检查从位置 i 开始的子串是否匹配
  for (let i = 0; i <= s.length - totalLen; i++) {
    const seen = new Map<string, number>();
    let j = 0;

    while (j < wordCount) {
      const word = s.substring(i + j * wordLen, i + (j + 1) * wordLen);

      if (!need.has(word)) break;

      seen.set(word, (seen.get(word) || 0) + 1);

      if (seen.get(word)! > need.get(word)!) break;

      j++;
    }

    if (j === wordCount) {
      result.push(i);
    }
  }

  return result;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法             | 时间               | 空间          | 特点              |
 * |-----------------|--------------------| --------------|------------------|
 * | 滑动窗口         | O(n × wordLen)     | O(m × wordLen) | 推荐，最优         |
 * | 暴力检查         | O(n × m × wordLen) | O(m × wordLen) | 简单，效率较低     |
 *
 * n = s.length, m = words.length
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 起始偏移量：
 *    - 需要枚举 0 到 wordLen-1 的起始位置
 *    - 否则可能漏掉解
 *
 * 2. valid 的更新：
 *    - 只有当 window[word] == need[word] 时才 valid++
 *    - 只有当 window[word] == need[word] 时才 valid--
 *
 * 3. 边界条件：
 *    - s.length < totalLen 直接返回空
 *    - right + wordLen <= s.length
 *
 * 4. words 可能有重复单词
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 找到字符串中所有字母异位词 → 滑动窗口（字符级别）
 * - 最小覆盖子串 → 滑动窗口
 * - 字符串的排列 → 滑动窗口
 *
 * 共同模式：滑动窗口 + 哈希计数
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 模板匹配：在文本中找特定模式的组合
 * 2. 日志分析：找出包含特定关键词组合的日志段
 * 3. 代码搜索：找出包含特定 token 组合的代码段
 * 4. 自然语言处理：短语检测
 */

// 导出主解法
export { findSubstring_v1, findSubstring_v2 };
export default findSubstring_v1;

