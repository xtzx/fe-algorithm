/**
 * 📝 题目：回文对
 * 🔗 链接：https://leetcode.cn/problems/palindrome-pairs/
 * 🏷️ 难度：Hard
 * 🏷️ 标签：数组、哈希表、字符串、字典树
 *
 * 📋 题目描述：
 * 给定一组 互不相同 的单词，找出所有 不同 的索引对 (i, j)，
 * 使得列表中的两个单词 words[i] + words[j] 可拼接成回文串。
 *
 * 示例：
 * 输入：words = ["abcd","dcba","lls","s","sssll"]
 * 输出：[[0,1],[1,0],[3,2],[2,4]]
 * 解释：
 * - words[0] + words[1] = "abcddcba" 是回文
 * - words[1] + words[0] = "dcbaabcd" 是回文
 * - words[3] + words[2] = "slls" 是回文
 * - words[2] + words[4] = "llssssll" 是回文
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 暴力：枚举所有 (i, j) 对，检查是否回文 → O(n² × k)，太慢
//
// 关键观察：两个词能组成回文的情况：
//
// 1. 完全互逆：word1 = reverse(word2)
//    例：abc + cba = abccba
//
// 2. word1 = prefix + 回文后缀，reverse(prefix) 存在
//    例：lls = ll + s，ll 是回文，s 的逆序 s 存在
//    所以 lls + s = llss 不是回文
//    但 s + lls = slls 是回文！
//
// 3. word1 = 回文前缀 + suffix，reverse(suffix) 存在
//    例：sssll = sss + ll，sss 是回文，ll 的逆序 ll 存在
//    所以 sssll + ll 不行
//    但 lls + sssll = llssssll 是回文！
//
// 优化思路：
// - 把所有单词的逆序存入哈希表
// - 对于每个单词，检查它的所有前缀和后缀

// ============================================================
// 解法：哈希表 + 前后缀枚举
// ============================================================
// ⏱️ 时间复杂度：O(n × k²) | 空间复杂度：O(n × k)

/**
 * 📊 核心思路图解：
 *
 * words = ["abcd", "dcba", "lls", "s", "sssll"]
 *
 * Step 1: 建立逆序哈希表
 *   reverseMap = {
 *     "dcba": 0,  // abcd 的逆序
 *     "abcd": 1,  // dcba 的逆序
 *     "sll": 2,   // lls 的逆序
 *     "s": 3,     // s 的逆序
 *     "llsss": 4  // sssll 的逆序
 *   }
 *
 * Step 2: 对于每个单词，枚举分割点
 *
 *   word = "lls", index = 2
 *
 *   分割成 prefix + suffix:
 *   "" + "lls"     → suffix "lls" 是回文？否
 *   "l" + "ls"     → suffix "ls" 是回文？否
 *   "ll" + "s"     → suffix "s" 是回文？是！
 *                    → 找 prefix "ll" 的逆序 "ll"
 *                    → 不存在
 *   "lls" + ""     → suffix "" 是回文？是！
 *                    → 找 prefix "lls" 的逆序 "sll"
 *                    → 存在，index=2，但是自己，跳过
 *
 *   分割成 prefix + suffix（前缀是回文）:
 *   "" + "lls"     → prefix "" 是回文？是！
 *                    → 找 suffix "lls" 的逆序 "sll"
 *                    → 存在，index=2，但是自己，跳过
 *   "l" + "ls"     → prefix "l" 是回文？是！
 *                    → 找 suffix "ls" 的逆序 "sl"
 *                    → 不存在
 *   "ll" + "s"     → prefix "ll" 是回文？是！
 *                    → 找 suffix "s" 的逆序 "s"
 *                    → 存在，index=3
 *                    → 结果：(3, 2) → "s" + "lls" = "slls" ✓
 */
function palindromePairs(words: string[]): number[][] {
  const result: number[][] = [];
  const wordMap = new Map<string, number>();

  // 建立单词到索引的映射
  for (let i = 0; i < words.length; i++) {
    wordMap.set(words[i], i);
  }

  // 判断是否是回文
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

  // 反转字符串
  function reverse(s: string): string {
    return s.split('').reverse().join('');
  }

  for (let i = 0; i < words.length; i++) {
    const word = words[i];

    // 枚举所有分割点
    for (let j = 0; j <= word.length; j++) {
      const prefix = word.substring(0, j);
      const suffix = word.substring(j);

      // 情况1: 后缀是回文，找前缀的逆序
      // word = prefix + suffix(回文)
      // 如果 reverse(prefix) 存在，则 reverse(prefix) + word 是回文
      if (isPalindrome(suffix)) {
        const reversedPrefix = reverse(prefix);
        if (wordMap.has(reversedPrefix)) {
          const k = wordMap.get(reversedPrefix)!;
          // 避免自己和自己配对，且避免重复（当 suffix 为空时）
          if (k !== i) {
            result.push([i, k]);
          }
        }
      }

      // 情况2: 前缀是回文，找后缀的逆序
      // word = prefix(回文) + suffix
      // 如果 reverse(suffix) 存在，则 word + reverse(suffix) 是回文
      // 注意：当 j === 0 时 prefix 为空，会和情况1重复，需要排除
      if (j > 0 && isPalindrome(prefix)) {
        const reversedSuffix = reverse(suffix);
        if (wordMap.has(reversedSuffix)) {
          const k = wordMap.get(reversedSuffix)!;
          if (k !== i) {
            result.push([k, i]);
          }
        }
      }
    }
  }

  return result;
}

// ============================================================
// 🔄 复杂度分析
// ============================================================
/**
 * 时间复杂度：O(n × k²)
 * - n 个单词
 * - 每个单词枚举 k 个分割点
 * - 每次分割需要 O(k) 判断回文和反转
 *
 * 空间复杂度：O(n × k)
 * - 哈希表存储 n 个单词
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 避免自己和自己配对：
 *    - 当 word 本身是回文时，reverse(word) = word
 *    - 需要检查 k !== i
 *
 * 2. 避免重复：
 *    - 情况1 和 情况2 在 j=0 时可能重复
 *    - 情况2 需要 j > 0 的条件
 *
 * 3. 空字符串：
 *    - "" 可以和任何回文词配对
 *    - 需要特殊处理
 *
 * 4. 顺序：
 *    - 情况1: [i, k]（word 在前）
 *    - 情况2: [k, i]（word 在后）
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 最长回文子串 → 中心扩展 / DP
 * - 验证回文串 → 双指针
 * - 最短回文串 → KMP / 哈希
 *
 * 共同模式：回文判断 + 字符串匹配
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 密码校验：检测可组合成回文的密码组合
 * 2. 文本游戏：找出能组成回文的词对
 * 3. 数据验证：检测特定格式的数据组合
 */

// 导出主解法
export { palindromePairs };
export default palindromePairs;

