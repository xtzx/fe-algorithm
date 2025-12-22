/**
 * 📝 题目：字符串相加
 * 🔗 链接：https://leetcode.cn/problems/add-strings/
 * 🏷️ 难度：Easy（但归类到 Medium 因为是面试高频且有变体）
 * 🏷️ 标签：数学、字符串、模拟
 *
 * 📋 题目描述：
 * 给定两个字符串形式的非负整数 num1 和 num2，计算它们的和并同样以字符串形式返回。
 * 你不能使用任何內建的用于处理大整数的库（比如 BigInteger），
 * 也不能直接将输入的字符串转换为整数形式。
 *
 * 示例：
 * 输入：num1 = "11", num2 = "123"
 * 输出："134"
 *
 * 输入：num1 = "456", num2 = "77"
 * 输出："533"
 *
 * 输入：num1 = "0", num2 = "0"
 * 输出："0"
 */

// ============================================================
// 💡 思路分析
// ============================================================
//
// 模拟竖式加法：
// - 从右往左逐位相加
// - 处理进位
// - 两个数长度可能不同，需要处理
//
//       1 2 3
//   +     7 7
//   ─────────
//       2 0 0   (carry=1)
//       ↑
//       从右往左
//
// 关键点：
// 1. 用双指针从末尾开始
// 2. 用 carry 记录进位
// 3. 结果需要反转（或从头插入）

// ============================================================
// 解法一：双指针模拟（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(max(m, n)) | 空间复杂度：O(max(m, n))

/**
 * 📊 执行过程图解：
 *
 * num1 = "456", num2 = "77"
 *
 *     4  5  6
 * +      7  7
 * ───────────
 *
 * i=2, j=1: 6+7=13, carry=1, result="3"
 * i=1, j=0: 5+7+1=13, carry=1, result="33"
 * i=0, j=-1: 4+0+1=5, carry=0, result="533"
 *
 * 反转得 "533"
 */
function addStrings_v1(num1: string, num2: string): string {
  const result: string[] = [];
  let carry = 0;
  let i = num1.length - 1;
  let j = num2.length - 1;

  // 当还有数字或进位时继续
  while (i >= 0 || j >= 0 || carry > 0) {
    const n1 = i >= 0 ? parseInt(num1[i]) : 0;
    const n2 = j >= 0 ? parseInt(num2[j]) : 0;

    const sum = n1 + n2 + carry;
    result.push(String(sum % 10));
    carry = Math.floor(sum / 10);

    i--;
    j--;
  }

  return result.reverse().join('');
}

// ============================================================
// 解法二：使用 charCodeAt（更快）
// ============================================================
// ⏱️ 时间复杂度：O(max(m, n)) | 空间复杂度：O(max(m, n))
// 📝 避免 parseInt，直接用 ASCII 码计算

function addStrings_v2(num1: string, num2: string): string {
  const result: string[] = [];
  let carry = 0;
  let i = num1.length - 1;
  let j = num2.length - 1;
  const zeroCode = '0'.charCodeAt(0);

  while (i >= 0 || j >= 0 || carry > 0) {
    const n1 = i >= 0 ? num1.charCodeAt(i) - zeroCode : 0;
    const n2 = j >= 0 ? num2.charCodeAt(j) - zeroCode : 0;

    const sum = n1 + n2 + carry;
    result.push(String.fromCharCode((sum % 10) + zeroCode));
    carry = Math.floor(sum / 10);

    i--;
    j--;
  }

  return result.reverse().join('');
}

// ============================================================
// 📊 变体：字符串相减（面试可能追问）
// ============================================================

/**
 * 字符串相减（假设 num1 >= num2，且都是非负整数）
 */
function subtractStrings(num1: string, num2: string): string {
  const result: string[] = [];
  let borrow = 0; // 借位
  let i = num1.length - 1;
  let j = num2.length - 1;

  while (i >= 0) {
    const n1 = parseInt(num1[i]);
    const n2 = j >= 0 ? parseInt(num2[j]) : 0;

    let diff = n1 - n2 - borrow;

    if (diff < 0) {
      diff += 10;
      borrow = 1;
    } else {
      borrow = 0;
    }

    result.push(String(diff));
    i--;
    j--;
  }

  // 去除前导零
  while (result.length > 1 && result[result.length - 1] === '0') {
    result.pop();
  }

  return result.reverse().join('');
}

// ============================================================
// 📊 变体：二进制字符串相加（LeetCode 67）
// ============================================================

/**
 * 二进制加法
 */
function addBinary(a: string, b: string): string {
  const result: string[] = [];
  let carry = 0;
  let i = a.length - 1;
  let j = b.length - 1;

  while (i >= 0 || j >= 0 || carry > 0) {
    const n1 = i >= 0 ? parseInt(a[i]) : 0;
    const n2 = j >= 0 ? parseInt(b[j]) : 0;

    const sum = n1 + n2 + carry;
    result.push(String(sum % 2)); // 二进制是模 2
    carry = Math.floor(sum / 2); // 进位是除以 2

    i--;
    j--;
  }

  return result.reverse().join('');
}

// ============================================================
// 📊 变体：36 进制加法（字节跳动面试题）
// ============================================================

/**
 * 36 进制加法
 * 0-9 表示 0-9，a-z 表示 10-35
 */
function addBase36(num1: string, num2: string): string {
  const result: string[] = [];
  let carry = 0;
  let i = num1.length - 1;
  let j = num2.length - 1;

  // 字符转数字
  const charToNum = (c: string): number => {
    if (c >= '0' && c <= '9') return parseInt(c);
    return c.charCodeAt(0) - 'a'.charCodeAt(0) + 10;
  };

  // 数字转字符
  const numToChar = (n: number): string => {
    if (n < 10) return String(n);
    return String.fromCharCode('a'.charCodeAt(0) + n - 10);
  };

  while (i >= 0 || j >= 0 || carry > 0) {
    const n1 = i >= 0 ? charToNum(num1[i].toLowerCase()) : 0;
    const n2 = j >= 0 ? charToNum(num2[j].toLowerCase()) : 0;

    const sum = n1 + n2 + carry;
    result.push(numToChar(sum % 36));
    carry = Math.floor(sum / 36);

    i--;
    j--;
  }

  return result.reverse().join('');
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法        | 时间         | 空间         | 特点              |
 * |------------|--------------|--------------|------------------|
 * | parseInt   | O(max(m,n))  | O(max(m,n))  | 简单直观          |
 * | charCode   | O(max(m,n))  | O(max(m,n))  | 更快，无需解析     |
 *
 * 变体题目：
 * - 二进制加法：模 2，进位除以 2
 * - 36 进制加法：字节跳动高频面试题
 * - 字符串相减：处理借位
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 最后的进位：
 *    如 "99" + "1" = "100"，最后 carry=1 需要处理
 *    循环条件要包含 carry > 0
 *
 * 2. 两个数长度不同：
 *    较短的数要补 0
 *
 * 3. 结果需要反转：
 *    从低位算起，最后要反转
 *
 * 4. 前导零：
 *    输入保证无前导零，但相减时可能产生，需要去除
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 字符串相乘 → 模拟竖式乘法
 * - 二进制求和 → 进位是除以 2
 * - 两数相加（链表）→ 本质相同
 * - 加一 → 特殊的加法
 *
 * 共同模式：逐位计算 + 进位处理
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 大数计算：金融系统中超出 Number 范围的计算
 * 2. 精确计算：避免浮点数精度问题
 * 3. 进制转换：各种进制间的转换和计算
 * 4. 版本号比较：虽然不是加法，但类似的逐位比较
 */

export { addStrings_v1, addStrings_v2, subtractStrings, addBinary, addBase36 };
export default addStrings_v1;

