/**
 * ============================================================
 * 📚 哈希表与字符串 - 前端业务场景代码示例
 * ============================================================
 *
 * 本文件展示哈希表和字符串处理在前端实际业务中的应用
 */

// ============================================================
// 1. 哈希表 - 请求缓存（带 LRU 淘汰）
// ============================================================

/**
 * 📝 业务场景：API 请求缓存
 *
 * 场景描述：
 * - 缓存接口返回数据，避免重复请求
 * - 设置最大缓存数量，超出时淘汰最久未使用的
 * - 设置过期时间
 */
class RequestCache<T> {
  private cache = new Map<string, { data: T; expireAt: number }>();
  private maxSize: number;
  private defaultTTL: number;

  constructor(maxSize = 100, defaultTTLMs = 5 * 60 * 1000) {
    this.maxSize = maxSize;
    this.defaultTTL = defaultTTLMs;
  }

  /**
   * 获取缓存
   * Map 保持插入顺序，利用这个特性实现 LRU
   */
  get(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) return null;

    // 检查是否过期
    if (Date.now() > entry.expireAt) {
      this.cache.delete(key);
      return null;
    }

    // LRU：重新插入以更新顺序
    this.cache.delete(key);
    this.cache.set(key, entry);

    return entry.data;
  }

  /**
   * 设置缓存
   */
  set(key: string, data: T, ttlMs?: number): void {
    // 如果已存在，先删除（确保顺序在最后）
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // 如果超出容量，删除最老的（Map 的第一个）
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }

    this.cache.set(key, {
      data,
      expireAt: Date.now() + (ttlMs ?? this.defaultTTL),
    });
  }

  /**
   * 清除缓存
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * 获取缓存大小
   */
  get size(): number {
    return this.cache.size;
  }
}

// 使用示例
const apiCache = new RequestCache<unknown>(50, 60000);

async function fetchWithCache<T>(url: string): Promise<T> {
  // 先查缓存
  const cached = apiCache.get(url);
  if (cached) {
    return cached as T;
  }

  // 无缓存则请求
  const response = await fetch(url);
  const data = await response.json();

  // 存入缓存
  apiCache.set(url, data);

  return data;
}

// ============================================================
// 2. 哈希表 - 表单字段依赖管理
// ============================================================

/**
 * 📝 业务场景：表单字段联动
 *
 * 场景描述：
 * - 字段 A 变化时，需要更新依赖它的字段 B、C
 * - 用哈希表记录依赖关系
 */
type FieldChangeHandler = (newValue: unknown, fieldName: string) => void;

class FormDependencyManager {
  // fieldName -> 依赖它的处理函数列表
  private dependencies = new Map<string, Set<FieldChangeHandler>>();

  /**
   * 注册依赖：当 sourceField 变化时，执行 handler
   */
  addDependency(sourceField: string, handler: FieldChangeHandler): () => void {
    if (!this.dependencies.has(sourceField)) {
      this.dependencies.set(sourceField, new Set());
    }

    this.dependencies.get(sourceField)!.add(handler);

    // 返回取消订阅函数
    return () => {
      this.dependencies.get(sourceField)?.delete(handler);
    };
  }

  /**
   * 触发依赖更新
   */
  triggerChange(fieldName: string, newValue: unknown): void {
    const handlers = this.dependencies.get(fieldName);
    if (handlers) {
      handlers.forEach((handler) => handler(newValue, fieldName));
    }
  }

  /**
   * 清除所有依赖
   */
  clear(): void {
    this.dependencies.clear();
  }
}

// 使用示例
const formDeps = new FormDependencyManager();

// 省份变化时，更新城市列表
formDeps.addDependency('province', (province) => {
  console.log(`省份变为 ${province}，更新城市列表`);
});

// 城市变化时，更新区县列表
formDeps.addDependency('city', (city) => {
  console.log(`城市变为 ${city}，更新区县列表`);
});

// ============================================================
// 3. Set - 防止重复提交
// ============================================================

/**
 * 📝 业务场景：接口防重复调用
 *
 * 场景描述：
 * - 用户快速点击按钮，可能发起多次相同请求
 * - 用 Set 记录进行中的请求，防止重复
 */
class DuplicateRequestGuard {
  private pendingRequests = new Set<string>();

  /**
   * 生成请求唯一键
   */
  private getKey(method: string, url: string, body?: unknown): string {
    return `${method}:${url}:${JSON.stringify(body || '')}`;
  }

  /**
   * 包装请求函数，自动防重复
   */
  async guard<T>(method: string, url: string, body: unknown, requestFn: () => Promise<T>): Promise<T> {
    const key = this.getKey(method, url, body);

    // 如果已有相同请求在进行中，抛出错误或返回 pending promise
    if (this.pendingRequests.has(key)) {
      throw new Error('请求正在进行中，请勿重复提交');
    }

    this.pendingRequests.add(key);

    try {
      const result = await requestFn();
      return result;
    } finally {
      this.pendingRequests.delete(key);
    }
  }

  /**
   * 检查是否有请求在进行中
   */
  isPending(method: string, url: string, body?: unknown): boolean {
    const key = this.getKey(method, url, body);
    return this.pendingRequests.has(key);
  }
}

// 使用示例
const requestGuard = new DuplicateRequestGuard();

async function submitOrder(orderData: object): Promise<void> {
  await requestGuard.guard('POST', '/api/order', orderData, async () => {
    // 实际的提交逻辑
    const response = await fetch('/api/order', {
      method: 'POST',
      body: JSON.stringify(orderData),
    });
    return response.json();
  });
}

// ============================================================
// 4. 字符计数 - 文本统计分析
// ============================================================

/**
 * 📝 业务场景：文章统计
 *
 * 场景描述：
 * - 统计文章字数、词频
 * - 用于 SEO 分析、关键词提取
 */
interface TextStats {
  charCount: number;
  wordCount: number;
  sentenceCount: number;
  topWords: [string, number][];
}

function analyzeText(text: string, topN = 10): TextStats {
  // 字符数（不含空格）
  const charCount = text.replace(/\s/g, '').length;

  // 分词（简单按空格和标点分割）
  const words = text
    .toLowerCase()
    .split(/[\s,.\-;:!?'"()[\]{}]+/)
    .filter((w) => w.length > 0);

  const wordCount = words.length;

  // 句子数
  const sentenceCount = (text.match(/[.!?]+/g) || []).length || 1;

  // 词频统计（哈希计数）
  const wordFreq = new Map<string, number>();
  for (const word of words) {
    wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
  }

  // 排序获取 Top N
  const topWords = [...wordFreq.entries()].sort((a, b) => b[1] - a[1]).slice(0, topN);

  return { charCount, wordCount, sentenceCount, topWords };
}

// 使用示例
const stats = analyzeText('The quick brown fox jumps over the lazy dog. The dog was not amused.');
// console.log(stats);

// ============================================================
// 5. 字符串哈希 - 内容指纹/去重
// ============================================================

/**
 * 📝 业务场景：内容去重
 *
 * 场景描述：
 * - 用户可能提交重复的内容
 * - 通过内容指纹快速判断是否重复
 */
class ContentDeduplicator {
  private seenHashes = new Set<string>();

  /**
   * 简单哈希函数（生产环境应使用 crypto API）
   */
  private hash(content: string): string {
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash.toString(16);
  }

  /**
   * 标准化内容（去除空格、换行等差异）
   */
  private normalize(content: string): string {
    return content.toLowerCase().replace(/\s+/g, ' ').trim();
  }

  /**
   * 检查是否重复
   */
  isDuplicate(content: string): boolean {
    const normalized = this.normalize(content);
    const contentHash = this.hash(normalized);
    return this.seenHashes.has(contentHash);
  }

  /**
   * 添加内容
   */
  add(content: string): boolean {
    const normalized = this.normalize(content);
    const contentHash = this.hash(normalized);

    if (this.seenHashes.has(contentHash)) {
      return false; // 重复
    }

    this.seenHashes.add(contentHash);
    return true; // 成功添加
  }

  /**
   * 清除记录
   */
  clear(): void {
    this.seenHashes.clear();
  }
}

// ============================================================
// 6. 回文检测 - 用户输入验证
// ============================================================

/**
 * 📝 业务场景：用户名/密码规则验证
 *
 * 场景描述：
 * - 某些系统禁止回文密码（太简单）
 * - 或需要检测回文用户名
 */
function isPalindrome(s: string): boolean {
  // 只保留字母数字，忽略大小写
  const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, '');

  let left = 0;
  let right = cleaned.length - 1;

  while (left < right) {
    if (cleaned[left] !== cleaned[right]) {
      return false;
    }
    left++;
    right--;
  }

  return true;
}

/**
 * 密码强度检查（禁止回文）
 */
function checkPasswordStrength(password: string): { valid: boolean; message: string } {
  if (password.length < 8) {
    return { valid: false, message: '密码长度至少 8 位' };
  }

  if (isPalindrome(password)) {
    return { valid: false, message: '密码不能是回文' };
  }

  // 更多检查规则...

  return { valid: true, message: '密码强度合格' };
}

// ============================================================
// 7. 字符串模板 - 占位符替换
// ============================================================

/**
 * 📝 业务场景：消息模板渲染
 *
 * 场景描述：
 * - 后端返回模板：「您好，{{name}}，您的订单 {{orderId}} 已发货」
 * - 前端替换占位符
 */
function renderTemplate(template: string, data: Record<string, string | number>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
    return key in data ? String(data[key]) : match;
  });
}

// 使用示例
const msg = renderTemplate('您好，{{name}}，您的订单 {{orderId}} 已发货', {
  name: '张三',
  orderId: 12345,
});
// console.log(msg); // "您好，张三，您的订单 12345 已发货"

// ============================================================
// 8. 异位词分组 - 搜索结果聚合
// ============================================================

/**
 * 📝 业务场景：搜索结果分组
 *
 * 场景描述：
 * - 搜索结果可能有相似项（如拼写变体）
 * - 按相似度分组展示
 */
function groupAnagrams(words: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();

  for (const word of words) {
    // 排序后的字符串作为分组 key
    const key = word.toLowerCase().split('').sort().join('');

    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(word);
  }

  return groups;
}

// 使用示例
const searchResults = ['tea', 'eat', 'ate', 'tan', 'ant', 'bat'];
const grouped = groupAnagrams(searchResults);
// Map { 'aet' => ['tea', 'eat', 'ate'], 'ant' => ['tan', 'ant'], 'abt' => ['bat'] }

// ============================================================
// 9. 字符串相似度 - 模糊搜索
// ============================================================

/**
 * 📝 业务场景：模糊搜索/拼写纠正
 *
 * 场景描述：
 * - 用户输入可能有拼写错误
 * - 计算编辑距离，找最相似的结果
 */
function levenshteinDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;

  // dp[i][j] = a[0..i-1] 和 b[0..j-1] 的编辑距离
  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map(() => Array(n + 1).fill(0));

  // 初始化
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  // 填表
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }

  return dp[m][n];
}

/**
 * 模糊搜索：找最相似的结果
 */
function fuzzySearch(query: string, candidates: string[], maxDistance = 2): string[] {
  return candidates
    .map((candidate) => ({
      word: candidate,
      distance: levenshteinDistance(query.toLowerCase(), candidate.toLowerCase()),
    }))
    .filter((item) => item.distance <= maxDistance)
    .sort((a, b) => a.distance - b.distance)
    .map((item) => item.word);
}

// 使用示例
const suggestions = fuzzySearch('teh', ['the', 'tea', 'team', 'tech', 'test']);
// console.log(suggestions); // ['the', 'tea', 'tech']

// ============================================================
// 10. 关键词高亮 - 搜索结果展示
// ============================================================

/**
 * 📝 业务场景：搜索结果关键词高亮
 *
 * 场景描述：
 * - 在搜索结果中高亮显示匹配的关键词
 * - 支持多个关键词
 */
function highlightKeywords(
  text: string,
  keywords: string[],
  highlightFn: (word: string) => string = (w) => `<mark>${w}</mark>`
): string {
  if (keywords.length === 0) return text;

  // 转义正则特殊字符
  const escaped = keywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

  // 构建正则（不区分大小写）
  const regex = new RegExp(`(${escaped.join('|')})`, 'gi');

  return text.replace(regex, (match) => highlightFn(match));
}

// 使用示例
const highlighted = highlightKeywords('The quick brown fox jumps over the lazy dog', ['fox', 'dog']);
// "The quick brown <mark>fox</mark> jumps over the lazy <mark>dog</mark>"

// ============================================================
// 导出
// ============================================================

export {
  RequestCache,
  FormDependencyManager,
  DuplicateRequestGuard,
  analyzeText,
  ContentDeduplicator,
  isPalindrome,
  checkPasswordStrength,
  renderTemplate,
  groupAnagrams,
  levenshteinDistance,
  fuzzySearch,
  highlightKeywords,
};

