/**
 * ============================================================
 * 📚 浏览器存储方案
 * ============================================================
 *
 * 面试考察重点：
 * 1. 各种存储方式的区别
 * 2. Cookie 的特性和使用
 * 3. Web Storage（localStorage/sessionStorage）
 * 4. IndexedDB
 * 5. 存储方案选型
 */

// ============================================================
// 1. 存储方案对比
// ============================================================

/**
 * 📊 浏览器存储方案对比
 *
 * ┌──────────────────┬──────────┬─────────────┬───────────────┬────────────┐
 * │ 特性              │ Cookie   │ localStorage│ sessionStorage│ IndexedDB  │
 * ├──────────────────┼──────────┼─────────────┼───────────────┼────────────┤
 * │ 存储大小          │ 4KB      │ 5-10MB      │ 5-10MB        │ 无限制     │
 * │ 生命周期          │ 可设置    │ 永久        │ 会话          │ 永久       │
 * │ 作用域            │ 同源+路径 │ 同源        │ 同源+标签页    │ 同源       │
 * │ 随请求发送        │ 是       │ 否          │ 否            │ 否         │
 * │ API               │ 简单     │ 简单        │ 简单          │ 复杂       │
 * │ 同步/异步         │ 同步     │ 同步        │ 同步          │ 异步       │
 * │ Web Worker 可用   │ 否       │ 否          │ 否            │ 是         │
 * └──────────────────┴──────────┴─────────────┴───────────────┴────────────┘
 */

// ============================================================
// 2. Cookie
// ============================================================

/**
 * 📖 Cookie 的特点
 *
 * - 最早的浏览器存储方案
 * - 主要用于服务器和浏览器之间传递数据
 * - 每次请求自动携带
 * - 大小限制 4KB
 * - 可设置过期时间
 */

/**
 * 📊 Cookie 属性
 *
 * - name=value：键值对
 * - Domain：作用域名
 * - Path：作用路径
 * - Expires/Max-Age：过期时间
 * - Secure：仅 HTTPS 发送
 * - HttpOnly：禁止 JS 访问（防 XSS）
 * - SameSite：跨站限制（防 CSRF）
 *   - Strict：完全禁止跨站发送
 *   - Lax：允许安全的跨站请求（链接、GET 表单）
 *   - None：不限制（需要 Secure）
 */

// Cookie 操作
const cookieUtils = {
  // 设置 Cookie
  set(name: string, value: string, days?: number, options: {
    path?: string;
    domain?: string;
    secure?: boolean;
    sameSite?: 'Strict' | 'Lax' | 'None';
  } = {}) {
    let cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)}`;

    if (days) {
      const date = new Date();
      date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
      cookie += `; expires=${date.toUTCString()}`;
    }

    if (options.path) cookie += `; path=${options.path}`;
    if (options.domain) cookie += `; domain=${options.domain}`;
    if (options.secure) cookie += '; secure';
    if (options.sameSite) cookie += `; samesite=${options.sameSite}`;

    document.cookie = cookie;
  },

  // 获取 Cookie
  get(name: string): string | null {
    const cookies = document.cookie.split('; ');
    for (const cookie of cookies) {
      const [key, value] = cookie.split('=');
      if (decodeURIComponent(key) === name) {
        return decodeURIComponent(value);
      }
    }
    return null;
  },

  // 删除 Cookie
  remove(name: string, path = '/') {
    document.cookie = `${encodeURIComponent(name)}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=${path}`;
  },

  // 获取所有 Cookie
  getAll(): Record<string, string> {
    const result: Record<string, string> = {};
    const cookies = document.cookie.split('; ');
    for (const cookie of cookies) {
      const [key, value] = cookie.split('=');
      if (key) {
        result[decodeURIComponent(key)] = decodeURIComponent(value || '');
      }
    }
    return result;
  },
};

// ============================================================
// 3. Web Storage
// ============================================================

/**
 * 📖 localStorage 和 sessionStorage
 *
 * localStorage：
 * - 永久存储，除非手动删除
 * - 同源策略限制
 * - 同源的所有标签页共享
 *
 * sessionStorage：
 * - 会话存储，标签页关闭即删除
 * - 同源策略限制
 * - 每个标签页独立
 * - 刷新页面不会清除
 */

// Web Storage 封装
class StorageWrapper {
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  // 设置值（支持对象）
  set<T>(key: string, value: T): void {
    try {
      this.storage.setItem(key, JSON.stringify(value));
    } catch (e) {
      // 可能是存储已满
      console.error('Storage set error:', e);
    }
  }

  // 获取值
  get<T>(key: string, defaultValue?: T): T | null {
    try {
      const item = this.storage.getItem(key);
      if (item === null) return defaultValue ?? null;
      return JSON.parse(item) as T;
    } catch {
      return defaultValue ?? null;
    }
  }

  // 删除值
  remove(key: string): void {
    this.storage.removeItem(key);
  }

  // 清空
  clear(): void {
    this.storage.clear();
  }

  // 获取所有 key
  keys(): string[] {
    const keys: string[] = [];
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key) keys.push(key);
    }
    return keys;
  }

  // 设置带过期时间的值
  setWithExpiry<T>(key: string, value: T, ttl: number): void {
    const item = {
      value,
      expiry: Date.now() + ttl,
    };
    this.set(key, item);
  }

  // 获取带过期时间的值
  getWithExpiry<T>(key: string): T | null {
    const item = this.get<{ value: T; expiry: number }>(key);
    if (!item) return null;
    if (Date.now() > item.expiry) {
      this.remove(key);
      return null;
    }
    return item.value;
  }
}

// 使用示例
const local = new StorageWrapper(localStorage);
const session = new StorageWrapper(sessionStorage);

// 监听 Storage 变化（跨标签页通信）
window.addEventListener('storage', (e) => {
  console.log('Storage changed:', {
    key: e.key,
    oldValue: e.oldValue,
    newValue: e.newValue,
    url: e.url,
  });
});

// ============================================================
// 4. IndexedDB
// ============================================================

/**
 * 📖 IndexedDB 特点
 *
 * - 大容量存储（无明确限制）
 * - 异步 API，不阻塞主线程
 * - 支持事务
 * - 支持索引
 * - 支持 Web Worker
 * - 适合存储大量结构化数据
 */

/**
 * 📊 IndexedDB 核心概念
 *
 * Database（数据库）
 *     │
 *     ├── Object Store（对象仓库，类似表）
 *     │       │
 *     │       ├── Record（记录，键值对）
 *     │       │
 *     │       └── Index（索引）
 *     │
 *     └── Transaction（事务）
 */

// IndexedDB 封装
class IndexedDBWrapper {
  private dbName: string;
  private version: number;
  private db: IDBDatabase | null = null;

  constructor(dbName: string, version = 1) {
    this.dbName = dbName;
    this.version = version;
  }

  // 打开数据库
  open(stores: { name: string; keyPath?: string; indexes?: { name: string; keyPath: string; unique?: boolean }[] }[]): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        for (const store of stores) {
          if (!db.objectStoreNames.contains(store.name)) {
            const objectStore = db.createObjectStore(store.name, {
              keyPath: store.keyPath || 'id',
              autoIncrement: !store.keyPath,
            });

            // 创建索引
            if (store.indexes) {
              for (const index of store.indexes) {
                objectStore.createIndex(index.name, index.keyPath, {
                  unique: index.unique || false,
                });
              }
            }
          }
        }
      };
    });
  }

  // 添加数据
  add<T>(storeName: string, data: T): Promise<IDBValidKey> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.add(data);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // 更新数据
  put<T>(storeName: string, data: T): Promise<IDBValidKey> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(data);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // 获取数据
  get<T>(storeName: string, key: IDBValidKey): Promise<T | undefined> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(key);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // 获取所有数据
  getAll<T>(storeName: string): Promise<T[]> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // 删除数据
  delete(storeName: string, key: IDBValidKey): Promise<void> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // 通过索引查询
  getByIndex<T>(storeName: string, indexName: string, value: IDBValidKey): Promise<T[]> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const index = store.index(indexName);
      const request = index.getAll(value);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // 关闭数据库
  close(): void {
    this.db?.close();
    this.db = null;
  }
}

// 使用示例
async function indexedDBExample() {
  const db = new IndexedDBWrapper('myDatabase', 1);

  await db.open([
    {
      name: 'users',
      keyPath: 'id',
      indexes: [
        { name: 'email', keyPath: 'email', unique: true },
        { name: 'age', keyPath: 'age' },
      ],
    },
  ]);

  // 添加数据
  await db.add('users', { id: 1, name: 'Tom', email: 'tom@example.com', age: 18 });

  // 查询数据
  const user = await db.get('users', 1);
  console.log(user);

  // 通过索引查询
  const usersByAge = await db.getByIndex('users', 'age', 18);
  console.log(usersByAge);

  db.close();
}

// ============================================================
// 5. Service Worker 与 Cache API
// ============================================================

/**
 * 📖 Service Worker
 *
 * - 独立于网页的后台脚本
 * - 可以拦截网络请求
 * - 实现离线缓存、推送通知
 * - 只能在 HTTPS 下使用
 */

/**
 * 📊 Cache API
 *
 * 与 Service Worker 配合，实现资源缓存
 */

// Service Worker 注册
async function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js');
      console.log('SW registered:', registration);
    } catch (error) {
      console.error('SW registration failed:', error);
    }
  }
}

// Service Worker 文件 (sw.js)
const swExample = `
  const CACHE_NAME = 'my-cache-v1';
  const urlsToCache = [
    '/',
    '/styles/main.css',
    '/scripts/main.js',
  ];

  // 安装时缓存资源
  self.addEventListener('install', (event) => {
    event.waitUntil(
      caches.open(CACHE_NAME)
        .then((cache) => cache.addAll(urlsToCache))
    );
  });

  // 拦截请求
  self.addEventListener('fetch', (event) => {
    event.respondWith(
      caches.match(event.request)
        .then((response) => {
          // 缓存命中，返回缓存
          if (response) {
            return response;
          }
          // 否则发起网络请求
          return fetch(event.request);
        })
    );
  });
`;

// ============================================================
// 6. 存储方案选型
// ============================================================

/**
 * 📊 选型建议
 *
 * Cookie：
 * - 需要发送给服务器的数据
 * - 会话标识、认证 token
 * - 注意：敏感数据用 HttpOnly
 *
 * localStorage：
 * - 需要持久化的小数据
 * - 用户偏好设置、主题
 * - 不敏感的缓存数据
 *
 * sessionStorage：
 * - 单次会话数据
 * - 表单数据暂存
 * - 单页应用状态
 *
 * IndexedDB：
 * - 大量结构化数据
 * - 离线应用数据
 * - 需要索引查询的数据
 *
 * Cache API：
 * - 静态资源缓存
 * - 离线优先策略
 * - PWA 应用
 */

// ============================================================
// 7. 高频面试题
// ============================================================

/**
 * 题目 1：Cookie、localStorage、sessionStorage 的区别？
 *
 * Cookie：
 * - 大小：4KB
 * - 随请求发送
 * - 可设置过期时间
 * - 同源 + 路径限制
 *
 * localStorage：
 * - 大小：5-10MB
 * - 永久存储
 * - 同源所有标签页共享
 *
 * sessionStorage：
 * - 大小：5-10MB
 * - 会话存储
 * - 每个标签页独立
 */

/**
 * 题目 2：如何实现跨标签页通信？
 *
 * 1. localStorage + storage 事件
 * 2. BroadcastChannel API
 * 3. SharedWorker
 * 4. Service Worker + postMessage
 * 5. WebSocket
 */

// BroadcastChannel 示例
const channel = new BroadcastChannel('my-channel');

// 发送消息
channel.postMessage({ type: 'update', data: 'hello' });

// 接收消息
channel.onmessage = (event) => {
  console.log('Received:', event.data);
};

/**
 * 题目 3：什么是 IndexedDB？适合什么场景？
 *
 * IndexedDB 是浏览器内置的 NoSQL 数据库：
 * - 大容量存储
 * - 异步 API
 * - 支持事务和索引
 * - 支持 Web Worker
 *
 * 适合场景：
 * - 离线应用数据
 * - 大量结构化数据
 * - 需要索引查询的数据
 * - 图片、文件等二进制数据
 */

/**
 * 题目 4：如何实现 localStorage 的过期功能？
 *
 * 存储时记录过期时间，读取时检查
 */
const storageWithExpiry = {
  set<T>(key: string, value: T, ttl: number) {
    const item = {
      value,
      expiry: Date.now() + ttl,
    };
    localStorage.setItem(key, JSON.stringify(item));
  },

  get<T>(key: string): T | null {
    const itemStr = localStorage.getItem(key);
    if (!itemStr) return null;

    const item = JSON.parse(itemStr);
    if (Date.now() > item.expiry) {
      localStorage.removeItem(key);
      return null;
    }
    return item.value;
  },
};

export {
  cookieUtils,
  StorageWrapper,
  local,
  session,
  IndexedDBWrapper,
  registerServiceWorker,
  storageWithExpiry,
};

