/**
 * ============================================================
 * 📚 API 设计与跨域
 * ============================================================
 *
 * 面试考察重点：
 * 1. RESTful API 设计
 * 2. GraphQL vs REST
 * 3. 跨域问题与解决方案
 * 4. 接口安全与认证
 */

// ============================================================
// 1. RESTful API 设计
// ============================================================

/**
 * 📊 REST 核心原则
 *
 * 1. 资源（Resource）
 *    - 用 URL 表示资源，名词复数
 *    - /users, /articles, /orders
 *
 * 2. 统一接口
 *    - GET：获取资源
 *    - POST：创建资源
 *    - PUT：完整更新资源
 *    - PATCH：部分更新资源
 *    - DELETE：删除资源
 *
 * 3. 无状态
 *    - 每个请求包含所有必要信息
 *    - 服务器不保存客户端状态
 *
 * 4. 可缓存
 *    - 响应明确标识是否可缓存
 */

/**
 * 📊 RESTful URL 设计
 *
 * ✅ 好的设计：
 * GET    /users          获取用户列表
 * GET    /users/123      获取单个用户
 * POST   /users          创建用户
 * PUT    /users/123      更新用户
 * DELETE /users/123      删除用户
 * GET    /users/123/orders   获取用户的订单
 *
 * ❌ 不好的设计：
 * GET    /getUsers
 * GET    /getUserById?id=123
 * POST   /createUser
 * POST   /deleteUser
 *
 * 📊 查询参数设计：
 * GET /users?page=1&limit=20           分页
 * GET /users?sort=created_at&order=desc 排序
 * GET /users?status=active&role=admin   过滤
 * GET /users?fields=id,name,email       字段选择
 */

/**
 * 📊 响应设计
 *
 * 成功响应：
 * {
 *   "code": 0,           // 业务状态码
 *   "message": "success",
 *   "data": {
 *     "id": 123,
 *     "name": "Tom"
 *   }
 * }
 *
 * 列表响应：
 * {
 *   "code": 0,
 *   "data": {
 *     "list": [...],
 *     "pagination": {
 *       "page": 1,
 *       "limit": 20,
 *       "total": 100,
 *       "totalPages": 5
 *     }
 *   }
 * }
 *
 * 错误响应：
 * {
 *   "code": 10001,
 *   "message": "用户名已存在",
 *   "errors": [
 *     { "field": "username", "message": "该用户名已被注册" }
 *   ]
 * }
 *
 * 💡 注意：HTTP 状态码和业务状态码的区别
 * - HTTP 状态码：表示请求的技术结果（200、400、500）
 * - 业务状态码：表示业务逻辑结果（0 成功，10001 用户名已存在）
 */

// ============================================================
// 2. GraphQL vs REST
// ============================================================

/**
 * 📊 GraphQL vs REST 对比
 *
 * ┌───────────────────┬────────────────────────┬────────────────────────┐
 * │ 特性               │ REST                   │ GraphQL                │
 * ├───────────────────┼────────────────────────┼────────────────────────┤
 * │ 数据获取           │ 多个端点               │ 单一端点               │
 * │ 过度获取           │ 可能获取不需要的字段   │ 精确获取需要的字段     │
 * │ 欠获取             │ 可能需要多次请求       │ 一次请求获取所有数据   │
 * │ 版本控制           │ /v1/users, /v2/users  │ 无需版本，字段废弃     │
 * │ 缓存               │ HTTP 缓存            │ 需要自定义缓存         │
 * │ 学习成本           │ 较低                  │ 较高                   │
 * │ 错误处理           │ HTTP 状态码           │ 响应中的 errors 字段   │
 * │ 文件上传           │ 原生支持              │ 需要额外处理           │
 * └───────────────────┴────────────────────────┴────────────────────────┘
 *
 * 💡 选型建议：
 * - REST：简单 CRUD、公开 API、缓存要求高
 * - GraphQL：复杂数据关系、移动端（带宽敏感）、快速迭代
 */

// ============================================================
// 3. 跨域详解（重要！）
// ============================================================

/**
 * 📊 同源策略回顾
 *
 * 同源：协议 + 域名 + 端口 完全相同
 *
 * 跨域限制：
 * 1. AJAX 请求
 * 2. Web 字体
 * 3. Canvas 绘制跨域图片
 * 4. 跨域脚本错误获取
 *
 * 不受限制：
 * - <script src>
 * - <link href>
 * - <img src>
 * - <video>/<audio>
 * - <iframe>（可加载，但不能操作 DOM）
 */

/**
 * 📊 CORS 详解
 *
 * ⚠️ 面试高频：简单请求和预检请求的区别
 *
 * 【简单请求】需同时满足：
 * 1. 方法：GET、HEAD、POST
 * 2. 头部只能是：
 *    - Accept
 *    - Accept-Language
 *    - Content-Language
 *    - Content-Type（仅限三种）
 * 3. Content-Type 只能是：
 *    - text/plain
 *    - multipart/form-data
 *    - application/x-www-form-urlencoded
 *
 * 【预检请求】触发条件（任意一个）：
 * - 使用 PUT、DELETE、PATCH 等方法
 * - Content-Type 是 application/json
 * - 带有自定义头部（如 Authorization）
 *
 * 💡 注意事项：
 * 1. 预检请求是浏览器自动发送的，不是开发者控制
 * 2. 预检请求可以被缓存（Access-Control-Max-Age）
 * 3. withCredentials 携带 Cookie 时，Allow-Origin 不能是 *
 */

/**
 * 📊 CORS 响应头详解
 *
 * Access-Control-Allow-Origin: https://example.com | *
 * - 允许的源，* 表示任意源
 * - ⚠️ 携带 Cookie 时不能是 *
 *
 * Access-Control-Allow-Methods: GET, POST, PUT, DELETE
 * - 允许的 HTTP 方法
 *
 * Access-Control-Allow-Headers: Content-Type, Authorization
 * - 允许的请求头
 *
 * Access-Control-Allow-Credentials: true
 * - 是否允许携带 Cookie
 * - 前端也要设置 withCredentials: true
 *
 * Access-Control-Expose-Headers: X-Custom-Header
 * - 暴露给前端的响应头
 * - 默认只能访问：Cache-Control、Content-Language、Content-Type、
 *   Expires、Last-Modified、Pragma
 *
 * Access-Control-Max-Age: 86400
 * - 预检请求缓存时间（秒）
 * - 避免频繁发送 OPTIONS 请求
 */

/**
 * 📊 跨域解决方案对比
 *
 * 1. CORS（推荐）
 *    - 服务器设置响应头
 *    - 最标准的方案
 *
 * 2. 代理服务器
 *    - 开发环境：webpack-dev-server proxy
 *    - 生产环境：Nginx 反向代理
 *    - 服务器之间无跨域限制
 *
 * 3. JSONP（已过时）
 *    - 利用 script 标签没有跨域限制
 *    - 只支持 GET 请求
 *    - 需要服务器配合
 *
 * 4. postMessage
 *    - 跨窗口通信
 *    - iframe 父子通信
 *
 * 5. WebSocket
 *    - 没有同源限制
 *    - 适合实时通信场景
 */

// JSONP 实现（了解原理即可）
function jsonp(url: string, callbackName: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');

    // 全局回调函数
    (window as any)[callbackName] = (data: any) => {
      resolve(data);
      document.body.removeChild(script);
      delete (window as any)[callbackName];
    };

    script.src = `${url}?callback=${callbackName}`;
    script.onerror = reject;
    document.body.appendChild(script);
  });
}

// 代理配置示例
const proxyConfig = `
  // webpack.config.js
  devServer: {
    proxy: {
      '/api': {
        target: 'http://backend.example.com',
        changeOrigin: true,
        pathRewrite: { '^/api': '' }
      }
    }
  }

  // Nginx 配置
  location /api {
    proxy_pass http://backend.example.com;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
  }
`;

// ============================================================
// 4. 接口安全与认证
// ============================================================

/**
 * 📊 认证方案对比
 *
 * 1. Cookie-Session
 *    - 传统方案，服务端存储会话
 *    - 天然防 CSRF（SameSite）
 *    - 不适合分布式、移动端
 *
 * 2. JWT（JSON Web Token）
 *    - 无状态，服务端不存储
 *    - 适合分布式、跨域、移动端
 *    - 无法主动失效，需要额外机制
 *
 * 3. OAuth 2.0
 *    - 第三方授权
 *    - 微信登录、GitHub 登录
 *
 * 💡 面试追问：
 * Q: JWT 如何实现登出？
 * A:
 * - 方案 1：维护 Token 黑名单
 * - 方案 2：设置较短过期时间 + Refresh Token
 * - 方案 3：修改用户密钥使所有 Token 失效
 */

/**
 * 📊 JWT 结构
 *
 * Header.Payload.Signature
 *
 * Header（Base64）：
 * {
 *   "alg": "HS256",
 *   "typ": "JWT"
 * }
 *
 * Payload（Base64）：
 * {
 *   "sub": "123",      // 主题（用户ID）
 *   "name": "Tom",
 *   "iat": 1234567890, // 签发时间
 *   "exp": 1234567890  // 过期时间
 * }
 *
 * Signature：
 * HMACSHA256(base64(header) + "." + base64(payload), secret)
 *
 * ⚠️ 注意：
 * - Payload 是 Base64 编码，不是加密！不要存敏感数据
 * - 签名只能验证完整性，不能保证机密性
 */

/**
 * 📊 Token 存储位置
 *
 * ┌───────────────────┬────────────────────────┬────────────────────────┐
 * │ 存储位置           │ 优点                   │ 缺点                   │
 * ├───────────────────┼────────────────────────┼────────────────────────┤
 * │ localStorage      │ 方便访问               │ XSS 可能被盗取         │
 * │ sessionStorage    │ 关闭标签页自动清除     │ XSS 可能被盗取         │
 * │ Cookie (HttpOnly) │ 防 XSS                 │ 容易 CSRF              │
 * │ 内存              │ 最安全                 │ 刷新页面丢失           │
 * └───────────────────┴────────────────────────┴────────────────────────┘
 *
 * 推荐方案：
 * - Access Token：内存或 localStorage（短期有效）
 * - Refresh Token：HttpOnly Cookie（长期有效）
 */

// ============================================================
// 5. 请求库封装最佳实践
// ============================================================

interface RequestConfig {
  baseURL?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

interface Response<T = any> {
  code: number;
  message: string;
  data: T;
}

// 请求封装示例
class HttpClient {
  private config: RequestConfig;

  constructor(config: RequestConfig = {}) {
    this.config = {
      baseURL: '',
      timeout: 10000,
      ...config,
    };
  }

  private async request<T>(url: string, options: RequestInit = {}): Promise<T> {
    const fullUrl = `${this.config.baseURL}${url}`;

    // 添加 token
    const token = localStorage.getItem('token');
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
      ...(options.headers as Record<string, string>),
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    // 超时控制
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(fullUrl, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // 处理 HTTP 错误
      if (!response.ok) {
        if (response.status === 401) {
          // Token 过期，刷新或跳转登录
          this.handleUnauthorized();
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: Response<T> = await response.json();

      // 处理业务错误
      if (result.code !== 0) {
        throw new Error(result.message);
      }

      return result.data;
    } catch (error) {
      clearTimeout(timeoutId);
      if ((error as Error).name === 'AbortError') {
        throw new Error('请求超时');
      }
      throw error;
    }
  }

  private handleUnauthorized() {
    localStorage.removeItem('token');
    window.location.href = '/login';
  }

  get<T>(url: string, params?: Record<string, any>): Promise<T> {
    const queryString = params ? '?' + new URLSearchParams(params).toString() : '';
    return this.request<T>(url + queryString, { method: 'GET' });
  }

  post<T>(url: string, data?: any): Promise<T> {
    return this.request<T>(url, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  put<T>(url: string, data?: any): Promise<T> {
    return this.request<T>(url, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  delete<T>(url: string): Promise<T> {
    return this.request<T>(url, { method: 'DELETE' });
  }
}

// ============================================================
// 6. 高频面试题（增强版）
// ============================================================

/**
 * 题目 1：什么情况下会发送 OPTIONS 预检请求？
 *
 * 触发条件（任意一个）：
 * 1. 方法不是 GET/HEAD/POST
 * 2. POST 的 Content-Type 不是表单默认的三种
 * 3. 带有自定义请求头
 *
 * 💡 追问：如何减少预检请求？
 * - Access-Control-Max-Age 缓存预检结果
 * - 尽量使用简单请求
 * - 合并请求减少次数
 */

/**
 * 题目 2：withCredentials 有什么作用？需要注意什么？
 *
 * 作用：跨域请求时携带 Cookie
 *
 * 前端设置：
 * fetch(url, { credentials: 'include' })
 * axios.defaults.withCredentials = true
 *
 * 后端要求：
 * - Access-Control-Allow-Credentials: true
 * - Access-Control-Allow-Origin 不能是 *
 *
 * ⚠️ 常见问题：
 * - Cookie 的 SameSite 属性会影响发送
 * - 第三方 Cookie 可能被浏览器阻止
 */

/**
 * 题目 3：前端如何处理接口错误？
 *
 * 错误分类：
 * 1. 网络错误：无法连接服务器
 * 2. HTTP 错误：4xx、5xx
 * 3. 业务错误：服务器返回的业务失败
 * 4. 超时错误：请求超时
 *
 * 处理策略：
 * - 全局错误拦截
 * - 统一错误提示
 * - 401 跳转登录
 * - 重试机制（网络错误）
 * - 降级方案（缓存数据）
 */

/**
 * 题目 4：JWT 和 Session 的区别？各自的优缺点？
 *
 * Session：
 * ✅ 可以主动失效
 * ✅ 不暴露用户信息
 * ❌ 服务端存储压力
 * ❌ 分布式需要共享
 *
 * JWT：
 * ✅ 无状态，易于扩展
 * ✅ 跨域、跨服务使用
 * ❌ 无法主动失效
 * ❌ 体积较大
 * ❌ Payload 不加密
 *
 * 💡 实际项目中，通常结合使用：
 * - Access Token（JWT）：短期有效，携带用户信息
 * - Refresh Token：长期有效，存储在 HttpOnly Cookie
 */

export {
  jsonp,
  proxyConfig,
  HttpClient,
};

