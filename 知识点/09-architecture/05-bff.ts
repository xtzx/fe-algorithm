/**
 * ============================================================
 * 📚 BFF 与接口设计
 * ============================================================
 *
 * 面试考察重点：
 * 1. BFF 的概念和价值
 * 2. GraphQL vs REST
 * 3. 接口设计规范
 * 4. 最佳实践
 */

// ============================================================
// 1. BFF 核心概念
// ============================================================

/**
 * 📖 什么是 BFF？
 *
 * BFF = Backend For Frontend（服务于前端的后端）
 *
 * 📊 为什么需要 BFF？
 *
 * 1. 接口聚合：多个后端接口合并为一个
 * 2. 数据裁剪：只返回前端需要的字段
 * 3. 格式转换：适配前端数据结构
 * 4. 业务逻辑：放置部分业务逻辑
 * 5. 解耦：前端与微服务解耦
 *
 * 📊 BFF 架构
 *
 * ┌───────────────────────────────────────────────────────────────┐
 * │                         客户端                                │
 * │    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
 * │    │   Web    │    │   App    │    │  小程序   │              │
 * │    └────┬─────┘    └────┬─────┘    └────┬─────┘              │
 * │         │               │               │                     │
 * │         └───────────────┼───────────────┘                     │
 * │                         │                                     │
 * │                         ▼                                     │
 * │    ┌──────────────────────────────────────────────────────┐  │
 * │    │                     BFF 层                           │  │
 * │    │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
 * │    │  │ Web BFF  │  │ App BFF  │  │ 小程序BFF │           │  │
 * │    │  └────┬─────┘  └────┬─────┘  └────┬─────┘           │  │
 * │    └───────┼─────────────┼─────────────┼──────────────────┘  │
 * │            │             │             │                      │
 * │            └─────────────┼─────────────┘                      │
 * │                          ▼                                    │
 * │    ┌──────────────────────────────────────────────────────┐  │
 * │    │                  微服务层                            │  │
 * │    │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐            │  │
 * │    │  │ 用户  │  │ 订单  │  │ 商品  │  │ 支付  │            │  │
 * │    │  └──────┘  └──────┘  └──────┘  └──────┘            │  │
 * │    └──────────────────────────────────────────────────────┘  │
 * └───────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. Node.js BFF 实现
// ============================================================

const bffImplementation = `
// 使用 Express/Koa/Nest.js

// 1. 接口聚合
app.get('/api/homepage', async (req, res) => {
  const [user, products, recommendations] = await Promise.all([
    userService.getCurrentUser(req.userId),
    productService.getHotProducts(),
    recommendService.getRecommendations(req.userId),
  ]);

  res.json({
    user: pick(user, ['id', 'name', 'avatar']),
    products: products.map(formatProduct),
    recommendations,
  });
});

// 2. 数据裁剪和转换
function formatProduct(product) {
  return {
    id: product.id,
    title: product.name,
    price: formatPrice(product.price),
    image: getImageUrl(product.coverImage),
    tags: product.tags?.slice(0, 3) || [],
  };
}

// 3. 错误处理
app.use((err, req, res, next) => {
  logger.error(err);
  
  // 统一错误格式
  res.status(err.status || 500).json({
    code: err.code || 'INTERNAL_ERROR',
    message: err.message || '服务异常',
    data: null,
  });
});

// 4. 缓存层
const cache = new LRUCache({ max: 1000, ttl: 60000 });

app.get('/api/config', async (req, res) => {
  const cacheKey = 'app-config';
  let config = cache.get(cacheKey);
  
  if (!config) {
    config = await configService.getAppConfig();
    cache.set(cacheKey, config);
  }
  
  res.json(config);
});
`;

// ============================================================
// 3. GraphQL
// ============================================================

/**
 * 📊 GraphQL vs REST
 *
 * ┌─────────────────┬────────────────────────┬────────────────────────┐
 * │ 特性             │ REST                   │ GraphQL                │
 * ├─────────────────┼────────────────────────┼────────────────────────┤
 * │ 数据获取         │ 固定结构               │ 按需获取               │
 * │ 请求数           │ 多个端点               │ 单一端点               │
 * │ 版本管理         │ URL 版本               │ Schema 演进            │
 * │ 类型系统         │ 无                     │ 强类型                 │
 * │ 学习成本         │ 低                     │ 中                     │
 * │ 缓存             │ HTTP 缓存              │ 需要客户端处理         │
 * │ 文档             │ 需要额外维护           │ 自动生成               │
 * └─────────────────┴────────────────────────┴────────────────────────┘
 *
 * GraphQL 适合场景：
 * - 复杂数据关系
 * - 多端复用
 * - 快速迭代
 *
 * REST 适合场景：
 * - 简单 CRUD
 * - 缓存友好
 * - 团队熟悉
 */

const graphqlExample = `
// Schema 定义
type Query {
  user(id: ID!): User
  products(filter: ProductFilter): [Product!]!
}

type Mutation {
  createOrder(input: CreateOrderInput!): Order!
  updateUser(id: ID!, input: UpdateUserInput!): User!
}

type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
}

type Product {
  id: ID!
  name: String!
  price: Float!
  category: Category!
}

// 前端查询
const GET_USER_WITH_ORDERS = gql\`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      orders {
        id
        total
        status
        items {
          product {
            name
            price
          }
          quantity
        }
      }
    }
  }
\`;

// React 使用
function UserOrders({ userId }) {
  const { data, loading, error } = useQuery(GET_USER_WITH_ORDERS, {
    variables: { id: userId },
  });

  if (loading) return <Loading />;
  if (error) return <Error />;

  return (
    <div>
      <h1>{data.user.name}</h1>
      <OrderList orders={data.user.orders} />
    </div>
  );
}
`;

// ============================================================
// 4. RESTful API 设计规范
// ============================================================

/**
 * 📊 RESTful 设计原则
 *
 * 1. 资源命名：使用名词，复数形式
 *    ✅ /users, /products
 *    ❌ /getUsers, /product
 *
 * 2. HTTP 方法语义
 *    GET：获取资源
 *    POST：创建资源
 *    PUT：完整更新
 *    PATCH：部分更新
 *    DELETE：删除资源
 *
 * 3. 状态码
 *    200：成功
 *    201：创建成功
 *    400：请求错误
 *    401：未认证
 *    403：无权限
 *    404：资源不存在
 *    500：服务器错误
 *
 * 4. 统一响应格式
 */

interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
  timestamp: number;
  traceId?: string;
}

// 成功响应
const successResponse = `
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 1,
    "name": "Tom"
  },
  "timestamp": 1703145600000
}
`;

// 错误响应
const errorResponse = `
{
  "code": 10001,
  "message": "用户不存在",
  "data": null,
  "timestamp": 1703145600000,
  "traceId": "abc123"
}
`;

/**
 * 📊 API 设计最佳实践
 */

const apiDesignExample = `
// 1. 资源路由设计
GET    /api/v1/users           # 获取用户列表
GET    /api/v1/users/:id       # 获取单个用户
POST   /api/v1/users           # 创建用户
PUT    /api/v1/users/:id       # 更新用户
DELETE /api/v1/users/:id       # 删除用户

// 2. 嵌套资源
GET    /api/v1/users/:id/orders    # 获取用户的订单

// 3. 过滤、分页、排序
GET    /api/v1/products?category=phone&minPrice=1000
GET    /api/v1/products?page=1&pageSize=20
GET    /api/v1/products?sortBy=price&order=desc

// 4. 批量操作
POST   /api/v1/users/batch         # 批量创建
DELETE /api/v1/users/batch         # 批量删除

// 5. 版本管理
/api/v1/users
/api/v2/users
`;

// ============================================================
// 5. 接口安全
// ============================================================

/**
 * 📊 接口安全措施
 *
 * 1. 认证（Authentication）
 *    - JWT Token
 *    - OAuth 2.0
 *    - Session
 *
 * 2. 授权（Authorization）
 *    - RBAC（基于角色）
 *    - ABAC（基于属性）
 *
 * 3. 数据校验
 *    - 参数校验
 *    - 类型校验
 *
 * 4. 限流
 *    - 请求频率限制
 *    - 并发限制
 *
 * 5. 防护
 *    - CSRF Token
 *    - XSS 过滤
 *    - SQL 注入防护
 */

const securityExample = `
// 1. JWT 认证中间件
function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  
  if (!token) {
    return res.status(401).json({ code: 401, message: 'Unauthorized' });
  }
  
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(401).json({ code: 401, message: 'Invalid token' });
  }
}

// 2. 权限检查中间件
function checkPermission(permission) {
  return (req, res, next) => {
    if (!req.user.permissions.includes(permission)) {
      return res.status(403).json({ code: 403, message: 'Forbidden' });
    }
    next();
  };
}

// 3. 请求限流
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 分钟
  max: 100, // 最多 100 次请求
  message: { code: 429, message: 'Too many requests' },
});

app.use('/api/', limiter);

// 4. 参数校验（使用 Joi）
const createUserSchema = Joi.object({
  name: Joi.string().min(2).max(50).required(),
  email: Joi.string().email().required(),
  password: Joi.string().min(8).required(),
});

app.post('/api/users', validate(createUserSchema), createUser);
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. BFF 职责边界不清
 *    - 不要放过多业务逻辑
 *    - 主要做聚合和适配
 *
 * 2. 接口设计不规范
 *    - 命名不统一
 *    - 响应格式不一致
 *
 * 3. 错误处理不完善
 *    - 统一错误码
 *    - 友好的错误信息
 *
 * 4. 缺少版本管理
 *    - API 变更导致前端崩溃
 *    - 使用版本号
 *
 * 5. 安全措施不足
 *    - 缺少认证授权
 *    - 参数校验不严
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: BFF 和 API Gateway 的区别？
 * A:
 *    API Gateway：
 *    - 通用网关
 *    - 路由、限流、认证
 *    - 不包含业务逻辑
 *
 *    BFF：
 *    - 面向特定前端
 *    - 接口聚合、数据裁剪
 *    - 可包含部分业务逻辑
 *
 * Q2: GraphQL 的 N+1 问题如何解决？
 * A:
 *    - DataLoader 批量加载
 *    - 字段级缓存
 *
 * Q3: 如何设计接口版本管理？
 * A:
 *    - URL 版本：/api/v1/
 *    - Header 版本：Accept-Version: v1
 *    - 版本兼容策略
 *
 * Q4: 前端如何处理接口错误？
 * A:
 *    - 统一错误拦截
 *    - 分类处理（认证、权限、业务）
 *    - 用户友好提示
 *    - 错误上报
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：前端请求封装
 */

const requestWrapperExample = `
// request.ts
import axios, { AxiosRequestConfig, AxiosError } from 'axios';

const instance = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
});

// 请求拦截
instance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = \`Bearer \${token}\`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截
instance.interceptors.response.use(
  (response) => {
    const { code, message, data } = response.data;
    
    if (code !== 0) {
      // 业务错误
      handleBusinessError(code, message);
      return Promise.reject(new Error(message));
    }
    
    return data;
  },
  (error: AxiosError) => {
    // HTTP 错误
    if (error.response) {
      const { status } = error.response;
      
      switch (status) {
        case 401:
          // 跳转登录
          window.location.href = '/login';
          break;
        case 403:
          message.error('没有权限');
          break;
        case 500:
          message.error('服务器错误');
          break;
      }
    } else {
      // 网络错误
      message.error('网络异常');
    }
    
    return Promise.reject(error);
  }
);

// 封装请求方法
export const request = {
  get: <T>(url: string, config?: AxiosRequestConfig) =>
    instance.get<any, T>(url, config),
    
  post: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    instance.post<any, T>(url, data, config),
    
  put: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    instance.put<any, T>(url, data, config),
    
  delete: <T>(url: string, config?: AxiosRequestConfig) =>
    instance.delete<any, T>(url, config),
};

// 使用
interface User {
  id: number;
  name: string;
}

const getUser = (id: number) => request.get<User>(\`/users/\${id}\`);
const createUser = (data: Partial<User>) => request.post<User>('/users', data);
`;

export {
  bffImplementation,
  graphqlExample,
  apiDesignExample,
  securityExample,
  requestWrapperExample,
  successResponse,
  errorResponse,
};

