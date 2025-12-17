/**
 * 极简 Express HTTP 服务
 *
 * 功能：
 * - 健康检查接口
 * - 简单 API 示例
 * - Redis 连接示例
 */

import express, { Request, Response } from 'express';
import Redis from 'ioredis';

// ============================================
// 配置
// ============================================
const PORT = process.env.PORT || 3000;
const REDIS_HOST = process.env.REDIS_HOST || 'localhost';
const REDIS_PORT = parseInt(process.env.REDIS_PORT || '6379', 10);

// ============================================
// 初始化
// ============================================
const app = express();

// Redis 客户端（延迟连接，容器启动顺序可能有延迟）
let redis: Redis | null = null;

const connectRedis = async () => {
  try {
    redis = new Redis({
      host: REDIS_HOST,
      port: REDIS_PORT,
      retryStrategy: (times) => {
        if (times > 3) return null; // 最多重试 3 次
        return Math.min(times * 200, 2000);
      },
    });

    redis.on('connect', () => {
      console.log('✅ Redis 已连接');
    });

    redis.on('error', (err) => {
      console.error('❌ Redis 连接错误:', err.message);
    });
  } catch (error) {
    console.error('❌ Redis 初始化失败:', error);
  }
};

// ============================================
// 中间件
// ============================================
app.use(express.json());

// 请求日志
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} ${req.method} ${req.url}`);
  next();
});

// ============================================
// 路由
// ============================================

/**
 * 健康检查接口
 * 用于 Docker 健康检查和负载均衡器探活
 */
app.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    redis: redis?.status === 'ready' ? 'connected' : 'disconnected',
  });
});

/**
 * API 示例：获取访问计数
 */
app.get('/api/visits', async (req: Request, res: Response) => {
  try {
    if (!redis) {
      return res.status(503).json({ error: 'Redis 未连接' });
    }

    // 增加访问计数
    const count = await redis.incr('visit_count');

    res.json({
      message: 'Hello from Node.js API!',
      visitCount: count,
      hostname: process.env.HOSTNAME || 'unknown',
    });
  } catch (error) {
    console.error('API 错误:', error);
    res.status(500).json({ error: '服务器内部错误' });
  }
});

/**
 * API 示例：获取服务器信息
 */
app.get('/api/info', (req: Request, res: Response) => {
  res.json({
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    uptime: process.uptime(),
    memoryUsage: process.memoryUsage(),
    env: process.env.NODE_ENV || 'development',
  });
});

// ============================================
// 启动服务器
// ============================================
const start = async () => {
  // 连接 Redis
  await connectRedis();

  // 启动 HTTP 服务
  app.listen(PORT, () => {
    console.log(`🚀 服务器运行在 http://localhost:${PORT}`);
    console.log(`📊 健康检查: http://localhost:${PORT}/health`);
    console.log(`📝 API 示例: http://localhost:${PORT}/api/visits`);
  });
};

start().catch(console.error);

// 优雅退出
process.on('SIGTERM', () => {
  console.log('收到 SIGTERM 信号，正在优雅退出...');
  redis?.disconnect();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('收到 SIGINT 信号，正在优雅退出...');
  redis?.disconnect();
  process.exit(0);
});

