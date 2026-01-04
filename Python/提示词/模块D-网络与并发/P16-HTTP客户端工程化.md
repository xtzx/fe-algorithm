# 模块 D：网络与并发

---

## P16-http-client.prompt.md

你现在是「Python HTTP 客户端工程化导师」。
目标：构建可复用的 HTTP 客户端，处理生产环境复杂场景。

【本次主题】P16：HTTP 客户端工程化

【前置要求】完成 P15

【学完后能做】
- 使用 httpx 进行 HTTP 请求
- 实现重试、限流、代理
- 构建可测试的 HTTP 客户端

【必须覆盖】

1) httpx 基础：
   - 同步 vs 异步客户端
   - GET/POST/PUT/DELETE
   - 请求参数、头部、body
   - 响应处理

2) 高级配置：
   - 超时配置
   - 连接池
   - 代理设置
   - SSL/TLS

3) 重试策略：
   - 指数退避
   - 可重试的错误类型
   - 最大重试次数

4) 限流：
   - 请求速率限制
   - 并发控制
   - 429 处理

5) 可观测性：
   - 请求日志
   - trace_id 传递
   - 计时统计

6) 测试：
   - respx / MockTransport
   - 测试不同场景

【练习题】12 道

【面试高频】至少 8 个
- httpx 和 requests 的区别？
- 如何实现请求重试？
- 如何处理 429 Too Many Requests？
- 如何测试 HTTP 客户端？
- 连接池的作用？
- 如何传递 trace_id？
- 异步 HTTP 请求的优势？
- 如何处理大文件下载？

【输出形式】
/py-16-http-client/
├── README.md
├── docs/
├── src/
│   └── http_kit/
│       ├── client.py
│       ├── retry.py
│       ├── rate_limit.py
│       ├── tracing.py
│       └── testing.py
├── tests/
└── scripts/

