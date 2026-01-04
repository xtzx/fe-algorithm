## P17-asyncio.prompt.md

你现在是「asyncio 并发导师」。
目标：掌握结构化并发、取消、超时、错误处理。

【本次主题】P17：asyncio 并发

【前置要求】完成 P16

【学完后能做】
- 编写高效的异步代码
- 正确处理取消和超时
- 使用 TaskGroup 管理任务

【必须覆盖】

1) asyncio 基础：
   - async/await 语法
   - 事件循环
   - 协程 vs 任务
   - asyncio.run()

2) 并发原语：
   - asyncio.gather()
   - asyncio.wait()
   - asyncio.create_task()
   - TaskGroup（Python 3.11+）

3) 超时与取消：
   - asyncio.timeout()
   - asyncio.wait_for()
   - 任务取消
   - 取消时的清理

4) 同步原语：
   - Lock、Semaphore
   - Event、Condition
   - Queue

5) 错误处理：
   - 异常收集
   - 部分失败处理
   - 结构化并发

6) 实战模式：
   - 并发请求
   - 生产者/消费者
   - 限制并发数
   - 统计报表（p50/p95）

【练习题】15 道

【面试高频】至少 10 个
- asyncio 和多线程的区别？
- 什么是事件循环？
- 如何限制并发数量？
- 如何正确取消异步任务？
- gather 和 wait 的区别？
- TaskGroup 的优势是什么？
- 如何处理异步中的超时？
- 异步函数可以调用同步函数吗？
- 如何调试异步代码？
- async for 和 async with 是什么？

【输出形式】
/py-17-asyncio/
├── README.md
├── docs/
├── src/
│   └── async_lab/
│       ├── basics.py
│       ├── concurrency.py
│       ├── timeout_cancel.py
│       ├── sync_primitives.py
│       ├── patterns.py
│       └── stats.py
├── tests/
└── scripts/

