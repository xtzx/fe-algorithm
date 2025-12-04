/**
 * ============================================================
 * 📚 Node.js 进阶
 * ============================================================
 *
 * 面试考察重点：
 * 1. 事件循环与异步
 * 2. Stream 流处理
 * 3. 进程与集群
 * 4. 性能优化
 */

// ============================================================
// 1. 事件循环
// ============================================================

/**
 * 📊 Node.js 事件循环
 *
 * 与浏览器不同，Node.js 有 6 个阶段：
 *
 * ┌───────────────────────────────────────────────────────────────────┐
 * │                          Event Loop                               │
 * │                                                                   │
 * │  ┌─────────────┐                                                  │
 * │  │   timers    │ ← setTimeout/setInterval                        │
 * │  └──────┬──────┘                                                  │
 * │         │                                                         │
 * │  ┌──────▼──────┐                                                  │
 * │  │  pending    │ ← 系统操作回调                                   │
 * │  └──────┬──────┘                                                  │
 * │         │                                                         │
 * │  ┌──────▼──────┐                                                  │
 * │  │ idle/prepare│ ← 内部使用                                      │
 * │  └──────┬──────┘                                                  │
 * │         │                                                         │
 * │  ┌──────▼──────┐                                                  │
 * │  │    poll     │ ← I/O 回调                                      │
 * │  └──────┬──────┘                                                  │
 * │         │                                                         │
 * │  ┌──────▼──────┐                                                  │
 * │  │   check     │ ← setImmediate                                  │
 * │  └──────┬──────┘                                                  │
 * │         │                                                         │
 * │  ┌──────▼──────┐                                                  │
 * │  │close callbacks│ ← socket.on('close')                          │
 * │  └─────────────┘                                                  │
 * │                                                                   │
 * │  每个阶段之间：执行 process.nextTick 和 微任务                     │
 * │                                                                   │
 * └───────────────────────────────────────────────────────────────────┘
 *
 * 执行顺序：
 * 1. process.nextTick（最高优先级）
 * 2. 微任务（Promise）
 * 3. 各阶段宏任务
 */

// 事件循环示例
const eventLoopExample = `
console.log('1');

setTimeout(() => console.log('2'), 0);

setImmediate(() => console.log('3'));

Promise.resolve().then(() => console.log('4'));

process.nextTick(() => console.log('5'));

console.log('6');

// 输出顺序：1 → 6 → 5 → 4 → 2 → 3
// 注意：setTimeout 和 setImmediate 的顺序在某些情况下不确定
`;

// 在 I/O 回调中，setImmediate 总是先执行
const ioCallbackExample = `
const fs = require('fs');

fs.readFile('file.txt', () => {
  setTimeout(() => console.log('timeout'), 0);
  setImmediate(() => console.log('immediate'));
});

// 输出：immediate → timeout（setImmediate 在 check 阶段，先执行）
`;

// ============================================================
// 2. Stream 流处理
// ============================================================

/**
 * 📊 Stream 类型
 *
 * - Readable：可读流（fs.createReadStream）
 * - Writable：可写流（fs.createWriteStream）
 * - Duplex：双工流（net.Socket）
 * - Transform：转换流（zlib.createGzip）
 *
 * 📊 优势
 *
 * - 内存效率：不需要一次性加载全部数据
 * - 时间效率：可以边读边处理
 */

// 大文件处理
const streamExample = `
const fs = require('fs');
const zlib = require('zlib');

// 读取大文件 → 压缩 → 写入
fs.createReadStream('big-file.txt')
  .pipe(zlib.createGzip())
  .pipe(fs.createWriteStream('big-file.txt.gz'))
  .on('finish', () => console.log('Done'));

// 对比：不使用流（内存可能溢出）
const content = fs.readFileSync('big-file.txt');
const compressed = zlib.gzipSync(content);
fs.writeFileSync('big-file.txt.gz', compressed);
`;

// 自定义 Transform 流
const customTransformExample = `
const { Transform } = require('stream');

class UpperCaseTransform extends Transform {
  _transform(chunk, encoding, callback) {
    this.push(chunk.toString().toUpperCase());
    callback();
  }
}

// 使用
process.stdin
  .pipe(new UpperCaseTransform())
  .pipe(process.stdout);
`;

// 流式 HTTP 响应
const streamHttpExample = `
const http = require('http');
const fs = require('fs');

http.createServer((req, res) => {
  // 流式发送大文件
  const stream = fs.createReadStream('large-video.mp4');

  // 设置 Content-Type
  res.setHeader('Content-Type', 'video/mp4');

  // 管道连接
  stream.pipe(res);

  // 错误处理
  stream.on('error', (err) => {
    res.statusCode = 500;
    res.end('Error');
  });
}).listen(3000);
`;

// ============================================================
// 3. 进程与集群
// ============================================================

/**
 * 📊 多进程模型
 *
 * Node.js 是单线程的，需要利用多核 CPU：
 *
 * 1. child_process：创建子进程
 * 2. cluster：集群模式
 * 3. worker_threads：工作线程（CPU 密集型）
 */

// child_process 使用
const childProcessExample = `
const { fork, exec, spawn } = require('child_process');

// exec：执行命令，有缓冲区大小限制
exec('ls -la', (error, stdout, stderr) => {
  console.log(stdout);
});

// spawn：流式输出，适合大输出
const ls = spawn('ls', ['-la']);
ls.stdout.on('data', (data) => console.log(data.toString()));

// fork：创建 Node.js 子进程，支持 IPC 通信
const child = fork('./child.js');
child.send({ type: 'start' });
child.on('message', (msg) => console.log('From child:', msg));
`;

// cluster 集群
const clusterExample = `
const cluster = require('cluster');
const http = require('http');
const numCPUs = require('os').cpus().length;

if (cluster.isMaster) {
  console.log(\`Master \${process.pid} is running\`);

  // 创建工作进程
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  // 监听工作进程退出，自动重启
  cluster.on('exit', (worker, code, signal) => {
    console.log(\`Worker \${worker.process.pid} died\`);
    cluster.fork(); // 重启
  });

} else {
  // 工作进程创建 HTTP 服务器
  http.createServer((req, res) => {
    res.writeHead(200);
    res.end(\`Hello from worker \${process.pid}\`);
  }).listen(8000);

  console.log(\`Worker \${process.pid} started\`);
}
`;

// worker_threads（CPU 密集型任务）
const workerThreadsExample = `
// main.js
const { Worker } = require('worker_threads');

function runWorker(data) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./worker.js', {
      workerData: data
    });

    worker.on('message', resolve);
    worker.on('error', reject);
    worker.on('exit', (code) => {
      if (code !== 0) {
        reject(new Error(\`Worker stopped with code \${code}\`));
      }
    });
  });
}

// worker.js
const { parentPort, workerData } = require('worker_threads');

// CPU 密集型计算
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

const result = fibonacci(workerData.n);
parentPort.postMessage(result);
`;

// ============================================================
// 4. 性能优化
// ============================================================

/**
 * 📊 Node.js 性能优化
 *
 * 1. 异步优化：避免阻塞事件循环
 * 2. 内存优化：避免内存泄漏
 * 3. I/O 优化：使用 Stream
 * 4. 并发优化：多进程/线程
 * 5. 缓存优化：Redis/内存缓存
 */

// 内存泄漏检测
const memoryLeakDetection = `
// 监控内存使用
setInterval(() => {
  const usage = process.memoryUsage();
  console.log({
    heapUsed: Math.round(usage.heapUsed / 1024 / 1024) + 'MB',
    heapTotal: Math.round(usage.heapTotal / 1024 / 1024) + 'MB',
    external: Math.round(usage.external / 1024 / 1024) + 'MB',
    rss: Math.round(usage.rss / 1024 / 1024) + 'MB',
  });
}, 5000);

// 常见内存泄漏
// 1. 全局变量
// 2. 闭包
// 3. 事件监听未移除
// 4. 定时器未清理
`;

// 性能分析
const profilingExample = `
// 1. 使用 --inspect 启动
node --inspect app.js

// 2. 使用 clinic.js
npx clinic doctor -- node app.js

// 3. CPU profiling
const { Session } = require('inspector');
const session = new Session();
session.connect();

session.post('Profiler.enable');
session.post('Profiler.start');

// ... 运行代码 ...

session.post('Profiler.stop', (err, { profile }) => {
  fs.writeFileSync('profile.cpuprofile', JSON.stringify(profile));
});
`;

// ============================================================
// 5. 常用模块
// ============================================================

/**
 * 📊 核心模块
 *
 * - fs：文件系统
 * - path：路径处理
 * - http/https：HTTP 服务
 * - net：TCP 服务
 * - crypto：加密
 * - buffer：二进制数据
 * - events：事件
 * - util：工具函数
 */

// 文件操作最佳实践
const fsExample = `
const fs = require('fs').promises;
const path = require('path');

// 递归读取目录
async function readDirRecursive(dir) {
  const files = [];

  async function walk(currentDir) {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(fullPath);
      } else {
        files.push(fullPath);
      }
    }
  }

  await walk(dir);
  return files;
}

// 安全地写入文件（原子操作）
async function safeWriteFile(filePath, content) {
  const tempPath = filePath + '.tmp';
  await fs.writeFile(tempPath, content);
  await fs.rename(tempPath, filePath);
}
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. 阻塞事件循环
 *    - 避免同步操作
 *    - CPU 密集任务用 Worker
 *
 * 2. 内存泄漏
 *    - 及时移除事件监听
 *    - 清理定时器
 *    - 避免大对象长期持有
 *
 * 3. 错误处理
 *    - Promise 必须 catch
 *    - unhandledRejection 监听
 *
 * 4. 回调地狱
 *    - 使用 async/await
 *    - 使用 util.promisify
 *
 * 5. 文件描述符泄漏
 *    - 及时关闭文件/流
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Node.js 事件循环和浏览器的区别？
 * A:
 *    - Node.js 有 6 个阶段
 *    - 每个阶段之间执行 nextTick 和微任务
 *    - setImmediate 是 Node 特有
 *
 * Q2: 如何处理 CPU 密集型任务？
 * A:
 *    - worker_threads
 *    - child_process.fork
 *    - 任务队列（如 Bull）
 *
 * Q3: Stream 的优势是什么？
 * A:
 *    - 内存效率：边读边处理
 *    - 时间效率：不等待全部数据
 *    - 管道组合：链式处理
 *
 * Q4: cluster 模块的工作原理？
 * A:
 *    - Master 进程 fork 多个 Worker
 *    - 共享同一个端口（负载均衡）
 *    - IPC 通信
 */

// ============================================================
// 8. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：高性能 HTTP 服务
 */

const httpServerExample = `
const cluster = require('cluster');
const http = require('http');
const numCPUs = require('os').cpus().length;

if (cluster.isMaster) {
  // 主进程
  console.log(\`Master \${process.pid} is running\`);

  // Fork workers
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker) => {
    console.log(\`Worker \${worker.process.pid} died, restarting...\`);
    cluster.fork();
  });

} else {
  // 工作进程
  const server = http.createServer((req, res) => {
    // 业务逻辑
  });

  server.listen(8000);
  console.log(\`Worker \${process.pid} started\`);
}

// 使用 PM2 更简单
// pm2 start app.js -i max
`;

/**
 * 🏢 场景：文件上传处理
 */

const fileUploadExample = `
const http = require('http');
const fs = require('fs');
const path = require('path');
const Busboy = require('busboy');

http.createServer((req, res) => {
  if (req.method === 'POST') {
    const busboy = Busboy({ headers: req.headers });

    busboy.on('file', (name, file, info) => {
      const savePath = path.join(__dirname, 'uploads', info.filename);
      const writeStream = fs.createWriteStream(savePath);

      // 流式写入，不占用大量内存
      file.pipe(writeStream);

      file.on('end', () => {
        console.log(\`File \${info.filename} uploaded\`);
      });
    });

    busboy.on('finish', () => {
      res.writeHead(200);
      res.end('Upload complete');
    });

    req.pipe(busboy);
  }
}).listen(3000);
`;

export {
  eventLoopExample,
  ioCallbackExample,
  streamExample,
  customTransformExample,
  streamHttpExample,
  childProcessExample,
  clusterExample,
  workerThreadsExample,
  memoryLeakDetection,
  profilingExample,
  fsExample,
  httpServerExample,
  fileUploadExample,
};

