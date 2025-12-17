# 03. Webpack/Vite 插件开发教程

> 扩展构建工具能力

---

## 📑 目录

1. [Webpack 插件机制](#webpack-插件机制)
2. [Webpack 插件实战](#webpack-插件实战)
3. [Vite 插件机制](#vite-插件机制)
4. [Vite 插件实战](#vite-插件实战)
5. [Babel + 构建工具组合](#babel--构建工具组合)

---

## Webpack 插件机制

### Tapable 钩子系统

Webpack 插件系统基于 **Tapable**，一个发布-订阅模式的钩子库。

```
┌─────────────────────────────────────────────────────────────────┐
│                   Webpack 构建生命周期                          │
│                                                                 │
│  Compiler                          Compilation                  │
│  (整体构建)                         (单次编译)                   │
│                                                                 │
│  ┌──────────┐                     ┌──────────┐                 │
│  │environment│                     │buildModule│                │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │  compile │ ──────────────────► │  seal    │                 │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │   make   │ ──────────────────► │ optimize │                 │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │afterCompile│                    │   emit   │                 │
│  └──────────┘                     └──────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 插件基本结构

```javascript
class MyWebpackPlugin {
  // 可选：接收配置
  constructor(options = {}) {
    this.options = options;
  }

  // 必须：apply 方法，Webpack 调用它来注册钩子
  apply(compiler) {
    // compiler: Webpack 编译器实例
    // 包含配置信息、文件系统等

    // 注册钩子
    compiler.hooks.done.tap('MyPlugin', (stats) => {
      console.log('构建完成！');
    });
  }
}

module.exports = MyWebpackPlugin;
```

### 常用 Compiler 钩子

| 钩子 | 触发时机 | 用途 |
|------|---------|------|
| `environment` | 环境准备好 | 修改配置 |
| `compile` | 开始编译 | 准备工作 |
| `make` | 开始构建模块 | - |
| `afterCompile` | 编译完成 | 添加额外资源 |
| `emit` | 生成资源到目录前 | 修改输出 |
| `done` | 构建完成 | 统计、通知 |

### 常用 Compilation 钩子

| 钩子 | 触发时机 | 用途 |
|------|---------|------|
| `buildModule` | 模块构建开始 | - |
| `succeedModule` | 模块构建成功 | - |
| `seal` | 封装开始 | - |
| `optimize` | 优化开始 | - |
| `optimizeChunks` | 优化 chunks | - |

---

## Webpack 插件实战

### 插件 1：构建信息输出

```javascript
// simple-build-info-plugin.js

/**
 * 构建信息输出插件
 * 在构建完成后输出构建统计信息
 */
class SimpleBuildInfoPlugin {
  constructor(options = {}) {
    this.options = {
      outputFile: options.outputFile || 'build-info.json',
      ...options
    };
  }

  apply(compiler) {
    const pluginName = 'SimpleBuildInfoPlugin';

    // 在 emit 阶段（资源输出前）添加构建信息文件
    compiler.hooks.emit.tapAsync(pluginName, (compilation, callback) => {
      // 收集构建信息
      const buildInfo = {
        buildTime: new Date().toISOString(),
        webpack: require('webpack').version,
        hash: compilation.hash,
        chunks: [],
        assets: []
      };

      // 收集 chunks 信息
      for (const chunk of compilation.chunks) {
        buildInfo.chunks.push({
          name: chunk.name,
          files: [...chunk.files],
          size: [...chunk.files].reduce((total, file) => {
            const asset = compilation.assets[file];
            return total + (asset ? asset.size() : 0);
          }, 0)
        });
      }

      // 收集 assets 信息
      for (const [name, asset] of Object.entries(compilation.assets)) {
        buildInfo.assets.push({
          name,
          size: asset.size()
        });
      }

      // 将信息写入输出
      const content = JSON.stringify(buildInfo, null, 2);
      compilation.assets[this.options.outputFile] = {
        source: () => content,
        size: () => content.length
      };

      callback();
    });

    // 构建完成后在控制台输出摘要
    compiler.hooks.done.tap(pluginName, (stats) => {
      console.log('\n========== 构建信息 ==========');
      console.log(`✓ 构建时间: ${stats.endTime - stats.startTime}ms`);
      console.log(`✓ Hash: ${stats.hash}`);
      console.log(`✓ 输出文件: ${this.options.outputFile}`);
      console.log('================================\n');
    });
  }
}

module.exports = SimpleBuildInfoPlugin;
```

### 插件 2：打包体积报告

```javascript
// bundle-size-report-plugin.js

/**
 * 打包体积报告插件
 * 分析打包产物体积，按大小排序输出
 */
class BundleSizeReportPlugin {
  constructor(options = {}) {
    this.options = {
      threshold: options.threshold || 100 * 1024, // 100KB 警告阈值
      showDetails: options.showDetails ?? true,
      ...options
    };
  }

  apply(compiler) {
    const pluginName = 'BundleSizeReportPlugin';

    compiler.hooks.done.tap(pluginName, (stats) => {
      const { assets } = stats.toJson({ assets: true });

      console.log('\n📦 打包体积报告\n');
      console.log('─'.repeat(60));

      // 按大小排序
      const sortedAssets = assets.sort((a, b) => b.size - a.size);

      // 计算总体积
      const totalSize = sortedAssets.reduce((sum, a) => sum + a.size, 0);

      // 输出详情
      if (this.options.showDetails) {
        sortedAssets.forEach((asset) => {
          const sizeStr = this.formatSize(asset.size);
          const percentage = ((asset.size / totalSize) * 100).toFixed(1);
          const warning = asset.size > this.options.threshold ? ' ⚠️' : '';

          console.log(
            `${sizeStr.padStart(10)} │ ${percentage.padStart(5)}% │ ${asset.name}${warning}`
          );
        });

        console.log('─'.repeat(60));
      }

      // 输出总计
      console.log(`${'Total:'.padStart(10)} ${this.formatSize(totalSize)}`);
      console.log(`${'Files:'.padStart(10)} ${sortedAssets.length}`);

      // 警告大文件
      const largeFiles = sortedAssets.filter(
        (a) => a.size > this.options.threshold
      );
      if (largeFiles.length > 0) {
        console.log(
          `\n⚠️  ${largeFiles.length} 个文件超过 ${this.formatSize(this.options.threshold)} 阈值`
        );
      }

      console.log('');
    });
  }

  formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
}

module.exports = BundleSizeReportPlugin;
```

### 使用插件

```javascript
// webpack.config.js
const SimpleBuildInfoPlugin = require('./plugins/simple-build-info-plugin');
const BundleSizeReportPlugin = require('./plugins/bundle-size-report-plugin');

module.exports = {
  // ...其他配置

  plugins: [
    new SimpleBuildInfoPlugin({
      outputFile: 'build-info.json'
    }),
    new BundleSizeReportPlugin({
      threshold: 50 * 1024,  // 50KB
      showDetails: true
    })
  ]
};
```

---

## Vite 插件机制

### Vite 插件 = Rollup 插件 + Vite 专属钩子

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vite 插件钩子                                │
│                                                                 │
│  Vite 专属钩子                     Rollup 兼容钩子              │
│  ──────────────                   ───────────────               │
│                                                                 │
│  ┌──────────┐                     ┌──────────┐                 │
│  │  config  │                     │  options │                 │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │configResolved│                  │buildStart│                 │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │configureServer│                 │ resolveId│                 │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │transformIndexHtml│              │   load   │                 │
│  └─────┬────┘                     └─────┬────┘                 │
│        ▼                                ▼                       │
│  ┌──────────┐                     ┌──────────┐                 │
│  │handleHotUpdate│                 │transform │                 │
│  └──────────┘                     └──────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 插件基本结构

```typescript
import type { Plugin } from 'vite';

export default function myPlugin(options = {}): Plugin {
  return {
    // 插件名称（必须）
    name: 'my-vite-plugin',

    // 插件执行顺序
    enforce: 'pre', // 'pre' | 'post'

    // 只在特定模式生效
    apply: 'build', // 'build' | 'serve'

    // 配置钩子
    config(config, env) {
      // 修改 Vite 配置
      return {
        define: {
          __BUILD_TIME__: JSON.stringify(new Date().toISOString())
        }
      };
    },

    // 转换钩子
    transform(code, id) {
      // 转换代码
      if (id.endsWith('.js')) {
        return code.replace(/console\.log/g, 'console.info');
      }
    }
  };
}
```

### 常用钩子

| 钩子 | 类型 | 用途 |
|------|------|------|
| `config` | Vite | 修改配置 |
| `configResolved` | Vite | 读取最终配置 |
| `configureServer` | Vite | 配置开发服务器 |
| `transformIndexHtml` | Vite | 转换 HTML |
| `handleHotUpdate` | Vite | 自定义 HMR |
| `resolveId` | Rollup | 解析模块 ID |
| `load` | Rollup | 加载模块内容 |
| `transform` | Rollup | 转换模块代码 |

---

## Vite 插件实战

### 插件 1：Banner 注入

```typescript
// banner-inject-plugin.ts
import type { Plugin } from 'vite';

interface BannerOptions {
  banner?: string;
  include?: RegExp;
  exclude?: RegExp;
}

/**
 * 为打包文件添加 banner 注释
 */
export default function bannerInjectPlugin(options: BannerOptions = {}): Plugin {
  const {
    banner = `/**\n * Built at ${new Date().toISOString()}\n */\n`,
    include = /\.(js|css)$/,
    exclude = /node_modules/
  } = options;

  return {
    name: 'vite-plugin-banner-inject',

    // 只在构建时生效
    apply: 'build',

    // 在 generateBundle 阶段处理（Rollup 钩子）
    generateBundle(outputOptions, bundle) {
      for (const [fileName, chunk] of Object.entries(bundle)) {
        // 检查文件类型
        if (!include.test(fileName)) continue;
        if (exclude && exclude.test(fileName)) continue;

        // 只处理有代码的 chunk
        if (chunk.type === 'chunk' || chunk.type === 'asset') {
          const source = chunk.type === 'chunk' ? chunk.code : chunk.source;

          if (typeof source === 'string') {
            if (chunk.type === 'chunk') {
              chunk.code = banner + source;
            } else {
              chunk.source = banner + source;
            }
          }
        }
      }
    }
  };
}
```

### 插件 2：环境变量替换

```typescript
// env-replace-plugin.ts
import type { Plugin } from 'vite';

interface EnvReplaceOptions {
  replacements?: Record<string, string>;
  prefix?: string;
}

/**
 * 自定义环境变量替换
 * 在代码中使用 __MY_VAR__ 格式的变量
 */
export default function envReplacePlugin(options: EnvReplaceOptions = {}): Plugin {
  const {
    replacements = {},
    prefix = '__'
  } = options;

  // 预处理替换规则
  const processedReplacements: Record<string, string> = {};
  for (const [key, value] of Object.entries(replacements)) {
    const pattern = `${prefix}${key}${prefix}`;
    processedReplacements[pattern] = JSON.stringify(value);
  }

  return {
    name: 'vite-plugin-env-replace',

    // 修改配置，添加 define
    config() {
      return {
        define: processedReplacements
      };
    },

    // 或使用 transform 手动替换
    transform(code, id) {
      // 排除 node_modules
      if (id.includes('node_modules')) return;

      let transformedCode = code;
      let hasChange = false;

      for (const [pattern, value] of Object.entries(processedReplacements)) {
        if (transformedCode.includes(pattern)) {
          transformedCode = transformedCode.split(pattern).join(value);
          hasChange = true;
        }
      }

      if (hasChange) {
        return {
          code: transformedCode,
          map: null // 简化处理，不生成 sourcemap
        };
      }
    },

    // 转换 HTML
    transformIndexHtml(html) {
      let transformedHtml = html;

      for (const [pattern, value] of Object.entries(processedReplacements)) {
        // 在 HTML 中不需要 JSON.stringify
        const rawValue = JSON.parse(value);
        transformedHtml = transformedHtml.split(pattern).join(rawValue);
      }

      return transformedHtml;
    }
  };
}
```

### 使用插件

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import bannerInjectPlugin from './vite-plugins/banner-inject-plugin';
import envReplacePlugin from './vite-plugins/env-replace-plugin';

export default defineConfig({
  plugins: [
    bannerInjectPlugin({
      banner: `/**\n * My App v1.0.0\n * Built: ${new Date().toISOString()}\n */\n`
    }),
    envReplacePlugin({
      replacements: {
        APP_VERSION: '1.0.0',
        BUILD_TIME: new Date().toISOString(),
        API_URL: 'https://api.example.com'
      }
    })
  ]
});
```

---

## Babel + 构建工具组合

### 在 Vite 中使用自定义 Babel 插件

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import babel from 'vite-plugin-babel';

export default defineConfig({
  plugins: [
    babel({
      babelConfig: {
        plugins: [
          './babel-plugins/log-inject-plugin.js'
        ]
      }
    })
  ]
});
```

### 职责划分

```
┌─────────────────────────────────────────────────────────────────┐
│                   插件职责划分                                  │
│                                                                 │
│  Babel 插件层                      构建工具插件层               │
│  ──────────────                   ───────────────               │
│                                                                 │
│  ✓ 语法转换                        ✓ 模块解析                   │
│    - ES6+ → ES5                     - 别名处理                  │
│    - JSX → JS                       - 虚拟模块                  │
│    - TypeScript → JS                                            │
│                                                                 │
│  ✓ 代码注入                        ✓ 资源处理                   │
│    - 自动导入                       - 文件加载                  │
│    - 日志注入                       - 图片优化                  │
│                                                                 │
│  ✓ 语法糖                          ✓ 构建优化                   │
│    - 装饰器转换                     - 代码分割                  │
│    - 宏展开                         - Tree Shaking             │
│                                                                 │
│  ✓ 静态分析                        ✓ 输出处理                   │
│    - 类型检查                       - 产物生成                  │
│    - 代码提取                       - 压缩混淆                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 选择原则

| 场景 | 使用 Babel 插件 | 使用构建工具插件 |
|------|:---------------:|:----------------:|
| 语法转换 (ES6/TS/JSX) | ✓ | |
| 代码注入 (import/日志) | ✓ | |
| 模块解析 (别名/虚拟) | | ✓ |
| 资源加载 (图片/CSS) | | ✓ |
| 构建产物处理 | | ✓ |
| HMR 自定义 | | ✓ |

