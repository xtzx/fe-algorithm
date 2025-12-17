/**
 * Webpack 插件：构建信息输出
 *
 * 功能：
 * 1. 在构建完成后，生成 build-info.json 文件
 * 2. 记录构建时间、hash、chunks、assets 信息
 * 3. 在控制台输出构建摘要
 *
 * 使用方法：
 *   const SimpleBuildInfoPlugin = require('./simple-build-info-plugin');
 *
 *   module.exports = {
 *     plugins: [
 *       new SimpleBuildInfoPlugin({
 *         outputFile: 'build-info.json'
 *       })
 *     ]
 *   };
 */

class SimpleBuildInfoPlugin {
  /**
   * 构造函数
   * @param {Object} options - 配置选项
   * @param {string} options.outputFile - 输出文件名，默认 'build-info.json'
   */
  constructor(options = {}) {
    this.options = {
      outputFile: options.outputFile || 'build-info.json',
      ...options
    };

    // 插件名称，用于 Tapable 标识
    this.pluginName = 'SimpleBuildInfoPlugin';
  }

  /**
   * Webpack 调用此方法来注册插件
   * @param {Compiler} compiler - Webpack 编译器实例
   */
  apply(compiler) {
    // ==========================================
    // 钩子 1: emit - 在输出资源到目录前执行
    // ==========================================
    // 使用 tapAsync 处理异步操作
    compiler.hooks.emit.tapAsync(this.pluginName, (compilation, callback) => {
      this.handleEmit(compilation, callback);
    });

    // ==========================================
    // 钩子 2: done - 构建完成后执行
    // ==========================================
    // 使用 tap 处理同步操作
    compiler.hooks.done.tap(this.pluginName, (stats) => {
      this.handleDone(stats);
    });
  }

  /**
   * 处理 emit 阶段：生成构建信息文件
   * @param {Compilation} compilation - 编译实例
   * @param {Function} callback - 完成回调
   */
  handleEmit(compilation, callback) {
    // 收集构建信息
    const buildInfo = {
      // 构建时间
      buildTime: new Date().toISOString(),

      // Webpack 版本
      webpackVersion: compilation.compiler.webpack
        ? compilation.compiler.webpack.version
        : 'unknown',

      // 构建 hash
      hash: compilation.hash,

      // 输出路径
      outputPath: compilation.outputOptions.path,

      // Chunks 信息
      chunks: [],

      // Assets 信息
      assets: [],

      // 模块数量
      moduleCount: compilation.modules.size,

      // 错误和警告数量
      errorsCount: compilation.errors.length,
      warningsCount: compilation.warnings.length
    };

    // 收集 chunks 信息
    for (const chunk of compilation.chunks) {
      const chunkInfo = {
        name: chunk.name || '(anonymous)',
        id: chunk.id,
        files: [...chunk.files],
        // 计算 chunk 总大小
        size: [...chunk.files].reduce((total, fileName) => {
          const asset = compilation.assets[fileName];
          return total + (asset ? asset.size() : 0);
        }, 0)
      };
      buildInfo.chunks.push(chunkInfo);
    }

    // 收集 assets 信息
    for (const [fileName, asset] of Object.entries(compilation.assets)) {
      buildInfo.assets.push({
        name: fileName,
        size: asset.size()
      });
    }

    // 按大小排序
    buildInfo.assets.sort((a, b) => b.size - a.size);

    // 将构建信息添加到输出
    const content = JSON.stringify(buildInfo, null, 2);

    // 添加新的 asset
    compilation.assets[this.options.outputFile] = {
      source: () => content,
      size: () => content.length
    };

    // 调用 callback 继续构建流程
    callback();
  }

  /**
   * 处理 done 阶段：输出构建摘要
   * @param {Stats} stats - 构建统计信息
   */
  handleDone(stats) {
    const duration = stats.endTime - stats.startTime;

    console.log('\n');
    console.log('╔══════════════════════════════════════════╗');
    console.log('║          构建信息摘要                     ║');
    console.log('╠══════════════════════════════════════════╣');
    console.log(`║  ✓ 构建时间:    ${this.padRight(duration + 'ms', 22)} ║`);
    console.log(`║  ✓ 构建 Hash:   ${this.padRight(stats.hash.substring(0, 16) + '...', 22)} ║`);
    console.log(`║  ✓ 输出文件:    ${this.padRight(this.options.outputFile, 22)} ║`);
    console.log('╚══════════════════════════════════════════╝');
    console.log('\n');

    // 如果有错误或警告，提示
    if (stats.hasErrors()) {
      console.log('❌ 构建有错误，请检查！');
    }
    if (stats.hasWarnings()) {
      console.log('⚠️  构建有警告，建议检查。');
    }
  }

  /**
   * 右侧填充字符串
   */
  padRight(str, length) {
    return str.padEnd(length);
  }
}

module.exports = SimpleBuildInfoPlugin;

