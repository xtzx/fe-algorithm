/**
 * Webpack 插件：打包体积报告
 *
 * 功能：
 * 1. 分析打包产物的体积
 * 2. 按大小排序输出各文件
 * 3. 对超过阈值的文件进行警告
 *
 * 使用方法：
 *   const BundleSizeReportPlugin = require('./bundle-size-report-plugin');
 *
 *   module.exports = {
 *     plugins: [
 *       new BundleSizeReportPlugin({
 *         threshold: 100 * 1024,  // 100KB 警告阈值
 *         showDetails: true       // 显示详细列表
 *       })
 *     ]
 *   };
 */

class BundleSizeReportPlugin {
  /**
   * 构造函数
   * @param {Object} options - 配置选项
   * @param {number} options.threshold - 大小警告阈值（字节），默认 100KB
   * @param {boolean} options.showDetails - 是否显示详细列表，默认 true
   * @param {string[]} options.exclude - 排除的文件模式
   */
  constructor(options = {}) {
    this.options = {
      threshold: options.threshold || 100 * 1024, // 100KB
      showDetails: options.showDetails !== false,
      exclude: options.exclude || [/\.map$/],
      ...options
    };

    this.pluginName = 'BundleSizeReportPlugin';
  }

  /**
   * 注册插件
   * @param {Compiler} compiler - Webpack 编译器
   */
  apply(compiler) {
    // 在构建完成后执行
    compiler.hooks.done.tap(this.pluginName, (stats) => {
      this.generateReport(stats);
    });
  }

  /**
   * 生成体积报告
   * @param {Stats} stats - 构建统计
   */
  generateReport(stats) {
    // 获取 assets 信息
    const { assets } = stats.toJson({
      assets: true,
      chunks: false,
      modules: false
    });

    // 过滤排除的文件
    const filteredAssets = assets.filter((asset) => {
      return !this.options.exclude.some((pattern) => {
        if (pattern instanceof RegExp) {
          return pattern.test(asset.name);
        }
        return asset.name.includes(pattern);
      });
    });

    // 按大小排序（从大到小）
    const sortedAssets = filteredAssets.sort((a, b) => b.size - a.size);

    // 计算总体积
    const totalSize = sortedAssets.reduce((sum, asset) => sum + asset.size, 0);

    // 找出大文件
    const largeFiles = sortedAssets.filter((a) => a.size > this.options.threshold);

    // 输出报告
    this.printReport(sortedAssets, totalSize, largeFiles);
  }

  /**
   * 打印报告
   */
  printReport(assets, totalSize, largeFiles) {
    console.log('\n');
    console.log('📦 ' + this.colorize('打包体积报告', 'cyan'));
    console.log('');
    console.log('─'.repeat(70));

    // 表头
    console.log(
      this.padLeft('Size', 12) +
      ' │ ' +
      this.padLeft('%', 6) +
      ' │ ' +
      'File'
    );
    console.log('─'.repeat(70));

    // 详细列表
    if (this.options.showDetails) {
      assets.forEach((asset) => {
        const sizeStr = this.formatSize(asset.size);
        const percentage = ((asset.size / totalSize) * 100).toFixed(1);
        const isLarge = asset.size > this.options.threshold;

        const line =
          this.padLeft(sizeStr, 12) +
          ' │ ' +
          this.padLeft(percentage + '%', 6) +
          ' │ ' +
          asset.name;

        if (isLarge) {
          console.log(this.colorize(line + ' ⚠️', 'yellow'));
        } else {
          console.log(line);
        }
      });

      console.log('─'.repeat(70));
    }

    // 总计
    console.log('');
    console.log(
      this.colorize(`Total Size: ${this.formatSize(totalSize)}`, 'green')
    );
    console.log(`Total Files: ${assets.length}`);

    // 阈值警告
    if (largeFiles.length > 0) {
      console.log('');
      console.log(
        this.colorize(
          `⚠️  警告: ${largeFiles.length} 个文件超过 ${this.formatSize(this.options.threshold)} 阈值:`,
          'yellow'
        )
      );

      largeFiles.forEach((file) => {
        console.log(
          this.colorize(
            `   - ${file.name} (${this.formatSize(file.size)})`,
            'yellow'
          )
        );
      });

      console.log('');
      console.log('建议:');
      console.log('  1. 检查是否有未 Tree-shaking 的依赖');
      console.log('  2. 考虑代码分割 (Code Splitting)');
      console.log('  3. 检查是否有意外打包的大型资源');
    } else {
      console.log('');
      console.log(this.colorize('✓ 所有文件体积正常', 'green'));
    }

    console.log('');
  }

  /**
   * 格式化文件大小
   * @param {number} bytes - 字节数
   * @returns {string} - 格式化的大小字符串
   */
  formatSize(bytes) {
    if (bytes < 1024) {
      return bytes + ' B';
    }
    if (bytes < 1024 * 1024) {
      return (bytes / 1024).toFixed(2) + ' KB';
    }
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  /**
   * 左侧填充
   */
  padLeft(str, length) {
    return String(str).padStart(length);
  }

  /**
   * 右侧填充
   */
  padRight(str, length) {
    return String(str).padEnd(length);
  }

  /**
   * 终端颜色输出
   * @param {string} text - 文本
   * @param {string} color - 颜色名称
   */
  colorize(text, color) {
    const colors = {
      red: '\x1b[31m',
      green: '\x1b[32m',
      yellow: '\x1b[33m',
      blue: '\x1b[34m',
      magenta: '\x1b[35m',
      cyan: '\x1b[36m',
      reset: '\x1b[0m'
    };

    return colors[color] + text + colors.reset;
  }
}

module.exports = BundleSizeReportPlugin;

