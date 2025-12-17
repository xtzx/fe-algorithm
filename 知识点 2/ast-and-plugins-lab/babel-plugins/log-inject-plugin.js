/**
 * Babel 插件：日志注入
 *
 * 功能：为所有 track() 函数调用自动注入文件名参数
 *
 * 转换示例：
 *   转换前: track('click');
 *   转换后: track('click', { __source: 'button.js' });
 *
 *   转换前: track('pageview', { page: '/home' });
 *   转换后: track('pageview', { page: '/home', __source: 'home.js' });
 *
 * 使用方法：
 *   npx babel input.js --plugins ./log-inject-plugin.js --out-file output.js
 */

module.exports = function ({ types: t }) {
  return {
    // 插件名称，用于调试和错误信息
    name: 'log-inject-plugin',

    // visitor 对象：定义要访问的 AST 节点类型
    visitor: {
      /**
       * 访问所有函数调用表达式
       * @param {NodePath} path - 节点路径对象
       * @param {Object} state - 插件状态，包含文件信息等
       */
      CallExpression(path, state) {
        const { node } = path;

        // ==========================================
        // Step 1: 判断是否是 track() 调用
        // ==========================================

        // 检查 callee 是否是 Identifier 类型且名称为 'track'
        if (!t.isIdentifier(node.callee, { name: 'track' })) {
          return;
        }

        // ==========================================
        // Step 2: 获取当前文件名
        // ==========================================

        // state.filename 包含完整路径，提取文件名
        const filename = state.filename || 'unknown';
        const shortFilename = filename.split('/').pop() || filename.split('\\').pop() || 'unknown';

        // ==========================================
        // Step 3: 检查是否已经注入过
        // ==========================================

        // 避免重复注入（可选的安全检查）
        const args = node.arguments;

        // 检查现有参数是否已包含 __source
        if (args.length >= 2 && t.isObjectExpression(args[1])) {
          const hasSource = args[1].properties.some(
            (prop) =>
              t.isObjectProperty(prop) &&
              t.isIdentifier(prop.key, { name: '__source' })
          );
          if (hasSource) {
            return; // 已注入，跳过
          }
        }

        // ==========================================
        // Step 4: 创建 __source 属性节点
        // ==========================================

        const sourceProperty = t.objectProperty(
          t.identifier('__source'),
          t.stringLiteral(shortFilename)
        );

        // ==========================================
        // Step 5: 根据参数情况处理
        // ==========================================

        if (args.length === 0) {
          // 情况 1: track() - 无参数
          // 添加空字符串和 source 对象
          args.push(
            t.stringLiteral(''),
            t.objectExpression([sourceProperty])
          );
        } else if (args.length === 1) {
          // 情况 2: track('event') - 只有事件名
          // 添加第二个对象参数
          args.push(t.objectExpression([sourceProperty]));
        } else if (t.isObjectExpression(args[1])) {
          // 情况 3: track('event', { data }) - 第二个参数是对象
          // 在对象中添加属性
          args[1].properties.push(sourceProperty);
        } else {
          // 情况 4: track('event', otherValue) - 第二个参数不是对象
          // 添加第三个参数（这种情况较少见）
          args.push(t.objectExpression([sourceProperty]));
        }
      }
    }
  };
};

