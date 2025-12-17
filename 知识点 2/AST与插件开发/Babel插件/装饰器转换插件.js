/**
 * Babel 插件：自定义装饰器转换
 *
 * 功能：将简单的 @log 装饰器转换为等价的 console.log 调用
 *
 * 转换示例：
 *   转换前:
 *     class MyClass {
 *       @log
 *       myMethod() {
 *         return 'hello';
 *       }
 *     }
 *
 *   转换后:
 *     class MyClass {
 *       myMethod() {
 *         console.log('[MyClass.myMethod] called with args:', arguments);
 *         return 'hello';
 *       }
 *     }
 *
 * 注意：需要配合 @babel/plugin-proposal-decorators 使用
 *
 * 使用方法：
 *   需要先解析装饰器语法，然后再用此插件转换
 */

module.exports = function ({ types: t }) {
  return {
    name: 'custom-decorator-transform',

    visitor: {
      /**
       * 访问类方法节点
       */
      ClassMethod(path) {
        const { node } = path;

        // ==========================================
        // Step 1: 检查是否有装饰器
        // ==========================================

        const decorators = node.decorators;
        if (!decorators || decorators.length === 0) {
          return;
        }

        // ==========================================
        // Step 2: 查找 @log 装饰器
        // ==========================================

        const logDecoratorIndex = decorators.findIndex((decorator) => {
          // 处理简单标识符 @log
          if (t.isIdentifier(decorator.expression, { name: 'log' })) {
            return true;
          }
          // 处理调用形式 @log()
          if (
            t.isCallExpression(decorator.expression) &&
            t.isIdentifier(decorator.expression.callee, { name: 'log' })
          ) {
            return true;
          }
          return false;
        });

        if (logDecoratorIndex === -1) {
          return;
        }

        // ==========================================
        // Step 3: 移除 @log 装饰器
        // ==========================================

        decorators.splice(logDecoratorIndex, 1);

        // 如果没有其他装饰器了，清空数组
        if (decorators.length === 0) {
          node.decorators = null;
        }

        // ==========================================
        // Step 4: 获取方法信息
        // ==========================================

        const methodName = t.isIdentifier(node.key)
          ? node.key.name
          : '[computed]';

        // 尝试获取类名
        const classPath = path.findParent((p) => p.isClassDeclaration() || p.isClassExpression());
        const className = classPath && classPath.node.id
          ? classPath.node.id.name
          : 'Anonymous';

        // ==========================================
        // Step 5: 创建日志语句
        // ==========================================

        const logMessage = `[${className}.${methodName}] called`;

        // 创建 console.log 调用
        // console.log('[ClassName.methodName] called with args:', arguments)
        const logStatement = t.expressionStatement(
          t.callExpression(
            t.memberExpression(
              t.identifier('console'),
              t.identifier('log')
            ),
            [
              t.stringLiteral(logMessage + ' with args:'),
              t.identifier('arguments')
            ]
          )
        );

        // ==========================================
        // Step 6: 在方法体开头插入日志
        // ==========================================

        // 使用 unshiftContainer 在方法体 body 数组开头插入
        path.get('body').unshiftContainer('body', logStatement);
      },

      /**
       * 同时处理类属性方法（箭头函数形式）
       */
      ClassProperty(path) {
        const { node } = path;

        // 检查是否是箭头函数
        if (!t.isArrowFunctionExpression(node.value)) {
          return;
        }

        // 检查装饰器
        const decorators = node.decorators;
        if (!decorators || decorators.length === 0) {
          return;
        }

        // 查找 @log 装饰器
        const logDecoratorIndex = decorators.findIndex((decorator) =>
          t.isIdentifier(decorator.expression, { name: 'log' })
        );

        if (logDecoratorIndex === -1) {
          return;
        }

        // 移除装饰器
        decorators.splice(logDecoratorIndex, 1);
        if (decorators.length === 0) {
          node.decorators = null;
        }

        // 获取属性名
        const propertyName = t.isIdentifier(node.key)
          ? node.key.name
          : '[computed]';

        // 创建日志语句
        const logStatement = t.expressionStatement(
          t.callExpression(
            t.memberExpression(
              t.identifier('console'),
              t.identifier('log')
            ),
            [t.stringLiteral(`[${propertyName}] called`)]
          )
        );

        // 处理箭头函数体
        const arrowFn = node.value;

        if (t.isBlockStatement(arrowFn.body)) {
          // 已经是块语句，直接插入
          arrowFn.body.body.unshift(logStatement);
        } else {
          // 表达式体，需要转换为块语句
          const returnStatement = t.returnStatement(arrowFn.body);
          arrowFn.body = t.blockStatement([logStatement, returnStatement]);
        }
      }
    }
  };
};

