/**
 * 示例输出文件
 *
 * 这是经过 Babel 插件转换后的代码
 * 展示 log-inject-plugin 和 custom-decorator-transform 的效果
 */

// ============================================
// 示例 1: track() 日志注入 - 转换后
// ============================================

// 简单调用 → 添加了 __source 参数
track('page_view', { __source: 'input-sample.js' });

// 带数据的调用 → 在现有对象中添加 __source
track('button_click', { buttonId: 'submit', __source: 'input-sample.js' });

// 带多个参数 → 在现有对象中添加 __source
track('form_submit', { formId: 'login', __source: 'input-sample.js' });

// 在函数中调用 → 同样注入
function handleClick() {
  track('click', { __source: 'input-sample.js' });
  console.log('clicked');
}

// 在箭头函数中调用 → 同样注入
const handleHover = () => {
  track('hover', { target: 'menu', __source: 'input-sample.js' });
};

// 在类方法中调用 → 同样注入
class Analytics {
  trackPageView() {
    track('page_view', { url: window.location.href, __source: 'input-sample.js' });
  }
}


// ============================================
// 示例 2: @log 装饰器转换 - 转换后
// ============================================

// 装饰器定义保留（实际项目中可能会移除）
function log(target, key, descriptor) {
  return descriptor;
}

// 使用装饰器的类 → 装饰器被移除，日志被注入
class UserService {
  // @log 被移除，方法体开头注入了 console.log
  getUser(id) {
    console.log('[UserService.getUser] called with args:', arguments);
    return { id, name: 'Alice' };
  }

  // 同上
  updateUser(id, data) {
    console.log('[UserService.updateUser] called with args:', arguments);
    console.log('Updating user', id, data);
    return { success: true };
  }

  // 没有装饰器的方法 → 保持不变
  deleteUser(id) {
    return { deleted: true };
  }
}

// 另一个类
class OrderService {
  createOrder(items) {
    console.log('[OrderService.createOrder] called with args:', arguments);
    return {
      orderId: Date.now(),
      items
    };
  }
}


// ============================================
// 导出 - 保持不变
// ============================================

export { handleClick, handleHover, Analytics, UserService, OrderService };


/**
 * 总结：转换效果
 *
 * 1. log-inject-plugin:
 *    - 所有 track() 调用都添加了 __source 参数
 *    - 便于追踪埋点来源
 *
 * 2. custom-decorator-transform:
 *    - @log 装饰器被移除
 *    - 方法体开头注入了日志输出
 *    - 日志包含类名、方法名和参数
 */

