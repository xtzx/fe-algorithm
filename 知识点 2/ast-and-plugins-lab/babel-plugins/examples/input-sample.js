/**
 * 示例输入文件
 *
 * 这个文件展示 Babel 插件转换前的代码
 * 运行 babel 转换后，可以对比 output-sample.js 查看效果
 */

// ============================================
// 示例 1: track() 日志注入
// ============================================

// 简单调用
track('page_view');

// 带数据的调用
track('button_click', { buttonId: 'submit' });

// 带多个参数
track('form_submit', { formId: 'login' });

// 在函数中调用
function handleClick() {
  track('click');
  console.log('clicked');
}

// 在箭头函数中调用
const handleHover = () => {
  track('hover', { target: 'menu' });
};

// 在类方法中调用
class Analytics {
  trackPageView() {
    track('page_view', { url: window.location.href });
  }
}


// ============================================
// 示例 2: @log 装饰器转换
// ============================================

// 简单的 log 装饰器定义（仅用于演示，实际会被移除）
function log(target, key, descriptor) {
  return descriptor;
}

// 使用装饰器的类
class UserService {
  @log
  getUser(id) {
    return { id, name: 'Alice' };
  }

  @log
  updateUser(id, data) {
    console.log('Updating user', id, data);
    return { success: true };
  }

  // 没有装饰器的方法
  deleteUser(id) {
    return { deleted: true };
  }
}

// 带装饰器的另一个类
class OrderService {
  @log
  createOrder(items) {
    return {
      orderId: Date.now(),
      items
    };
  }
}


// ============================================
// 导出
// ============================================

export { handleClick, handleHover, Analytics, UserService, OrderService };

