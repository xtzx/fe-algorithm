/**
 * API 模块
 * 示例 JavaScript 文件
 */

// 配置
const API_BASE_URL = 'https://api.example.com';
const DEFAULT_TIMEOUT = 5000;

/**
 * 发送 GET 请求
 * @param {string} path - API 路径
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} - 响应数据
 */
async function get(path, options = {}) {
    const url = `${API_BASE_URL}${path}`;
    const response = await fetch(url, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
}

/**
 * 发送 POST 请求
 * @param {string} path - API 路径
 * @param {Object} data - 请求数据
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} - 响应数据
 */
async function post(path, data, options = {}) {
    const url = `${API_BASE_URL}${path}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        body: JSON.stringify(data),
        ...options,
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
}

/*
 * 用户相关 API
 */

async function getUsers() {
    return get('/users');
}

async function getUser(id) {
    return get(`/users/${id}`);
}

async function createUser(userData) {
    return post('/users', userData);
}

// 导出
export { get, post, getUsers, getUser, createUser };

