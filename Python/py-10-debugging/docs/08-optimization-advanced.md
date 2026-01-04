# 高级性能优化

> Cython、Numba、C 扩展与 JIT 编译

## 优化层次

```
┌─────────────────────────────────────────┐
│          算法优化（最重要）               │
├─────────────────────────────────────────┤
│          数据结构优化                     │
├─────────────────────────────────────────┤
│          Python 层面优化                  │
├─────────────────────────────────────────┤
│          NumPy/Pandas 向量化             │
├─────────────────────────────────────────┤
│          Cython/Numba/C 扩展             │
└─────────────────────────────────────────┘
```

---

## NumPy 向量化

### 避免 Python 循环

```python
import numpy as np

# ❌ Python 循环（慢）
def python_sum(arr):
    total = 0
    for x in arr:
        total += x * x
    return total

# ✅ NumPy 向量化（快 100x+）
def numpy_sum(arr):
    return np.sum(arr ** 2)

# 性能对比
arr = np.random.rand(1_000_000)
# python_sum: ~500ms
# numpy_sum:  ~2ms
```

### 广播机制

```python
import numpy as np

# ❌ 手动循环
def normalize_python(matrix):
    result = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i, j] = (matrix[i, j] - matrix.mean()) / matrix.std()
    return result

# ✅ 广播（快 1000x+）
def normalize_numpy(matrix):
    return (matrix - matrix.mean()) / matrix.std()
```

### 使用 np.where 代替条件

```python
import numpy as np

# ❌ Python 循环
def clip_python(arr, min_val, max_val):
    return [min(max(x, min_val), max_val) for x in arr]

# ✅ NumPy 向量化
def clip_numpy(arr, min_val, max_val):
    return np.clip(arr, min_val, max_val)

# 或使用 np.where
def clip_where(arr, min_val, max_val):
    result = np.where(arr < min_val, min_val, arr)
    return np.where(result > max_val, max_val, result)
```

---

## Numba JIT 编译

### 安装

```bash
pip install numba
```

### 基础用法

```python
from numba import jit
import numpy as np

# @jit 装饰器
@jit(nopython=True)  # nopython=True 强制编译模式
def numba_sum(arr):
    total = 0.0
    for x in arr:
        total += x * x
    return total

# 第一次调用触发编译（稍慢）
# 之后调用接近 C 速度
arr = np.random.rand(1_000_000)
result = numba_sum(arr)  # 首次: ~100ms, 之后: ~2ms
```

### 并行化

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def parallel_sum(arr):
    total = 0.0
    for i in prange(len(arr)):  # prange 并行
        total += arr[i] ** 2
    return total

# 多核加速
arr = np.random.rand(10_000_000)
result = parallel_sum(arr)
```

### 类型签名

```python
from numba import jit, float64, int64

# 指定类型加速编译
@jit(float64(float64[:]), nopython=True)
def typed_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total
```

### Numba 限制

```python
# ❌ 不支持的操作
@jit(nopython=True)
def bad_example():
    import json  # 不能在 jit 函数内导入
    return {"key": "value"}  # 不支持任意 dict

# ✅ 支持的数据类型
# - 数值类型（int, float, complex）
# - NumPy 数组
# - 元组（固定类型）
# - 简单类（使用 @jitclass）
```

---

## Cython

### 安装

```bash
pip install cython
```

### 基础用法

创建 `fast_sum.pyx`:

```cython
# fast_sum.pyx
def python_sum(arr):
    """纯 Python 版本"""
    total = 0.0
    for x in arr:
        total += x * x
    return total

def cython_sum(double[:] arr):  # 类型声明
    """Cython 优化版本"""
    cdef double total = 0.0
    cdef int i
    cdef int n = arr.shape[0]

    for i in range(n):
        total += arr[i] * arr[i]

    return total
```

### 编译

创建 `setup.py`:

```python
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fast_sum.pyx"),
    include_dirs=[np.get_include()]
)
```

```bash
python setup.py build_ext --inplace
```

### 使用

```python
import numpy as np
from fast_sum import cython_sum

arr = np.random.rand(1_000_000)
result = cython_sum(arr)  # 接近 C 速度
```

### Cython 优化技巧

```cython
# 1. 关闭边界检查
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sum(double[:] arr):
    cdef double total = 0.0
    cdef Py_ssize_t i
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

# 2. 使用 memoryview
def process_2d(double[:, :] matrix):
    cdef Py_ssize_t i, j
    cdef double total = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total += matrix[i, j]
    return total

# 3. 并行 (使用 OpenMP)
from cython.parallel import prange

@cython.boundscheck(False)
def parallel_sum(double[:] arr):
    cdef double total = 0.0
    cdef Py_ssize_t i
    for i in prange(arr.shape[0], nogil=True):
        total += arr[i]
    return total
```

---

## C 扩展

### 使用 ctypes

```python
# 调用已有的 C 库
import ctypes

# 加载库
libc = ctypes.CDLL("libc.so.6")  # Linux
# libc = ctypes.CDLL("libc.dylib")  # macOS

# 定义参数和返回类型
libc.strlen.argtypes = [ctypes.c_char_p]
libc.strlen.restype = ctypes.c_size_t

# 调用
length = libc.strlen(b"Hello, World!")
print(length)  # 13
```

### 使用 cffi

```python
from cffi import FFI

ffi = FFI()

# 声明 C 函数
ffi.cdef("""
    double sqrt(double x);
""")

# 加载库
lib = ffi.dlopen(None)  # 加载默认 C 库

# 调用
result = lib.sqrt(16.0)
print(result)  # 4.0
```

### 使用 pybind11

创建 `example.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double sum_array(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);

    double total = 0.0;
    for (size_t i = 0; i < buf.size; i++) {
        total += ptr[i];
    }
    return total;
}

PYBIND11_MODULE(example, m) {
    m.def("sum_array", &sum_array, "Sum array elements");
}
```

---

## 性能对比

```python
import numpy as np
import timeit

arr = np.random.rand(1_000_000)

# Python 循环
def python_sum(arr):
    return sum(x * x for x in arr)

# NumPy
def numpy_sum(arr):
    return np.sum(arr ** 2)

# Numba
from numba import jit
@jit(nopython=True)
def numba_sum(arr):
    total = 0.0
    for x in arr:
        total += x * x
    return total

# 预热 Numba
numba_sum(arr)

# 测试
print(f"Python: {timeit.timeit(lambda: python_sum(arr), number=10):.3f}s")
print(f"NumPy:  {timeit.timeit(lambda: numpy_sum(arr), number=10):.3f}s")
print(f"Numba:  {timeit.timeit(lambda: numba_sum(arr), number=10):.3f}s")

# 典型结果:
# Python: 2.500s
# NumPy:  0.020s  (125x 加速)
# Numba:  0.015s  (167x 加速)
```

---

## 选择指南

| 场景 | 推荐方案 |
|------|---------|
| 简单数组操作 | NumPy |
| 复杂循环无法向量化 | Numba |
| 需要与 C 代码交互 | pybind11 / cffi |
| 需要最大性能 | Cython |
| 快速原型验证 | Numba |
| 生产环境稳定性 | Cython |

### 决策树

```
需要优化?
├── 是否可以向量化?
│   ├── 是 → NumPy
│   └── 否 → 是否纯数值计算?
│       ├── 是 → Numba
│       └── 否 → 是否需要 C 库?
│           ├── 是 → ctypes/cffi/pybind11
│           └── 否 → Cython
```

---

## PyPy

### 简介

PyPy 是 Python 的替代解释器，内置 JIT 编译器。

```bash
# 安装
# macOS
brew install pypy3

# 使用 PyPy 运行
pypy3 your_script.py
```

### 性能提升

```python
# fibonacci.py
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(35))

# CPython: ~3.5s
# PyPy:    ~0.1s (35x 加速)
```

### PyPy 限制

- 不支持部分 C 扩展（如 NumPy 需要 numpy-pypy）
- 启动较慢（JIT 预热）
- 内存占用较高

---

## 实战：图像处理优化

```python
import numpy as np
from numba import jit, prange

# ❌ Python 版本（慢）
def blur_python(image, kernel_size=3):
    h, w = image.shape
    result = np.zeros_like(image)
    pad = kernel_size // 2

    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            total = 0.0
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    total += image[i + ki, j + kj]
            result[i, j] = total / (kernel_size ** 2)

    return result

# ✅ Numba 版本（快 100x+）
@jit(nopython=True, parallel=True)
def blur_numba(image, kernel_size=3):
    h, w = image.shape
    result = np.zeros_like(image)
    pad = kernel_size // 2

    for i in prange(pad, h - pad):
        for j in range(pad, w - pad):
            total = 0.0
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    total += image[i + ki, j + kj]
            result[i, j] = total / (kernel_size ** 2)

    return result

# ✅ NumPy + SciPy 版本
from scipy.ndimage import uniform_filter

def blur_scipy(image, kernel_size=3):
    return uniform_filter(image, size=kernel_size)
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| Numba 首次调用慢 | JIT 编译需要时间 | 预热或缓存编译 |
| NumPy 复制 vs 视图 | arr[::2] 是视图，arr[[0,1]] 是复制 | 了解内存布局 |
| Cython 忘记类型声明 | 没有加速 | 声明所有变量类型 |
| 过度优化 | 代码可读性下降 | 只优化瓶颈 |

---

## 小结

1. **NumPy 向量化**：首选方案，简单高效
2. **Numba**：无法向量化时使用，零配置
3. **Cython**：需要最大性能时使用
4. **C 扩展**：与现有 C 代码交互
5. **PyPy**：纯 Python 代码的简单加速
6. **原则**：先测量，只优化瓶颈

