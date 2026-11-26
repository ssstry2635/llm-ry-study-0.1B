# 第14章示例代码：NumPy
# 运行方式： python code/ch14_numpy.py
# 前置： python -m pip install numpy

import time

import numpy as np

# ---------- 向量化加速对比 ----------
n = 1_000_000
t0 = time.time()
c = [x + y for x, y in zip(range(n), range(n))]
t1 = time.time()
a, b = np.arange(n), np.arange(n)
t2 = time.time()
c = a + b
t3 = time.time()
print(f"纯Python: {t1 - t0:.3f}s | NumPy: {t3 - t2:.4f}s")

# ---------- 创建与属性 ----------
rng = np.random.default_rng(42)
print(np.zeros(3), np.ones((2, 3)))
print(np.arange(0, 10, 2), np.linspace(0, 1, 5))
print(rng.normal(0, 1, (2, 2)))

m = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
print(m.shape, m.ndim, m.dtype, m.size)
print(m.reshape(3, 2))
print(m.reshape(-1))
print(m.T)

# ---------- 索引切片 ----------
grid = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(grid[0, 1], grid[1], grid[:, 0])
print(grid[0:2, 1:], grid[-1], sep="\n")

# ---------- 布尔索引 ----------
scores = np.array([0.9, 0.2, 0.7, 0.4])
print(scores[scores > 0.5], (scores > 0.5).sum(), np.where(scores > 0.5)[0])

# ---------- 广播 ----------
a2 = np.array([[1, 2, 3], [4, 5, 6]])
print(a2 + np.array([10, 20, 30]))
print(a2 + np.array([[100], [200]]))

# ---------- 数学运算 ----------
v = np.array([1.0, 2.0, 3.0])
print(v + 1, v * 2, v ** 2)
print(np.exp(v), np.sqrt(v), np.log(v))

# ---------- 矩阵乘法 ----------
W = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])            # (3,2)
xv = np.array([1.0, 1.0])            # (2,)
print("W @ xv =", W @ xv)            # [ 3.  7. 11.]

x = rng.normal(0, 1, (4, 2))         # (4,2) 4个样本，每个2个特征
print("x shape:", x.shape, "W shape:", W.shape)
# x @ W 会报错: (4,2)@(3,2) 中间的 2 和 3 对不上；要用 W.T (2,3)
print("x @ W.T shape:", (x @ W.T).shape)   # (4,3) —— 4个样本→3个输出

W2 = rng.normal(0, 0.1, (2, 3))      # (2,3) 随机初始化的权重
b = np.zeros(3)                      # (3,)  偏置
print("前向传播 y = x @ W2 + b:")
print(x @ W2 + b)                    # (4,3)

# ---------- 聚合与axis ----------
print(m.sum(), m.mean(), m.max(), m.argmax())
print(m.sum(axis=0), m.sum(axis=1))
print(m.mean(axis=1, keepdims=True))

# ---------- 实战：softmax / normalize ----------
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
probs = softmax(logits)
print(probs)
print("每行和:", probs.sum(axis=-1))       # [1. 1.]

def normalize(x):
    return (x - x.mean()) / x.std()

z = normalize(np.array([1, 2, 3, 4, 5], dtype=float))
print("归一化后均值:", z.mean())            # 0.0

# ---------- 练习答案 ----------
a = np.array([3, -1, 4, -1, 5])
a[a < 0] = 0
print(a)
