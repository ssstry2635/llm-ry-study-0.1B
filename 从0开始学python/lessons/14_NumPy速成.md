# 第 14 章：NumPy 速成——通往张量的桥梁

> 本章目标：掌握 ndarray 的创建、索引、广播、矩阵运算。**PyTorch 的 Tensor 就是"能上 GPU 的 NumPy 数组"**，API 几乎同名。这一章学好了，PyTorch 的张量操作直接白送。

## 0. 安装与导入

```
python -m pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
import numpy as np        # 全世界统一的别名
```

## 1. 为什么要 NumPy

用纯 Python 算 100 万个数的加法要写循环；NumPy 把整个数组当成**一个对象**交给底层 C 代码批量算（**向量化**），快 50~100 倍：

```python
import time

n = 1_000_000
a_list = list(range(n))
b_list = list(range(n))

t0 = time.time()
c = [x + y for x, y in zip(a_list, b_list)]      # 纯Python
t1 = time.time()

a = np.arange(n)                                  # NumPy数组
b = np.arange(n)
t2 = time.time()
c = a + b                                         # 一个+号，整组相加！
t3 = time.time()

print(f"纯Python: {t1 - t0:.3f}s | NumPy: {t3 - t2:.4f}s")
```

**从此换脑子：能对整组数一起做的运算，绝不写 for 循环。**

## 2. 创建数组

```python
a = np.array([1, 2, 3])                 # 列表 → 数组
m = np.array([[1.0, 2.0], [3.0, 4.0]])  # 嵌套列表 → 矩阵（二维）

print(np.zeros(3))                      # [0. 0. 0.]
print(np.ones((2, 3)))                  # 2行3列全1
print(np.arange(0, 10, 2))              # [0 2 4 6 8]   类似range
print(np.linspace(0, 1, 5))             # [0.   0.25 0.5  0.75 1.  ]  均分5点

rng = np.random.default_rng(42)         # 推荐的随机数生成器（42=种子，可复现）
print(rng.random((2, 2)))               # 0~1均匀随机
print(rng.normal(0, 1, (2, 2)))         # 正态随机（均值0,标准差1）——初始化权重用它！
```

## 3. 属性：shape / dtype / ndim（重中之重）

```python
m = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

print(m.shape)      # (2, 3)   形状：2行3列 ← 记住这个词，Tensor第一属性
print(m.ndim)       # 2        维度个数
print(m.dtype)      # float64  元素类型（整个数组只能装一种类型！）
print(m.size)       # 6        元素总数
print(len(m))       # 2        第一维的长度

# 改形状（元素总数必须匹配）
print(m.reshape(3, 2))     # 变成3行2列
print(m.reshape(-1))       # -1=自动推算 → 拍平成1维
print(m.T)                 # 转置（行变列）
```

**深度学习里 90% 的形状 bug，就是 shape 对不上**。看到张量先 print 它的 `.shape`。

## 4. 索引与切片（和列表一样的语法，升维的用法）

```python
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(m[0, 1])        # 2         第0行第1列（一对方括号里逗号分隔）
print(m[1])           # [4 5 6]   整行
print(m[:, 0])        # [1 4 7]   整列！  : 代表"这一维全要"
print(m[0:2, 1:])     # [[2 3] \n [5 6]]  每个维度都可以切片
print(m[-1])          # [7 8 9]   最后一行
```

`m[:, 0]` 这种"冒号+逗号"写法在 PyTorch 里一模一样，如 `logits[:, -1, :]`（取每个序列最后一个位置的输出）。

### 布尔索引：按条件筛选（超实用）

```python
scores = np.array([0.9, 0.2, 0.7, 0.4])
mask = scores > 0.5                # [True False True False]
print(mask)
print(scores[mask])                # [0.9 0.7]   挑出满足条件的
print((scores > 0.5).sum())        # 2           True当1数——数一数有几个满足
print(np.where(scores > 0.5)[0])   # [0 2]       满足条件的下标
```

## 5. 广播（Broadcasting）：不同形状怎么一起算

**规则：从右往左逐维比较，两个维度要么相等，要么其中一个是 1（或缺失），就能自动"复制扩展"对齐。**

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)
b = np.array([10, 20, 30])      # shape (3,)

print(a + b)
# [[11 22 33]      b被"广播"成两行，分别加到每一行上
#  [14 25 36]]

c = np.array([[100], [200]])    # shape (2, 1)
print(a + c)
# [[101 102 103]   c被广播成(2,3)
#  [204 205 206]]
```

最常用场景——**给每一行/每一列整体做运算**：全数据减均值（`x - x.mean()`）、softmax 里减最大值（见第 8 节）。
shape 不满足规则就报错，看懂规则就知道怎么改。

## 6. 数学运算

### 逐元素运算（同形状，一一对应）

```python
a = np.array([1.0, 2.0, 3.0])
print(a + 1, a * 2)         # 和标量：广播
print(a ** 2)               # [1. 4. 9.]
print(np.exp(a))            # e^x        ← softmax 要用
print(np.sqrt(a))
print(np.log(a))            # 自然对数   ← 交叉熵loss要用
```

### 矩阵乘法（神经网络的命根子）

```python
W = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])      # shape (3, 2)
x = np.array([1.0, 1.0])        # shape (2,)    输入向量

print(W @ x)                    # [ 3.  7. 11.]   @ = 矩阵乘法
# 检查: 第0行 1*1+2*1=3, 第1行 3*1+4*1=7, 第2行 5*1+6*1=11 —— 每行和x做点积
```

⚠️ **矩阵乘法规则：`(m,n) @ (n,p) = (m,p)`，中间的两个 n 必须相等**。
`A @ B` 等价 `np.dot(A, B)`；`*` 是逐元素乘（不是矩阵乘！），别混淆。

```python
X = np.array([[1.0, 2.0],
              [3.0, 4.0]])      # (2,2)  两个样本
print(X @ W)                    # (2,2)@(3,2)? ❌ 报错！中间2和3对不上
print(X @ W.T)                  # (2,2)@(2,3) = (2,3) ✅ 每行样本得到3个输出
```

**"X @ W.T 还是 X @ W"是新手常纠结点：先 print 两边的 shape，对上中间维度再写。**

## 7. 聚合运算与 axis

```python
m = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])      # shape (2, 3)

print(m.sum())          # 21.0   全部加起来
print(m.mean())         # 3.5
print(m.max(), m.min())
print(m.argmax())       # 5      最大值展平后的下标

# axis：沿哪个方向压扁（记住：axis=消掉哪个维度）
print(m.sum(axis=0))    # [5. 7. 9.]    消掉行 → 每列求和（得到shape(3,)）
print(m.sum(axis=1))    # [ 6. 15.]     消掉列 → 每行求和（得到shape(2,)）
print(m.mean(axis=1, keepdims=True))   # [[2.],[5.]]  keepdims保持维度（利于广播）
```

axis 记法：**shape (2,3) 沿 axis=0 求和后变 (3,)；沿 axis=1 变 (2,)**——结果的 shape 告诉你对不对。

## 8. 综合实战：手写 softmax 和归一化

```python
def softmax(x):
    """把向量变成概率分布（和为1）。x可以是(批量, 类别数)的矩阵"""
    x = x - x.max(axis=-1, keepdims=True)     # 减最大值防爆（广播！）
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)  # 广播相除

logits = np.array([[2.0, 1.0, 0.1],
                   [1.0, 3.0, 0.2]])
probs = softmax(logits)
print(probs)                    # 每行加起来=1
print(probs.sum(axis=-1))       # [1. 1.]  ✅

def normalize(x):
    """减均值除标准差——数据预处理的标配"""
    return (x - x.mean()) / x.std()
```

这两个函数你会在大模型课程里反复见到。**看懂它们=本章毕业。**

## 9. 大模型视角 🤖

| NumPy | PyTorch 对应 | 用途 |
|---|---|---|
| `np.array` | `torch.tensor` | 建张量 |
| `.shape` / `.reshape` / `.T` | 同名 | 形状操作 |
| `@` 矩阵乘 | 同名 | 线性层 `y = xW + b` |
| `rng.normal` 初始化 | `nn.init` | 随机初始化权重 |
| `axis` 聚合 | `dim` 参数 | loss 对 batch 求平均 |
| 布尔索引 | 同名 | 取 mask、过滤 |

```python
# 一个"线性层"在 NumPy 里的完整样子（第15章会训练它！）
W = rng.normal(0, 0.1, (in_dim, out_dim))    # 权重
b = np.zeros(out_dim)                        # 偏置
y = x @ W + b                                # 前向传播，就这么短
```

## 本章小结

- 向量化思想：整组一起算，不写 for
- 创建：`array / zeros / arange / linspace / rng.normal`；属性：`shape / dtype`
- 索引 `m[i, j]`、切片 `m[:, 0]`、布尔索引 `x[mask]`
- 广播：右边逐维对齐，1 或缺失可扩展
- `@` 矩阵乘看中间维度；聚合用 axis（消掉哪个维度）

## 练习

1. 创建 5~10（含）的整数数组；算它的均值和方差（`np.var`）。
2. `m = np.arange(12).reshape(3, 4)`：取出第 2 行、第 1 列、右下角的 2×2 子矩阵。
3. 用布尔索引把 `np.array([3, -1, 4, -1, 5])` 里的负数替换成 0（提示：`a[a<0]=0`）。
4. 验证矩阵乘法形状：随机造 `A(4,3)`、`B(3,2)`，算 `A@B` 并 print 其 shape。
5. 手写 `normalize`（减均值除标准差）处理 `[1,2,3,4,5]`，结果均值应为 0（用 `.mean()` 验证，约等于 0 即可）。

---

### 参考答案

```python
import numpy as np

# 1
a = np.arange(5, 11)
print(a, a.mean(), a.var())

# 2
m = np.arange(12).reshape(3, 4)
print(m[1], m[:, 1], m[1:, 2:])

# 3
a = np.array([3, -1, 4, -1, 5])
a[a < 0] = 0
print(a)                     # [3 0 4 0 5]

# 4
rng = np.random.default_rng(0)
A = rng.random((4, 3))
B = rng.random((3, 2))
C = A @ B
print(C.shape)               # (4, 2)

# 5
def normalize(x):
    return (x - x.mean()) / x.std()
z = normalize(np.array([1, 2, 3, 4, 5], dtype=float))
print(z.mean())              # 0.0（浮点误差级别）
```
