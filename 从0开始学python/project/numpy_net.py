"""
毕业项目（NumPy 版）：用矩阵运算重写神经网络，一次处理整个 batch。

任务：让网络学会函数  y = sin(x)   （x 在 [-π, π] 之间）—— 比 x² 更难
结构：1 → 32 → 1，中间用 tanh 激活
训练：全批量梯度下降（每步用全部 200 个样本算梯度）

运行：  python project/numpy_net.py
前置：  python -m pip install numpy

与 pure_python_net.py 对比着看：
  * 类的结构完全一样（Linear / Tanh / Sequential / MSELoss / SGD）
  * 所有 for 循环都变成了矩阵运算 —— 这就是"向量化"
  * 输入从"一个样本"变成"一批样本"，形状是 (N, 特征数)
"""

import numpy as np


# ============================================================
# 1. 网络层
# ============================================================
class Linear:
    """全连接层  Y = X·W + b     X:(N,in)  W:(in,out)  b:(out,)  Y:(N,out)"""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        self.W = rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b                 # 广播：b 自动加到每一行

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out:(N,out)  →  返回 dLoss/dX:(N,in)"""
        self.gW = self.x.T @ grad_out              # (in,N)@(N,out) = (in,out)  ← 对比纯Python的双重for
        self.gb = grad_out.sum(axis=0)             # 沿样本维度求和 → (out,)
        return grad_out @ self.W.T                 # (N,out)@(out,in) = (N,in)


class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1 - self.out ** 2)


class ReLU:
    """备选激活函数：max(0, x)。大模型里更常见（试试把 Tanh 换成它！）"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (self.x > 0)             # x>0 处导数为1，否则为0（布尔当0/1用）


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self, x):
        return self.forward(x)


# ============================================================
# 2. 损失函数
# ============================================================
class MSELoss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.diff = pred - target
        return float(np.mean(self.diff ** 2))

    def backward(self) -> np.ndarray:
        return 2 * self.diff / self.diff.size      # 对"平均值"求导要除以元素个数


# ============================================================
# 3. 优化器
# ============================================================
class SGD:
    def __init__(self, layers, lr: float):
        self.layers = [l for l in layers if isinstance(l, Linear)]
        self.lr = lr

    def zero_grad(self):
        for l in self.layers:
            l.gW[...] = 0                          # [...] 原地清零，不新建数组
            l.gb[...] = 0

    def step(self):
        for l in self.layers:
            l.W -= self.lr * l.gW                  # 一行完成整个矩阵的更新
            l.b -= self.lr * l.gb


# ============================================================
# 4. 数据
# ============================================================
def make_data(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-np.pi, np.pi, n).reshape(-1, 1)   # (N,1)  reshape 成"N个样本、每个1个特征"
    y = np.sin(x)                                       # (N,1)
    return x, y


# ============================================================
# 5. 训练循环
# ============================================================
def train(epochs: int = 3000, lr: float = 0.05, hidden: int = 32,
          seed: int = 42, log_every: int = 300, activation=Tanh) -> Sequential:
    rng = np.random.default_rng(seed)
    x, y = make_data(200)

    model = Sequential(Linear(1, hidden, rng), activation(), Linear(hidden, 1, rng))
    loss_fn = MSELoss()
    opt = SGD(model.layers, lr)

    for epoch in range(epochs):
        opt.zero_grad()                     # ① 清零
        pred = model(x)                     # ② 前向：一次算 200 个样本
        loss = loss_fn.forward(pred, y)     # ③ 损失
        model.backward(loss_fn.backward())  # ④ 反向
        opt.step()                          # ⑤ 更新
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} | loss {loss:.6f}")
    return model


if __name__ == "__main__":
    import time

    print("=== NumPy 手写神经网络：学习 y = sin(x) ===")
    t0 = time.time()
    model = train()
    print(f"训练耗时 {time.time() - t0:.2f}s")

    print("\n=== 训练后的预测 vs 真实值 ===")
    test_x = np.array([[-3.0], [-1.5], [0.0], [1.5], [3.0]])
    pred = model(test_x)
    for xi, pi in zip(test_x[:, 0], pred[:, 0]):
        print(f"x={xi:5.2f}  真实={np.sin(xi):7.4f}  预测={pi:7.4f}  误差={abs(pi - np.sin(xi)):.4f}")

    print("\n=== 挑战：把 Tanh 换成 ReLU 会怎样？ ===")
    train(epochs=3000, log_every=1000, activation=ReLU)
