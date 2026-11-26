"""
毕业项目（纯 Python 版）：不借助任何第三方库，从零手写一个"会学习"的神经网络。

任务：让网络学会函数  y = x²   （x 在 [-1, 1] 之间）
结构：1 → 16 → 1，中间用 tanh 激活
训练：随机梯度下降（SGD），一次喂一个样本

运行：  python project/pure_python_net.py

这份代码刻意模仿 PyTorch 的组织方式（Linear / Sequential / MSELoss / SGD、
forward / backward / zero_grad / step），学完它再看 PyTorch 会有强烈的熟悉感。
"""

import math
import random


# ============================================================
# 1. 网络层：每一层都有 forward（算输出）和 backward（算梯度）
# ============================================================
class Linear:
    """全连接层  y = x·W + b     （对应 PyTorch 的 nn.Linear）"""

    def __init__(self, in_dim: int, out_dim: int, rng: random.Random):
        self.in_dim, self.out_dim = in_dim, out_dim
        scale = 1 / math.sqrt(in_dim)                     # 让初始输出不过大（Xavier 风格）
        self.W = [[rng.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
        self.b = [0.0] * out_dim
        self.gW = [[0.0] * out_dim for _ in range(in_dim)]  # W 的梯度，形状与 W 相同
        self.gb = [0.0] * out_dim                           # b 的梯度
        self.x: list[float] = []                            # 缓存输入，反向传播要用

    def forward(self, x: list[float]) -> list[float]:
        self.x = x
        return [
            sum(x[i] * self.W[i][j] for i in range(self.in_dim)) + self.b[j]
            for j in range(self.out_dim)
        ]

    def backward(self, grad_out: list[float]) -> list[float]:
        """输入 dLoss/dy，算出 dLoss/dW 与 dLoss/db（累加），返回 dLoss/dx 传给前一层"""
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                self.gW[i][j] += self.x[i] * grad_out[j]       # 因为 dy_j/dW_ij = x_i
        for j in range(self.out_dim):
            self.gb[j] += grad_out[j]                          # 因为 dy_j/db_j = 1
        return [
            sum(self.W[i][j] * grad_out[j] for j in range(self.out_dim))
            for i in range(self.in_dim)
        ]                                                      # 因为 dy_j/dx_i = W_ij


class Tanh:
    """激活函数 tanh，把数压到 (-1, 1)，给网络加入"非线性"（对应 nn.Tanh）"""

    def forward(self, x: list[float]) -> list[float]:
        self.out = [math.tanh(v) for v in x]
        return self.out

    def backward(self, grad_out: list[float]) -> list[float]:
        return [g * (1 - o * o) for g, o in zip(grad_out, self.out)]   # tanh' = 1 - tanh²


class Sequential:
    """把多层串起来（对应 nn.Sequential）"""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: list[float]) -> list[float]:
        for layer in self.layers:
            x = layer.forward(x)          # 数据从前往后流
        return x

    def backward(self, grad: list[float]) -> list[float]:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)   # 梯度从后往前流——这就是"反向传播"
        return grad

    def __call__(self, x):                # 让 model(x) 等价于 model.forward(x)
        return self.forward(x)


# ============================================================
# 2. 损失函数：衡量"预测离答案有多远"
# ============================================================
class MSELoss:
    """均方误差 (pred - target)² 的平均值（对应 nn.MSELoss）"""

    def forward(self, pred: list[float], target: list[float]) -> float:
        self.diff = [p - t for p, t in zip(pred, target)]
        return sum(d * d for d in self.diff) / len(self.diff)

    def backward(self) -> list[float]:
        n = len(self.diff)
        return [2 * d / n for d in self.diff]      # d(diff²)/d(pred) = 2·diff


# ============================================================
# 3. 优化器：拿着梯度去更新参数
# ============================================================
class SGD:
    """随机梯度下降：参数 -= 学习率 × 梯度  （对应 torch.optim.SGD）"""

    def __init__(self, layers, lr: float):
        self.layers = [l for l in layers if isinstance(l, Linear)]   # 只有 Linear 有参数
        self.lr = lr

    def zero_grad(self):
        """清零梯度。backward 是累加的，每步开始前必须清零！"""
        for l in self.layers:
            for row in l.gW:
                for j in range(len(row)):
                    row[j] = 0.0
            for j in range(len(l.gb)):
                l.gb[j] = 0.0

    def step(self):
        """沿梯度反方向走一小步"""
        for l in self.layers:
            for i in range(l.in_dim):
                for j in range(l.out_dim):
                    l.W[i][j] -= self.lr * l.gW[i][j]
            for j in range(l.out_dim):
                l.b[j] -= self.lr * l.gb[j]


# ============================================================
# 4. 数据
# ============================================================
def make_data(n: int, rng: random.Random) -> tuple[list[float], list[float]]:
    xs = [rng.uniform(-1, 1) for _ in range(n)]
    ys = [x * x for x in xs]                    # 目标函数 y = x²
    return xs, ys


# ============================================================
# 5. 训练循环：所有深度学习代码的心脏
# ============================================================
def train(epochs: int = 1500, lr: float = 0.05, hidden: int = 16,
          seed: int = 42, log_every: int = 100) -> Sequential:
    rng = random.Random(seed)                   # 固定种子，结果可复现
    xs, ys = make_data(40, rng)

    model = Sequential(Linear(1, hidden, rng), Tanh(), Linear(hidden, 1, rng))
    loss_fn = MSELoss()
    opt = SGD(model.layers, lr)

    order = list(range(len(xs)))
    for epoch in range(epochs):
        rng.shuffle(order)                      # 每轮打乱样本顺序
        total_loss = 0.0
        for k in order:
            opt.zero_grad()                     # ① 清零梯度
            pred = model([xs[k]])               # ② 前向传播：算预测
            loss = loss_fn.forward(pred, [ys[k]])  # ③ 算损失
            grad = loss_fn.backward()           # ④ 反向传播：算梯度
            model.backward(grad)
            opt.step()                          # ⑤ 更新参数
            total_loss += loss
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d} | loss {total_loss / len(xs):.6f}")
    return model


if __name__ == "__main__":
    print("=== 纯 Python 手写神经网络：学习 y = x² ===")
    model = train()

    print("\n=== 训练后的预测 vs 真实值 ===")
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        pred = model([x])[0]
        print(f"x={x:5.2f}  真实={x * x:.4f}  预测={pred:.4f}  误差={abs(pred - x * x):.4f}")
