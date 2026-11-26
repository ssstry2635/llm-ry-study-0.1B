# 第11章示例代码：面向对象
# 运行方式： python code/ch11_classes.py

import math
from dataclasses import dataclass

# ---------- 第一个类 ----------
class Linear:
    """最简单的线性层: y = w*x + b"""

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        return self.w * x + self.b

    def __call__(self, x):          # 让对象能像函数一样调用
        return self.forward(x)


layer = Linear(2.0, 1.0)
print(layer.w, layer.b, layer.forward(3.0), layer(3.0))   # 2.0 1.0 7.0 7.0

# ---------- 类属性 vs 实例属性 ----------
class Model:
    count = 0

    def __init__(self, name):
        self.name = name
        Model.count += 1

a, b = Model("gpt"), Model("llama")
print(a.name, b.name, Model.count)   # gpt llama 2

# ---------- 魔术方法：Dataset ----------
class Dataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __repr__(self):
        return f"Dataset({len(self)}条样本)"

ds = Dataset(["我爱", "大模型", "训练"])
print(len(ds), ds[1], ds)
for s in ds:
    print("  样本:", s)

# ---------- 继承：迷你 nn.Module ----------
class Layer:
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        raise NotImplementedError("子类必须实现forward")

    def __call__(self, x):
        return self.forward(x)


class ReLU(Layer):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        return max(0.0, x)


class Scale(Layer):
    def __init__(self, k):
        super().__init__("scale")
        self.k = k

    def forward(self, x):
        return self.k * x


relu, scale = ReLU(), Scale(10)
print(relu(-3), relu(3), scale(2))         # 0.0 3 20
print(isinstance(relu, Layer))              # True

# 层可以像积木一样串起来
layers = [Scale(2), ReLU(), Scale(0.5)]
x = -4
for lyr in layers:
    x = lyr(x)
    print(f"  经过 {lyr.name}: {x}")

# ---------- @property ----------
class Config:
    def __init__(self, hidden, heads):
        self.hidden, self.heads = hidden, heads

    @property
    def head_dim(self):
        return self.hidden // self.heads

print(Config(768, 12).head_dim)             # 64

# ---------- @dataclass ----------
@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 10
    model_name: str = "mini-gpt"

cfg = TrainConfig(lr=1e-3)
print(cfg)
print(cfg.batch_size)

# ---------- 练习答案：Tokenizer / Loss ----------
class Tokenizer:
    def __init__(self, vocab):
        self.word2id = {w: i + 1 for i, w in enumerate(vocab)}

    def encode(self, text):
        return [self.word2id.get(w, 0) for w in text.split()]

    def __len__(self):
        return len(self.word2id) + 1

tok = Tokenizer(["我", "爱", "大模型"])
print(tok.encode("我 爱 你"), len(tok))     # [1, 2, 0] 4


class Loss:
    def __call__(self, pred, target):
        return self.compute(pred, target)

class MSE(Loss):
    def compute(self, pred, target):
        return (pred - target) ** 2

class MAE(Loss):
    def compute(self, pred, target):
        return abs(pred - target)

print(MSE()(3, 1), MAE()(3, 1))             # 4 2
