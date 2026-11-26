# 第13章示例代码：类型注解
# 运行方式： python code/ch13_type_hints.py

from typing import Any, Callable, Optional, TypeAlias

# ---------- 基本注解 ----------
epochs: int = 10
lr: float = 3e-4
name: str = "mini-gpt"
best_loss: Optional[float] = None          # 也可以写 float | None


def train(model: str, epochs: int = 10, lr: float = 1e-3) -> bool:
    print(f"训练{model} {epochs}轮, lr={lr}")
    return True


print(train("gpt", epochs=3))

# ---------- 容器注解 ----------
tokens: list[int] = [1, 2, 3]
vocab: dict[str, int] = {"我": 0}
shape: tuple[int, int] = (2, 3)
results: list[dict[str, float]] = [{"loss": 0.5}, {"loss": 0.3}]
print(tokens, vocab, shape, results)

# ---------- 类型别名 ----------
Batch: TypeAlias = list[list[int]]


def collate(batch: Batch) -> list[int]:
    """把嵌套的batch拍平"""
    return [tok for seq in batch for tok in seq]


print(collate([[1, 2], [3, 4]]))           # [1, 2, 3, 4]

Sample: TypeAlias = dict[str, Any]


def get_input_ids(s: Sample) -> list[int]:
    return s["input_ids"]


print(get_input_ids({"input_ids": [5, 6, 7], "label": 1}))

# ---------- 函数类型 ----------
def apply_twice(f: Callable[[int], int], x: int) -> int:
    return f(f(x))


print(apply_twice(lambda v: v * 3, 2))     # 18

# ---------- 真实风格签名（能读懂的程度） ----------
def train_one_epoch(model: str, loader: list[list[int]], lr: float = 1e-3) -> float:
    """模拟一个带完整注解的训练函数"""
    total: float = 0.0
    for batch in loader:
        total += sum(batch) * lr
    return total / len(loader)


print(f"fake loss = {train_one_epoch('gpt', [[1, 2], [3, 4]]):.4f}")

# ---------- 练习答案 ----------
def top_k(scores: list[float], k: int = 5) -> list[int]:
    """返回分数最大的k个下标，按分数降序"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:k]

print(top_k([0.1, 0.9, 0.5, 0.7], 2))      # [1, 3]
