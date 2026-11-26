# 第06章示例代码：函数
# 运行方式： python code/ch06_functions.py

# ---------- 定义与return ----------
def add(a, b):
    return a + b

print(add(3, 5) * 2)              # 16

def min_max(nums):
    return min(nums), max(nums)

low, high = min_max([3, 1, 4, 1, 5])
print(low, high)                  # 1 5

# ---------- 参数四种玩法 ----------
def power(base, exp=2):           # exp有默认值
    return base ** exp

print(power(2, 10), power(3))     # 1024 9
print(power(exp=10, base=2))      # 1024 关键字参数

def summary(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

summary(1, 2, 3, lr=0.01, epochs=5)

# ---------- 可变默认值坑 ----------
def bad(item, box=[]):
    box.append(item)
    return box

print(bad(1), bad(2))             # [1, 2] [1, 2] —— 第二次带着第一次的！坏！

def good(item, box=None):
    if box is None:
        box = []
    box.append(item)
    return box

print(good(1), good(2))           # [1] [2] —— 正确

# ---------- 函数是对象 / lambda ----------
def relu(x):
    return max(0, x)

f = relu
print(f(-3), f(5))                # 0 5

square = lambda x: x * x
print(square(9))                  # 81

models = [("gpt", 175), ("llama", 65), ("qwen", 7)]
print(sorted(models, key=lambda m: m[1]))

# ---------- 递归 ----------
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))               # 120

# ---------- 神经网络就是函数嵌套 ----------
def linear_then_relu(x, w, b):
    return max(0, w * x + b)

print(linear_then_relu(1.0, 2.0, -3.0))   # 0 （2*1-3<0 被relu压平）
print(linear_then_relu(2.0, 2.0, -3.0))   # 1.0
