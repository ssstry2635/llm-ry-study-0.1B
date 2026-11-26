# 第12章示例代码：迭代器、生成器、装饰器、上下文管理器
# 运行方式： python code/ch12_generators_decorators.py

import time
from functools import wraps
from contextlib import contextmanager

# ---------- 迭代器原理 ----------
it = iter([1, 2, 3])
print(next(it), next(it), next(it))      # 1 2 3

# ---------- 生成器 ----------
def gen_squares(n):
    for i in range(n):
        yield i * i

g = gen_squares(5)
print(g)                                  # <generator object ...>
print(next(g), next(g))                   # 0 1
print(list(g))                            # [4, 9, 16]  剩下的

print(sum(x * x for x in range(1_000_000)))   # 生成器表达式

# ---------- 分批喂数据 ----------
def batch_iter(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

for batch in batch_iter(list(range(10)), 4):
    print("batch:", batch)

# ---------- 装饰器 ----------
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.time() - t0:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    return sum(range(n))

print(slow_sum(5_000_000))
print(slow_sum.__name__)                  # slow_sum（有@wraps才保得住名字）

def repeat(n):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def hello():
    print("hi")

hello()

def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用了 {func.__name__}，参数 {args}")
        return func(*args, **kwargs)
    return wrapper

@log_call
def add(a, b):
    return a + b

print(add(1, 2))

# ---------- 上下文管理器 ----------
class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        print(f"[Timer类] 耗时 {time.time() - self.t0:.4f}s")

with Timer():
    sum(range(3_000_000))

@contextmanager
def timer_ctx(name):
    t0 = time.time()
    yield
    print(f"[{name}] 耗时 {time.time() - t0:.4f}s")

with timer_ctx("训练一轮"):
    sum(range(3_000_000))

@contextmanager
def suppress_errors():
    try:
        yield
    except Exception as e:
        print("捕获到异常:", e)

with suppress_errors():
    1 / 0
print("程序还活着")

# ---------- 练习答案 ----------
def countdown(n):
    while n > 0:
        yield n
        n -= 1
print(list(countdown(5)))

def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
out = []
for v in fib():
    if len(out) == 10:
        break
    out.append(v)
print(out)
