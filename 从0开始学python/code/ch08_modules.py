# 第08章示例代码：模块、标准库
# 运行方式： python code/ch08_modules.py
# 注意：本文件 import 了同目录的 mymath.py

import math
import random
import os
import sys
import time
from collections import Counter

import mymath   # ← 自己写的模块！（同目录）

# ---------- math ----------
print(math.sqrt(9), round(math.log(math.e), 4), round(math.exp(1), 4))

# ---------- random ----------
# 注意：random 只用于模拟/抽样/打乱数据；涉及密码、密钥请用 secrets 模块
random.seed(42)                      # 固定种子，结果可复现
print(random.random())
print(random.randint(1, 6))
print(random.choice(["gpt", "llama"]))

lst = [1, 2, 3]
random.shuffle(lst)                  # 原地打乱，返回None
print("打乱后:", lst)

# ---------- os / sys ----------
print("当前目录:", os.getcwd())
print("README存在吗:", os.path.exists("../README.md"))
print("拼接路径:", os.path.join("data", "train.txt"))
print("Python版本:", sys.version.split()[0])

# ---------- time 计时 ----------
t0 = time.time()
s = sum(range(1_000_000))
print(f"sum耗时 {(time.time() - t0)*1000:.1f} ms, 结果={s}")

# ---------- Counter ----------
c = Counter(["我", "爱", "大", "模型", "大"])
print(c, c.most_common(1))

# ---------- 导入自己的模块 ----------
print("sigmoid(0) =", mymath.sigmoid(0))
print("softmax([2,1,0.1]) =", [round(p, 4) for p in mymath.softmax([2.0, 1.0, 0.1])])
print("注意：mymath里的自测代码没有执行——这就是 __main__ 开关的作用")
