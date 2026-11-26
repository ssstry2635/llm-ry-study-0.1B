# 第04章示例代码：条件与循环
# 运行方式： python code/ch04_loops.py

# ---------- if / elif / else ----------
loss = 0.08
if loss < 0.1:
    print("收敛得不错")
elif loss < 0.5:
    print("继续训练")
else:
    print("哪里出问题了")

# ---------- for + range ----------
for i in range(3):
    print(f"第{i}次")

for i in range(5, 0, -1):        # 倒数
    print(i, end=" ")
print()

# ---------- while ----------
loss, step = 1.0, 0
while loss > 0.1:
    loss *= 0.6
    step += 1
print(f"{step}步后loss={loss:.3f}")

# ---------- break / continue ----------
for i in range(100):
    if i * i >= 50:
        print(f"第一个平方>=50的数是{i}")
        break

for i in range(5):
    if i == 2:
        continue
    print(i, end=" ")            # 0 1 3 4
print()

# ---------- enumerate / zip ----------
models = ["gpt", "llama", "qwen"]
for idx, name in enumerate(models):
    print(f"第{idx}名: {name}")

names = ["loss", "acc"]
values = [0.12, 0.95]
for n, v in zip(names, values):
    print(f"{n} = {v}")

# ---------- 嵌套循环：矩阵 ----------
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
for row in matrix:
    for x in row:
        print(x, end=" ")
    print()

# ---------- 训练主循环骨架（看眼熟） ----------
print("--- 模拟训练循环 ---")
for step in range(6):
    fake_loss = 1.0 * (0.6 ** step)
    if step % 2 == 0:
        print(f"step {step} loss {fake_loss:.4f}")
    if fake_loss < 0.2:
        print("提前停止!")
        break
