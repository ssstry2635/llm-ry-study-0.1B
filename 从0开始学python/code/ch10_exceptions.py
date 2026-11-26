# 第10章示例代码：异常处理
# 运行方式： python code/ch10_exceptions.py

# ---------- try / except 基本 ----------
try:
    x = int("abc")
    print("这行不会执行")
except ValueError:
    print("转换失败，用默认值0")
    x = 0
print("程序没有崩，x =", x)

# ---------- as e 拿信息 ----------
try:
    open("这个文件不存在.txt", encoding="utf-8")
except FileNotFoundError as e:
    print("出错了:", e)

# ---------- 多种异常 ----------
def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return float("inf")
    except TypeError:
        return None

print(safe_div(1, 0), safe_div(1, "x"), safe_div(6, 3))   # inf None 2.0

# ---------- else / finally ----------
try:
    v = int("42")
except ValueError:
    print("转换失败")
else:
    print("转换成功:", v)          # 没出错才执行
finally:
    print("finally 无论如何都执行")

# ---------- raise ----------
def set_lr(lr):
    if lr <= 0:
        raise ValueError(f"学习率必须为正数，收到 {lr}")
    return lr

print("lr =", set_lr(0.001))
try:
    set_lr(-1)
except ValueError as e:
    print("捕获到:", e)

# ---------- assert ----------
batch = [[1, 2, 3], [4, 5, 6]]
assert len(batch) == 2, "batch大小应该是2"
assert all(len(row) == 3 for row in batch), "每行长度应该是3"
print("形状检查通过: 2行3列")

# ---------- 训练脚本的容错骨架（模拟） ----------
def fake_train():
    for step in range(3):
        print(f"  训练 step {step}")
        if step == 2:
            raise RuntimeError("CUDA out of memory (模拟显存爆了)")

try:
    fake_train()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("显存不足，请减小 batch_size")
    else:
        raise
finally:
    print("已保存 checkpoint（无论成败）")

# ---------- 练习答案 ----------
def safe_int(s):
    try:
        return int(s)
    except ValueError:
        return None

print(safe_int("12"), safe_int("ab"))

def check_config(cfg):
    if "lr" not in cfg:
        raise KeyError("配置缺少 'lr'")
    if cfg["lr"] <= 0:
        raise ValueError(f"lr 必须为正，收到 {cfg['lr']}")
    return True

for cfg in [{"lr": 0.1}, {"lr": -1}, {}]:
    try:
        check_config(cfg)
        print(cfg, "-> 合法")
    except (KeyError, ValueError) as e:
        print(cfg, "-> 不合法:", e)
