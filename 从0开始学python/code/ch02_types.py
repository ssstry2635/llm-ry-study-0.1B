# 第02章示例代码：变量与数据类型
# 运行方式： python code/ch02_types.py

# ---------- 变量 ----------
model = "GPT"
layers = 12
learning_rate = 0.0003
is_training = True
best_loss = None

print(model, layers, learning_rate, is_training, best_loss)

# ---------- type() ----------
print(type(42))          # <class 'int'>
print(type(0.0003))      # <class 'float'>
print(type(model))       # <class 'str'>
print(type(is_training)) # <class 'bool'>
print(type(best_loss))   # <class 'NoneType'>

# ---------- 类型转换 ----------
print(int("123"))        # 123
print(int(3.99))         # 3 （向零取整）
print(int(-3.99))        # -3
print(float("0.5"))      # 0.5
print(str(2026))         # '2026'
print(bool(0), bool(""), bool(None))   # False False False
print(bool(42), bool("a"))             # True True

# ---------- 浮点数陷阱 ----------
print(0.1 + 0.2)                    # 0.30000000000000004
print(0.1 + 0.2 == 0.3)             # False！
print(abs(0.1 + 0.2 - 0.3) < 1e-9)  # True —— 正确的小数比较方式
