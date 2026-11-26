# 第03章示例代码：运算符与字符串
# 运行方式： python code/ch03_ops_strings.py

# ---------- 算术 ----------
print(7 + 3, 7 - 3, 7 * 3)       # 10 4 21
print(7 / 3)                     # 2.3333...
print(7 // 3, 7 % 3)             # 2 1
print(2 ** 10)                   # 1024

# batch 计算的例子
n_tokens, batch_size = 10, 4
print(f"{n_tokens}个token, batch={batch_size}: 能切{n_tokens//batch_size}批, 剩{n_tokens%batch_size}个")

# ---------- 比较 / 逻辑 ----------
loss, epoch = 0.05, 10
print(loss < 0.1 and epoch > 5)  # True
print(loss > 0.1 or epoch > 5)   # True
print(not (loss < 0.1))          # False

# ---------- 复合赋值 ----------
total = 0
total += 5
total *= 2
print(total)                     # 10

# ---------- 字符串 ----------
s = "transformer"
print(s[0], s[-1], len(s))       # t r 11
print(s[0:6], s[6:], s[-6:])     # trans former former
print(s[::2])                    # taso_e 之外的自己去数

text = "  Hello World  "
print(text.strip())
print(text.lower())
print(text.replace("World", "LLM"))
print("a,b,c".split(","))
print("-".join(["a", "b", "c"]))
print("llm" in "hello llm")      # True

# ---------- f-string ----------
epoch, loss = 7, 0.123456
print(f"第{epoch}轮, loss={loss}")
print(f"loss={loss:.2f}")
print(f"进度={7/10:.1%}")
print(f"参数量={175000000000:,}")

# ---------- 不可变 ----------
s = "hello"
s = "H" + s[1:]
print(s)                         # Hello
