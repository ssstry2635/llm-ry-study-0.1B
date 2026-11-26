# 第05章示例代码：四大容器
# 运行方式： python code/ch05_containers.py

# ---------- 列表 ----------
tokens = [15496, 11, 616, 9822]
print(tokens[0], tokens[-1], tokens[1:3], len(tokens))
print(11 in tokens)

models = ["gpt", "bert"]
models.append("llama")
models.insert(1, "t5")
models[0] = "gpt-4"
models.remove("t5")
last = models.pop()
print(models, "弹出了", last)

losses = [0.9, 0.2, 0.5, 0.1]
print(sorted(losses), min(losses), max(losses), sum(losses))

batch = [[1, 5, 8], [2, 9], [3, 3, 3, 7]]
print(batch[0], batch[0][2])

# ---------- 元组 ----------
shape = (2, 3)
rows, cols = shape
print(f"rows={rows}, cols={cols}")
a, b = 1, 2
a, b = b, a
print("交换后:", a, b)

# ---------- 字典 ----------
vocab = {"hello": 15496, "world": 9822}
print(vocab["hello"], vocab.get("xxx", -1))

config = {"hidden_size": 768, "layers": 12, "vocab_size": 50257}
for key, value in config.items():
    print(f"{key} = {value}")

model_config = {
    "name": "mini-gpt",
    "params": {"layers": 6, "dim": 512},
}
print(model_config["params"]["layers"])

# ---------- 集合 ----------
tokens2 = [5, 3, 5, 8, 3, 9, 5]
print(set(tokens2))
a_set, b_set = {1, 2, 3, 4}, {3, 4, 5}
print(a_set & b_set, a_set | b_set, a_set - b_set)
print(set("我爱大模型") == set("大模型爱我"))

# ---------- 可变与拷贝 ----------
x = [1, 2, 3]
y = x                       # 贴标签，不拷贝
y.append(4)
print(x)                    # [1, 2, 3, 4]
z = x.copy()                # 真拷贝
z.append(99)
print(x, z)                 # [1, 2, 3, 4] / [1, 2, 3, 4, 99]
