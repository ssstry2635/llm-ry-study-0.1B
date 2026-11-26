# 第07章示例代码：推导式与内置函数
# 运行方式： python code/ch07_comprehensions.py

nums = [1, 2, 3, 4, 5]

# ---------- 列表推导式 ----------
print([n * n for n in nums])                      # [1, 4, 9, 16, 25]
print([n for n in nums if n % 2 == 0])            # [2, 4]
print(["小" if n < 3 else "大" for n in nums])    # ['小','小','大','大','大']

# ---------- 双层推导 ----------
batch = ["Hello World", "Big Model"]
print([text.lower().split() for text in batch])

matrix = [[1, 2], [3, 4], [5, 6]]
print([x for row in matrix for x in row])         # [1,2,3,4,5,6]

# ---------- 字典/集合推导式 ----------
names, sizes = ["gpt", "llama"], [175, 65]
print({n: s for n, s in zip(names, sizes)})

vocab_list = ["我", "爱", "模型"]
word2id = {w: i for i, w in enumerate(vocab_list)}
print(word2id)                                    # {'我': 0, '爱': 1, '模型': 2}
print({len(w) for w in vocab_list})               # {1, 2}

# ---------- 内置函数 ----------
vals = [3, 1, 4, 1, 5, 9, 2, 6]
print(sum(vals), max(vals), min(vals), len(vals))
print(sum(vals) / len(vals))

losses = [0.5, 0.02, 0.4]
print(any(l < 0.1 for l in losses), all(l < 1.0 for l in losses))

words = ["banana", "apple", "kiwi"]
print(sorted(words, key=len))

runs = [("a", 0.3), ("b", 0.1), ("c", 0.2)]
print(sorted(runs, key=lambda r: r[1])[0])

# ---------- 生成器表达式 ----------
print(sum(x * x for x in range(1_000_000)))

# ---------- 真实风格：文本批处理 ----------
vocab = {"hello": 15496, "world": 9822}
batch = ["hello world", "hello unk_word"]
batch_ids = [[vocab.get(t, 0) for t in text.split()] for text in batch]
print(batch_ids)      # [[15496, 9822], [15496, 0]]
