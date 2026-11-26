# 第 09 章：文件读写与 JSON

> 本章目标：让程序"记住"东西——读写文本文件、CSV、JSON。**训练数据和模型配置都从文件来。**

## 1. open 与 with：文件操作的标配

### 最古老的写法（了解即可）

```python
f = open("data.txt", "r", encoding="utf-8")   # 打开
content = f.read()                             # 读取
f.close()                                      # 必须手动关！忘了就出事
```

### with 写法（永远用这个！）

```python
with open("data.txt", "r", encoding="utf-8") as f:   # 离开with块自动关闭
    content = f.read()      # 一次性读完整个文件成一个字符串
```

`with open(...) as f:` 的意思是"打开文件交给 f 管，用完自动关"。以后见到任何文件操作都是这个形态。

**`encoding="utf-8"` 必写**：中文 Windows 默认编码是 GBK，不写的话读写中文必乱码/报错。

### 三种模式

| 模式 | 含义 | 文件不存在 | 文件已存在 |
|---|---|---|---|
| `"r"` 读 | 只读 | 报错 | 正常 |
| `"w"` 写 | 写入 | 自动创建 | **清空重写！** |
| `"a"` 追加 | 续写 | 自动创建 | 接在末尾 |

⚠️ `"w"` 会无提示清空原文件，选模式时手别抖。

## 2. 读文件的三种姿势

```python
# 姿势1：整个读成一个字符串
with open("data.txt", encoding="utf-8") as f:
    text = f.read()

# 姿势2：按行读成列表（保留每行末尾的\n）
with open("data.txt", encoding="utf-8") as f:
    lines = f.readlines()

# 姿势3：逐行迭代（推荐！大文件也不爆内存）
with open("data.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()       # 去掉行尾换行符，几乎每处都要做
        print(line)
```

## 3. 写文件

```python
# 写文本
with open("out.txt", "w", encoding="utf-8") as f:
    f.write("第一行\n")            # write不会自动换行，要自己加\n
    f.write("第二行\n")

# 追加训练日志
with open("train.log", "a", encoding="utf-8") as f:
    f.write(f"epoch=3 loss=0.4213\n")
```

一次写多行用 `f.writelines([...])`（同样不自动加换行），或者用 print 的文件参数：`print("hello", file=f)`。

## 4. CSV：用标准库 csv 模块

```python
import csv

# 写
rows = [["模型", "loss"], ["gpt", 0.42], ["llama", 0.38]]
with open("results.csv", "w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerows(rows)

# 读
with open("results.csv", encoding="utf-8") as f:
    for row in csv.reader(f):
        print(row)                 # 每行是一个列表
```

`newline=""` 是 Windows 下的防坑参数（防止空行），写 CSV 时带上。

## 5. JSON：配置文件的通用格式

JSON 长这样（和 Python 字典/列表几乎一样）：

```json
{"model": "gpt2", "layers": 12, "scores": [0.9, 0.85]}
```

Python 对象 ⇄ JSON 字符串/文件：

```python
import json

config = {"model": "mini-gpt", "layers": 6, "lr": 3e-4}

# 对象 → JSON字符串
s = json.dumps(config, indent=2)        # indent=2 美化缩进
print(s)

# JSON字符串 → 对象
config2 = json.loads(s)

# 对象 → 存文件 / 读文件
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

with open("config.json", encoding="utf-8") as f:
    config3 = json.load(f)
```

记忆：**dumps/loads 处理字符串（s=string），dump/load 处理文件**。

类型对照：dict→object、list→array、str→string、int/float→number、True→true、None→null。

## 6. pathlib：现代路径操作

```python
from pathlib import Path

p = Path("data")
print(p / "train.txt")            # data\train.txt   用/拼路径，优雅
print(p.exists())                 # 是否存在
Path("saved").mkdir(exist_ok=True)  # 建文件夹（已存在不报错）
for f in Path(".").glob("*.py"):  # 找当前目录所有py文件
    print(f.name)
```

新代码用 `pathlib`，看到 `os.path.join` 要能读懂。

## 7. 大模型视角 🤖

- **训练语料**：一行一个样本的 `txt/jsonl`，逐行 `for line in f` 流式读（几 GB 也稳）
- **config.json**：HuggingFace 每个模型都有——就是本章的 `json.load`
- **checkpoint**：训练中断要保存进度；`Path("ckpt").exists()` 决定"接着练"还是"从头练"
- **train.log**：`"a"` 追加模式记录每步 loss，事后画曲线

```python
# 真实风格的断点续训逻辑
import json
from pathlib import Path

ckpt = Path("checkpoint.json")
if ckpt.exists():
    state = json.loads(ckpt.read_text(encoding="utf-8"))
    start_epoch = state["epoch"] + 1
    print(f"从第{start_epoch}轮继续训练")
```

## 本章小结

- `with open(路径, 模式, encoding="utf-8") as f:` 一把梭
- `"w"` 清空重写、`"a"` 追加；逐行迭代读大文件
- json：dumps/loads 管字符串，dump/load 管文件
- pathlib 用 `/` 拼路径、glob 找文件

## 练习

1. 把 `"从前有座山"` 写入 `story.txt`，再读出来打印。
2. 生成 `nums.txt`：内容是 1~10 每行一个数字；读回后求它们的和（提示：`int(line)`）。
3. 把字典 `{"name": "mini-gpt", "dim": 512}` 存成 `model.json`，再读回打印 `dim`。
4. 用 `Path` 检查 `model.json` 是否存在，存在则打印其大小（提示：`p.stat().st_size`）。
5. 模拟训练日志：用追加模式写入 3 行 `epoch=?, loss=?`（loss 递减）。

---

### 参考答案

```python
# 1
with open("story.txt", "w", encoding="utf-8") as f:
    f.write("从前有座山")
with open("story.txt", encoding="utf-8") as f:
    print(f.read())

# 2
with open("nums.txt", "w", encoding="utf-8") as f:
    for i in range(1, 11):
        f.write(f"{i}\n")
total = 0
with open("nums.txt", encoding="utf-8") as f:
    for line in f:
        total += int(line.strip())
print(total)   # 55

# 3
import json
with open("model.json", "w", encoding="utf-8") as f:
    json.dump({"name": "mini-gpt", "dim": 512}, f, indent=2)
with open("model.json", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg["dim"])   # 512

# 4
from pathlib import Path
p = Path("model.json")
if p.exists():
    print(p.stat().st_size, "字节")

# 5
losses = [0.9, 0.5, 0.2]
for epoch, loss in enumerate(losses, start=1):   # start=1 让编号从1开始
    with open("train.log", "a", encoding="utf-8") as f:
        f.write(f"epoch={epoch}, loss={loss}\n")
```
