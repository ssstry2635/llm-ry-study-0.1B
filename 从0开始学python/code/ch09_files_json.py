# 第09章示例代码：文件读写与JSON
# 运行方式：在"从0开始学python"目录下执行  python code/ch09_files_json.py
# 本文件会在当前目录下的 io_demo/ 文件夹里创建演示文件（可随意删除）

import json
import os
from pathlib import Path

os.makedirs("io_demo", exist_ok=True)     # 建演示文件夹（已存在不报错）

# ---------- 写文本（"w"模式） ----------
with open("io_demo/story.txt", "w", encoding="utf-8") as f:
    f.write("从前有座山\n")
    f.write("山里有座庙\n")

# ---------- 逐行读 ----------
with open("io_demo/story.txt", encoding="utf-8") as f:
    for line in f:
        print("读到:", line.strip())

# ---------- 追加模式（"a"）写训练日志 ----------
with open("io_demo/train.log", "w", encoding="utf-8") as f:   # 先"w"重置
    for epoch, loss in enumerate([0.9, 0.5, 0.2], start=1):
        f.write(f"epoch={epoch}, loss={loss:.4f}\n")
with open("io_demo/train.log", "a", encoding="utf-8") as f:   # 再"a"追加一行
    f.write("epoch=4, loss=0.1000 (这是追加的一行)\n")
with open("io_demo/train.log", encoding="utf-8") as f:
    print("日志内容:")
    print(f.read(), end="")

# ---------- 写数字再读回计算 ----------
with open("io_demo/nums.txt", "w", encoding="utf-8") as f:
    for i in range(1, 11):
        f.write(f"{i}\n")
total = 0
with open("io_demo/nums.txt", encoding="utf-8") as f:
    for line in f:
        total += int(line.strip())
print("1~10的和 =", total)

# ---------- JSON ----------
config = {"name": "mini-gpt", "dim": 512, "lr": 3e-4, "scores": [0.9, 0.85]}
with open("io_demo/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)   # ensure_ascii=False让中文原样显示

with open("io_demo/config.json", encoding="utf-8") as f:
    cfg2 = json.load(f)
print("读回JSON:", cfg2["name"], cfg2["dim"])
print("JSON字符串预览:", json.dumps(config)[:50], "...")

# ---------- pathlib（读操作） ----------
p = Path("io_demo/config.json")
print("config.json存在吗:", p.exists(), "| 大小:", p.stat().st_size, "字节")
print("目录下的文件:", sorted(q.name for q in Path("io_demo").iterdir()))

print("\n提示: io_demo/ 是本次运行生成的演示目录，可随时删除")
