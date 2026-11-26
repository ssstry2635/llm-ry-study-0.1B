"""M1 辅助：抽样查看原始语料，回答三个问题——
  1. 数据长什么样（随机打印几篇）
  2. 文档长度分布（决定"太短"阈值）
  3. 中文维基里繁体占多少（决定要不要繁转简）

用法：
    python scripts/inspect_raw.py            # 每种语料抽 20000 篇统计
    python scripts/inspect_raw.py -n 5000
"""
import argparse
import random
import sys
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.data_prep import han_ratio, iter_tinystories, iter_wiki_zh, to_simplified  # noqa: E402

RAW = ROOT / "data" / "raw"


def percentiles(values: list[int], ps=(5, 25, 50, 75, 95)) -> str:
    s = sorted(values)
    return "  ".join(f"p{p}={s[min(len(s) - 1, len(s) * p // 100)]:,}" for p in ps)


def show_samples(docs: list[str], k: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    for text in rng.sample(docs, min(k, len(docs))):
        preview = text[:160].replace("\n", "⏎")
        print(f"  [{len(text):>6} 字] {preview}")


def inspect_zh(n: int) -> None:
    docs = list(islice(iter_wiki_zh(RAW / "wiki_zh" / "20231101.zh"), n))
    lens = [len(d) for d in docs]
    print(f"\n===== 中文维基（前 {len(docs):,} 篇）=====")
    print("长度分位：", percentiles(lens))
    print(f"短于 100 字的占 {sum(l < 100 for l in lens) / len(lens):.1%}，"
          f"汉字比例 <0.5 的占 {sum(han_ratio(d) < 0.5 for d in docs) / len(docs):.1%}")
    # 繁体检测：转成简体后有变化的字符占比
    changed = sum(sum(a != b for a, b in zip(d, to_simplified(d))) for d in docs)
    total = sum(lens)
    trad_docs = sum(to_simplified(d) != d for d in docs)
    print(f"繁转简后有变化的文档占 {trad_docs / len(docs):.1%}，变化字符占全部字符的 {changed / total:.2%}")
    print("随机样本：")
    show_samples(docs, 5)
    print("最短的 3 篇（看看是什么垃圾）：")
    for d in sorted(docs, key=len)[:3]:
        print(f"  [{len(d):>6} 字] {d[:120]!r}")


def inspect_en(n: int) -> None:
    docs = list(islice(iter_tinystories(RAW / "tinystories" / "TinyStoriesV2-GPT4-train.txt"), n))
    lens = [len(d) for d in docs]
    print(f"\n===== TinyStories（前 {len(docs):,} 篇）=====")
    print("长度分位：", percentiles(lens))
    print(f"短于 200 字符的占 {sum(l < 200 for l in lens) / len(lens):.1%}")
    print("随机样本：")
    show_samples(docs, 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20000)
    args = ap.parse_args()
    inspect_zh(args.n)
    if (RAW / "tinystories" / "TinyStoriesV2-GPT4-train.txt").exists():
        inspect_en(args.n)
    else:
        print("\nTinyStories 尚未下载完成，跳过。")
