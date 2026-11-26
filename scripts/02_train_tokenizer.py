"""M2 第二步：在清洗后的语料上训练 BPE 词表（32k）。

只用语料的一个采样（默认 3 亿字符）来训词表——BPE 学的是"哪些字节对常出现"，
几亿字符的统计量已经非常稳，用全部语料只是浪费时间。

用法：
    python scripts/02_train_tokenizer.py                  # 默认 32768 词表
    python scripts/02_train_tokenizer.py --vocab-size 32768 --sample-chars 300000000
输出：data/tokenizer/tokenizer.json
"""
import argparse
import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.bpe import BPETokenizer  # noqa: E402

CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "tokenizer" / "tokenizer.json"


def read_sample(path: Path, max_chars: int) -> str:
    """从 train.jsonl 顺序读文档拼成采样文本（读够 max_chars 就停）。"""
    chunks: list[str] = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line)["text"])
            total += len(chunks[-1])
            if total >= max_chars:
                break
    print(f"[sample] 从 {path.name} 读取 {total:,} 字符（{len(chunks):,} 篇）", flush=True)
    return "".join(chunks)


def eval_compression(tok: BPETokenizer, val_path: Path, n_docs: int = 200) -> None:
    """在没参与训词表的验证集上量压缩率：token/字符 越低越好。"""
    zh_chars = zh_toks = en_chars = en_toks = 0
    with open(val_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            d = json.loads(line)
            n = len(tok.encode(d["text"]))
            if d["lang"] == "zh":
                zh_chars += len(d["text"]); zh_toks += n
            else:
                en_chars += len(d["text"]); en_toks += n
    if zh_chars:
        print(f"[eval] 中文：{zh_toks / zh_chars:.3f} token/字符 "
              f"（即平均 {zh_chars / zh_toks:.2f} 字/token；目标 ≤1.5 token/字）")
    if en_chars:
        print(f"[eval] 英文：{en_toks / en_chars:.3f} token/字符 "
              f"（约 {en_chars / en_toks:.2f} 字符/token；GPT-2 水平约 4）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--sample-chars", type=int, default=150_000_000)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    text = read_sample(CLEAN / "train.jsonl", args.sample_chars)
    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size, min_count=args.min_count)
    print(f"[bpe] 训练耗时 {(time.time() - t0) / 60:.1f} 分钟")
    tok.save(OUT)  # 先保存再评估：评估再慢也不能把训好的词表丢在内存里
    print(f"[bpe] 词表已保存：{OUT}", flush=True)
    eval_compression(tok, CLEAN / "val.jsonl")
