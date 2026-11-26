"""M4：把清洗后的全部语料 token 化，写成 uint16 的 .bin（memmap 用）。

多进程并行：主进程按行块分发，每个 worker 独立加载 tokenizer 编码，
返回 uint16 字节流，主进程顺序追加写文件。
每篇文档之间插入 <|endoftext|>（让模型学会"文档有边界"）。

用法：
    python scripts/03_tokenize_corpus.py               # 全量 → data/tokenized/{train,val}.bin
    python scripts/03_tokenize_corpus.py --workers 1   # 单进程调试
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.bpe import BPETokenizer  # noqa: E402

CLEAN = ROOT / "data" / "clean"
OUT_DIR = (ROOT / "data" / "tokenized").resolve()
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
ALLOWED_SPLITS = ("train", "val")

_TOK: BPETokenizer | None = None


def _worker_init():
    global _TOK
    _TOK = BPETokenizer.load(TOKENIZER_PATH)


def _encode_chunk(lines: list[str]) -> bytes:
    eot = _TOK.special_id("<|endoftext|>")
    ids: list[int] = []
    for line in lines:
        ids.extend(_TOK.encode(json.loads(line)["text"]))
        ids.append(eot)
    return np.asarray(ids, dtype=np.uint16).tobytes()


def tokenize_split(name: str, workers: int, chunk_lines: int = 1000) -> None:
    assert name in ALLOWED_SPLITS, f"非法的 split 名：{name!r}"
    src = CLEAN / f"{name}.jsonl"
    dst = (OUT_DIR / f"{name}.bin").resolve()
    assert dst.is_relative_to(OUT_DIR), f"输出路径越界：{dst}"
    if not src.exists():
        print(f"[{name}] 缺少 {src}，跳过")
        return

    def line_chunks():
        buf: list[str] = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                buf.append(line)
                if len(buf) >= chunk_lines:
                    yield buf
                    buf = []
        if buf:
            yield buf

    n_chars = total = 0
    t0 = time.time()
    with open(dst, "wb") as out:
        if workers == 1:
            _worker_init()
            for chunk in tqdm(line_chunks(), desc=f"[{name}]"):
                blob = _encode_chunk(chunk)
                out.write(blob)
                total += len(blob) // 2
                n_chars += sum(len(json.loads(l)["text"]) for l in chunk)
        else:
            with mp.Pool(workers, initializer=_worker_init) as pool:
                for blob in tqdm(pool.imap(_encode_chunk, line_chunks(), chunksize=1),
                                 desc=f"[{name}]"):
                    out.write(blob)
                    total += len(blob) // 2
    dt = time.time() - t0
    print(f"[{name}] {total:,} tokens，{n_chars:,} 字符 → {dst.name} "
          f"({dst.stat().st_size / 2**30:.2f} GB)，耗时 {dt / 60:.1f} 分钟")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert TOKENIZER_PATH.exists(), "先运行 scripts/02_train_tokenizer.py"
    for split in ALLOWED_SPLITS:
        tokenize_split(split, args.workers)
    print("全部完成 ✔  下一步：python scripts/04_train.py --config test")
