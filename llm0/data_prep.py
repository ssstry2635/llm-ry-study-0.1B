"""M1：语料清洗管线。

分工：
  - 标有 TODO(你来实现) 的 4 个函数由你完成，规格写在 docstring 里，
    验收标准是 tests/test_data_prep.py 全绿：
        python -m pytest tests/test_data_prep.py -v
  - 其余（读 parquet / 读 txt / 繁转简）是管道工程，已写好。

为什么要清洗？预训练模型是"数据的镜子"：
  * 太短的文档（维基消歧页、重定向残留）几乎没有语言信号，只会稀释训练。
  * 表格/列表/乱码会让模型学到无意义的符号分布。
  * 完全重复的文档会让模型"背书"，损害泛化，还浪费算力。
  * 中文维基繁简混杂——同一个词两套写法，对 0.1B 的小模型是双倍负担，统一成简体。
"""
from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re  # 支持 \p{Han} 等 Unicode 属性类，标准库 re 不支持

# =====================================================================
# 第一部分：你来实现（4 个函数）
# =====================================================================


def normalize_whitespace(text: str) -> str:
    """把文本的空白规整成统一形态。

    规则（按顺序做，顺序错了测试会挂）：
      1. Unicode NFC 归一化（用 unicodedata.normalize；例如分解形式的 "é" 合成一个码位）
      2. \\r\\n 和 \\r 统一成 \\n
      3. 制表符 \\t、全角空格 \\u3000、不间断空格 \\xa0 都当作普通空格
      4. 每一行首尾空白去掉，行内连续多个空格压成 1 个
      5. 连续 3 个及以上换行压成 2 个（保留段落分隔，去掉大段空行）
      6. 整体首尾空白去掉
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ").replace("\u3000", " ").replace("\xa0", " ")
    lines = [re.sub(r" {2,}", " ", line.strip()) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_good_zh_doc(text: str) -> bool:
    """中文文档质量过滤：True 保留，False 丢弃。

    规则：
      1. 去掉空白后长度 < 100 字符 → 丢（维基里大量一两句话的条目）
      2. 汉字占"非空白字符"的比例 < 0.5 → 丢（表格、纯数字、纯外文、乱码）
         汉字用 regex 的 \\p{Han} 匹配（包括生僻字与扩展区，比手写 \\u4e00-\\u9fff 全）

    注意：标点、数字、拉丁字母都算在分母里但不算汉字，所以一篇正常中文文章
    的汉字比例大约在 0.7~0.9，阈值 0.5 是留了余地的。
    """
    body = re.sub(r"\s+", "", text)  # 删掉所有空白：空格、换行、制表符……
    if len(body) < 100:              # 规则 1：太短 → 没有学习价值
        return False
    han = re.findall(r"\p{Han}", body)   # 规则 2：数一数汉字有几个
    return len(han) / len(body) >= 0.5   # 占比 ≥ 0.5 才保留


def is_good_en_doc(text: str) -> bool:
    """英文文档质量过滤：True 保留，False 丢弃。

    规则：
      1. 去掉空白后长度 < 200 字符 → 丢
      2. ASCII 字母 [A-Za-z] 占"非空白字符"的比例 < 0.6 → 丢
    """
    body = re.sub(r"\s+", "", text)          # 和中文版完全同一个套路
    if len(body) < 200:
        return False
    letters = re.findall(r"[A-Za-z]", body)  # 只是"数什么"换成英文字母
    return len(letters) / len(body) >= 0.6


def dedupe(docs: Iterable[str]) -> Iterator[str]:
    """精确去重：只产出每篇文档的第一次出现，保持原顺序。

    生成器实现：用 md5 的 16 字节摘要做集合成员。300 万篇文档的全文会撑爆内存，
    而 16 字节摘要的 set 只需要 ~100MB 量级。
    """
    seen: set[bytes] = set()
    for text in docs:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield text


# =====================================================================
# 第二部分：管道工程（已写好，读懂即可）
# =====================================================================

WIKI_ZH_DIR = Path("data/raw/wiki_zh/20231101.zh")
TINYSTORIES_FILE = Path("data/raw/tinystories/TinyStoriesV2-GPT4-train.txt")


def iter_wiki_zh(parquet_dir: Path = WIKI_ZH_DIR) -> Iterator[str]:
    """流式读取中文维基 parquet 的 text 列，一次一篇，内存占用与文件大小无关。"""
    import pyarrow.parquet as pq

    for file in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text:
                    yield text


def iter_tinystories(path: Path = TINYSTORIES_FILE, max_bytes: int | None = None) -> Iterator[str]:
    """流式读取 TinyStories：故事之间用单独一行 <|endoftext|> 分隔。

    max_bytes：最多读取这么多字节（用来控制英文语料占比），末尾不完整的故事丢弃。
    """
    buf: list[str] = []
    read = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            read += len(line)
            if line.strip() == "<|endoftext|>":
                if buf:
                    yield "".join(buf)
                buf = []
                if max_bytes is not None and read >= max_bytes:
                    return
            else:
                buf.append(line)
    # 文件末尾若没有分隔符收尾，最后一段仍是完整故事
    if buf and max_bytes is None:
        yield "".join(buf)


_t2s = None


def to_simplified(text: str) -> str:
    """繁体 → 简体（OpenCC t2s）。首次调用时加载转换表。"""
    global _t2s
    if _t2s is None:
        import opencc

        _t2s = opencc.OpenCC("t2s")
    return _t2s.convert(text)


_HAN = re.compile(r"\p{Han}")


def han_ratio(text: str) -> float:
    """汉字占非空白字符的比例（0~1）。抽样检查数据时用。"""
    body = re.sub(r"\s+", "", text)
    if not body:
        return 0.0
    return len(_HAN.findall(body)) / len(body)


def stable_bucket(text: str, buckets: int = 100) -> int:
    """把文档稳定地映射到 0..buckets-1，用于可复现的 train/val 划分。

    为什么不用"最后 1% 当 val"？维基是按标题排序存的，最后 1% 全是某些字母开头
    的条目，val 分布会和 train 不一致。按内容哈希分桶就没有这个问题。
    """
    h = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % buckets
