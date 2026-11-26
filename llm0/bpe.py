"""M2：byte-level BPE tokenizer（GPT-2 同款方案），从零实现。

== 原理（三步）==

1. 预分词（pre-tokenization）：用正则把文本切成"词块"（英文单词、数字串、
   连续汉字、标点……）。先切这一刀，是为了让 BPE 的合并永远不会跨越词块边界
   （否则可能学出"词尾+下一词开头"这种无意义拼接）。

2. 字节映射：每个词块转成 UTF-8 字节，再把每个字节（0~255）映射成一个
   "可见 Unicode 字符"。为什么要映射？因为 BPE 的词表要用字符串表示，
   而部分字节（控制字符、换行）既不好显示也不好在 JSON 里存。
   映射表是 GPT-2 的原版（比如空格 0x20 → 'Ġ'）。

3. BPE 合并：统计所有相邻符号对的出现次数，把最高频的一对合并成新符号、
   加入词表，重复 ~32000 次。高频中文词（如"的人"）的字节会被逐步合并成
   单个 token，于是常见文本能用很少的 token 表示。

== 使用 ==

    tok = BPETokenizer()
    tok.train(sample_text, vocab_size=32768)   # 见 scripts/01_train_tokenizer.py
    ids = tok.encode("你好，世界")
    assert tok.decode(ids) == "你好，世界"
    tok.save("data/tokenizer/tokenizer.json")
"""
from __future__ import annotations

import heapq
import json
import os
from collections import defaultdict
from pathlib import Path

import regex as re

# GPT-2 的预分词正则：
#   缩写 ('s 't ...) | 前导空格+字母串 | 前导空格+数字串 | 前导空格+符号串 | 行尾空白 | 空白
# regex 库的 \p{L} 是"任意语言的字母"（含汉字），所以中文连续段会被切成一个词块。
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# 词块最大长度（字符）。中文没有空格，一整句话会成一个上百字节的词块，
# 逐词 BPE 的合并扫描是 O(长度²)——超长词块是编码速度杀手。
# 超过就等长切断。训词表与编码用同一条规则，语义自洽；代价只是切断点上的
# 少数合并机会丢失（约 1/长度 的比例），压缩率几乎不受影响。
MAX_WORD_CHARS = 20


def iter_pretokens(text: str):
    """预分词：PAT 切分 + 超长词块等长截断。训练与编码共用，保证一致。"""
    for m in PAT.finditer(text):
        w = m.group()
        if len(w) <= MAX_WORD_CHARS:
            yield w
        else:
            for i in range(0, len(w), MAX_WORD_CHARS):
                yield w[i:i + MAX_WORD_CHARS]


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 的字节→可见字符映射。可打印字节保持原样，其余映射到 256 之后的位置。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}

ENDOFTEXT = "<|endoftext|>"


def merge_symbols(syms: list[str], pair: tuple[str, str], new_sym: str) -> list[str]:
    """把 syms 里所有（不重叠、从左到右）的 pair 替换成 new_sym。"""
    out: list[str] = []
    i = 0
    while i < len(syms):
        if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
            out.append(new_sym)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


class BPETokenizer:
    """byte-level BPE。token = 一个字符串（初始为单字节映射字符，合并后越来越长）。"""

    def __init__(self, merges: list[tuple[str, str]] | None = None,
                 special_tokens: tuple[str, ...] = (ENDOFTEXT,)):
        self.special_tokens = list(special_tokens)
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {c: i for i, c in enumerate(B2U.values())}  # 0..255
        for pair in merges or []:
            self._add_merge(pair)
        self._cache: dict[str, list[int]] = {}
        self._ranks: dict[tuple[str, str], int] | None = None  # 惰性构建，训练期间不占
        self._special_pat = re.compile(
            "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")

    # ------------------------------------------------ 词表维护

    def _add_merge(self, pair: tuple[str, str]) -> None:
        new_sym = pair[0] + pair[1]
        if new_sym in self.vocab:
            raise ValueError(f"合并产物 {new_sym!r} 已在词表中")
        self.vocab[new_sym] = len(self.vocab)
        self.merges.append(pair)

    @property
    def vocab_size(self) -> int:
        # 词表 = 256 字节 + 合并产物 + 特殊 token（排在最末尾）
        return len(self.vocab) + len(self.special_tokens)

    def special_id(self, token: str) -> int:
        return len(self.vocab) + self.special_tokens.index(token)

    # ------------------------------------------------ 训练

    def train(self, text: str, vocab_size: int, min_count: int = 2, log_every: int = 2000) -> None:
        """在样本文本上学合并规则，直到词表达到 vocab_size。

        高效实现：不每轮全量重数，而是维护每对符号的总计数，
        每次合并只更新"受影响的词"的计数（增量更新），用延迟堆找最大对。

        min_count：只统计出现 ≥min_count 次的词块。中文连续段会产生近两千万种
        唯一词块，其中大半只出现一次（对"高频对统计"几乎无贡献，却拖慢每一轮
        合并）。丢掉它们是 HuggingFace tokenizers 的默认做法（min_frequency=2）。
        """
        assert not self.merges, "已训练过的 tokenizer 不能再训练"
        n_merges = vocab_size - len(self.special_tokens) - 256
        assert n_merges > 0

        # 1) 预分词并统计词块频次（只保留去重后的词块，语料再大内存也可控）
        word_counts: dict[str, int] = defaultdict(int)
        for w in iter_pretokens(text):
            word_counts[w] += 1
        if min_count > 1:
            word_counts = {w: c for w, c in word_counts.items() if c >= min_count}
        print(f"[bpe] 唯一词块 {len(word_counts):,} 种（已过滤出现 <{min_count} 次的）", flush=True)

        # 2) 词块 → 符号序列（初始每个 UTF-8 字节一个符号）
        words: list[list] = []  # [符号list, 词频]
        for w, c in word_counts.items():
            words.append([[B2U[b] for b in w.encode("utf-8")], c])

        # 3) 相邻符号对统计 + 倒排索引（对 → 含它的词下标）
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
        for i, (syms, c) in enumerate(words):
            for p in zip(syms, syms[1:]):
                pair_count[p] += c
                pair_words[p].add(i)

        # 4) 延迟最大堆：堆里可能有陈旧条目，弹出时用 pair_count 校验
        heap = [(-cnt, p) for p, cnt in pair_count.items()]
        heapq.heapify(heap)

        # 5) 反复合并最高频对
        while len(self.merges) < n_merges and heap:
            while heap:  # 弹到第一条"计数仍有效"的记录
                neg_cnt, p = heapq.heappop(heap)
                if pair_count.get(p, 0) == -neg_cnt and -neg_cnt > 0:
                    break
            else:
                break  # 堆空：没有可合并的了
            a, b = p
            new_sym = a + b
            touched: set[tuple[str, str]] = set()
            for i in list(pair_words[p]):
                syms, c = words[i]
                new_syms = merge_symbols(syms, p, new_sym)
                if len(new_syms) == len(syms):
                    continue  # 陈旧索引：这个词里已经没有这个对了（每次合并必然变短）
                for q in zip(syms, syms[1:]):        # 撤销旧符号对的计数
                    pair_count[q] -= c
                for q in zip(new_syms, new_syms[1:]):  # 登记新符号对的计数
                    pair_count[q] += c
                    pair_words[q].add(i)
                    if pair_count[q] > 0:
                        touched.add(q)
                words[i][0] = new_syms
            pair_count.pop(p, None)
            pair_words.pop(p, None)
            self._add_merge(p)
            for q in touched:  # 计数涨过的对，压一份新记录进堆
                heapq.heappush(heap, (-pair_count[q], q))
            n = len(self.merges)
            if log_every and n % log_every == 0:
                top = max(pair_count.items(), key=lambda kv: kv[1])
                print(f"[bpe] 合并 {n:,}/{n_merges:,}  最新={new_sym!r}  "
                      f"当前最高频对={top[0]}×{top[1]:,}", flush=True)

        print(f"[bpe] 训练完成：{len(self.merges):,} 条合并，词表 {self.vocab_size:,}", flush=True)

    # ------------------------------------------------ 编码 / 解码

    def _word_to_ids(self, word: str) -> list[int]:
        """单个词块 → token id 序列。

        实现：双向链表 + 最小堆。经典写法是每轮全扫描找"优先级最高的可合并对"，
        复杂度 O(L²)（L=词块字节数，中文很长）；链表+堆把每次合并的影响限制在
        左右邻居，整体 O(L·logL)，长词块快 5~10 倍。堆里的陈旧条目弹出时校验丢弃。
        """
        syms = [B2U[b] for b in word.encode("utf-8")]
        vocab = self.vocab
        # 快路径 1：整个词块已经是词表里的一个 token（常见词的常态）
        whole = "".join(syms)
        hit = vocab.get(whole)
        if hit is not None:
            return [hit]
        n = len(syms)
        if n == 1:
            return [vocab[syms[0]]]
        if self._ranks is None:  # 只在第一次编码时构建（训练阶段不受影响）
            self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        ranks = self._ranks

        prev_ = list(range(-1, n - 1))          # prev_[i]：i 左边还活着的符号
        next_ = list(range(1, n + 1)); next_[n - 1] = -1
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            r = ranks.get((syms[i], syms[i + 1]))
            if r is not None:
                heap.append((r, i))
        heapq.heapify(heap)
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            r, i = pop(heap)
            if not alive[i]:
                continue
            j = next_[i]
            if j == -1 or not alive[j] or ranks.get((syms[i], syms[j])) != r:
                continue  # 陈旧条目：位置上的符号已经变了
            # 合并 j 进 i
            syms[i] += syms[j]
            alive[j] = False
            k = next_[j]
            next_[i] = k
            if k != -1:
                prev_[k] = i
                r2 = ranks.get((syms[i], syms[k]))
                if r2 is not None:
                    push(heap, (r2, i))
            p = prev_[i]
            if p != -1:
                r2 = ranks.get((syms[p], syms[i]))
                if r2 is not None:
                    push(heap, (r2, p))
        return [vocab[syms[i]] for i in range(n) if alive[i]]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        cache = self._cache
        for piece in self._special_pat.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                ids.append(self.special_id(piece))
                continue
            for w in iter_pretokens(piece):
                got = cache.get(w)
                if got is None:
                    got = self._word_to_ids(w)
                    if len(cache) >= 2_000_000:  # 防止缓存无限膨胀
                        cache.clear()
                    cache[w] = got
                ids.extend(got)
        return ids

    def decode(self, ids: list[int]) -> str:
        n_base = len(self.vocab)
        parts = []
        for i in ids:
            if i < n_base:
                parts.append(self.id_to_token(i))
            else:
                parts.append(self.special_tokens[i - n_base])  # 特殊 token 原样输出
        buf = bytes(U2B[ch] for ch in "".join(parts))
        return buf.decode("utf-8", errors="replace")

    def id_to_token(self, i: int) -> str:
        if not hasattr(self, "_id2tok"):
            self._id2tok = {v: k for k, v in self.vocab.items()}
        return self._id2tok[i]

    # ------------------------------------------------ 存取

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "byte-level-bpe",
            "merges": [list(p) for p in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(p) for p in data["merges"]],
                   special_tokens=tuple(data["special_tokens"]))
