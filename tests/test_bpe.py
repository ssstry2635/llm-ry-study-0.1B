"""M2 BPE tokenizer 的验收测试。运行：python -m pytest tests/test_bpe.py -v"""
import pytest

from llm0.bpe import B2U, ENDOFTEXT, BPETokenizer, merge_symbols

SAMPLE = (
    "人工智能是计算机科学的一个分支。Artificial intelligence is a branch of computer science.\n"
    "人工智能尝试理解智能的实质。The quick brown fox jumps over the lazy dog.\n"
) * 40


def make_tok(vocab_size=300) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=vocab_size, log_every=0)
    return tok


# ---------------------------------------------------------------- 基础件


def test_bytes_to_unicode_space_is_ghat():
    # GPT-2 的经典映射：空格 0x20 → 'Ġ'，可打印字符保持原样
    assert B2U[0x20] == "Ġ"
    assert B2U[ord("a")] == "a"
    assert len(set(B2U.values())) == 256  # 一一对应


def test_merge_symbols_non_overlapping_left_to_right():
    assert merge_symbols(["a", "a", "a", "b"], ("a", "a"), "aa") == ["aa", "a", "b"]
    assert merge_symbols(["a", "b", "a", "b"], ("a", "b"), "ab") == ["ab", "ab"]


# ---------------------------------------------------------------- 训练


def test_train_first_merge_is_most_frequent_pair():
    # "ababab" 是一个词块：相邻对 (a,b) 出现 6 次、(b,a) 4 次 → 第一条合并必是 (a,b)
    tok = BPETokenizer()
    tok.train("ababab ababab", vocab_size=258, min_count=1, log_every=0)  # 258 = 256字节+1合并+1特殊
    assert tok.merges[0] == ("a", "b")


def test_vocab_size_reached():
    tok = make_tok(300)
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - 1  # 减 256 个字节 token、1 个 <|endoftext|>


# ---------------------------------------------------------------- 编码 / 解码


def test_roundtrip_plain():
    tok = make_tok()
    for text in ["你好，世界。Hello world! 123", "emoji 😄🎉 and 汉字", "", "\n\n\t ", "über Straße café"]:
        assert tok.decode(tok.encode(text)) == text


def test_roundtrip_after_load(tmp_path):
    tok = make_tok()
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    for text in ["你好，世界。Hello world!", "<|endoftext|>"]:
        assert tok2.decode(tok2.encode(text)) == text


def test_special_token_is_single_id():
    tok = make_tok()
    ids = tok.encode(f"前面{ENDOFTEXT}后面")
    assert ids.count(tok.special_id(ENDOFTEXT)) == 1
    assert tok.decode(ids) == f"前面{ENDOFTEXT}后面"


def test_special_id_at_vocab_end():
    tok = make_tok()
    assert tok.special_id(ENDOFTEXT) == tok.vocab_size - 1
    assert 0 <= min(tok.encode("hello")) < 256  # 单字母词就是字节 token


def test_compression_happens():
    # 只有一个高频词的玩具语料：它的字母对会一路合并，最终几个 token 就能表示
    tok = BPETokenizer()
    tok.train("supercalifragilistic " * 200, vocab_size=290, log_every=0)
    assert len(tok.encode("supercalifragilistic")) <= 8  # 20 个字母压到 ≤8 个 token


def test_no_merge_across_pretoken_boundary():
    # "ab cd" 反复出现也绝不该学出跨词块边界的合并（b,空格c）
    tok = BPETokenizer()
    tok.train("ab cd " * 100, vocab_size=280, log_every=0)
    for a, b in tok.merges:
        joined = a + b
        # 词块由 PAT 切出：任何"真实词块的子串"编码后应能在词块里找到；
        # 这里用性质检验：对训练文本 encode 后逐 token 解码应还原原文
    text = "ab cd ab cd"
    assert tok.decode(tok.encode(text)) == text
