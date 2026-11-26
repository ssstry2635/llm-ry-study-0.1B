"""M1 清洗函数的验收测试。

运行：python -m pytest tests/test_data_prep.py -v
全绿 = 4 个 TODO 函数实现正确。
"""
import textwrap

import pytest

from llm0.data_prep import (
    dedupe,
    han_ratio,
    is_good_en_doc,
    is_good_zh_doc,
    iter_tinystories,
    normalize_whitespace,
    stable_bucket,
)

# ---------------------------------------------------------------- normalize_whitespace


class TestNormalizeWhitespace:
    def test_nfc_composes_combining_marks(self):
        # "e" + 组合重音符 → 单码位 "é"
        assert normalize_whitespace("cafe\u0301") == "caf\u00e9"

    def test_crlf_and_cr_become_lf(self):
        assert normalize_whitespace("a\r\nb\rc") == "a\nb\nc"

    def test_tab_fullwidth_nbsp_become_space(self):
        assert normalize_whitespace("a\tb\u3000c\xa0d") == "a b c d"

    def test_strip_lines_and_collapse_inner_spaces(self):
        assert normalize_whitespace("  hello    world  \n  foo  ") == "hello world\nfoo"

    def test_three_or_more_newlines_become_two(self):
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_two_newlines_preserved(self):
        assert normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_whitespace_only_lines_count_as_blank(self):
        # 中间两行只有空格：先按行 strip 变成空行，再压缩 → 只剩一个段落分隔
        assert normalize_whitespace("a\n   \n \t \nb") == "a\n\nb"

    def test_strip_outer(self):
        assert normalize_whitespace("\n\n  a  \n\n") == "a"

    def test_idempotent(self):
        messy = "  第一段\u3000内容  \r\n\r\n\r\n\r\n第二段\t内容 \n"
        once = normalize_whitespace(messy)
        assert normalize_whitespace(once) == once
        assert once == "第一段 内容\n\n第二段 内容"


# ---------------------------------------------------------------- is_good_zh_doc

ZH_PARA = "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"


class TestIsGoodZhDoc:
    def test_empty_and_whitespace(self):
        assert not is_good_zh_doc("")
        assert not is_good_zh_doc("   \n\n  ")

    def test_length_boundary(self):
        assert is_good_zh_doc("中" * 100)
        assert not is_good_zh_doc("中" * 99)

    def test_length_ignores_whitespace(self):
        # 99 个汉字 + 一堆空白，仍然不够 100
        assert not is_good_zh_doc("中 " * 99)

    def test_real_paragraph_with_some_latin(self):
        text = ZH_PARA * 3 + "（Artificial Intelligence, AI）"
        assert is_good_zh_doc(text)

    def test_table_junk_rejected(self):
        assert not is_good_zh_doc("| 2001 | 2002 | 2003 | 2004 |\n" * 30)

    def test_han_ratio_boundary(self):
        assert is_good_zh_doc("中" * 50 + "1" * 50)  # 恰好 0.5 → 保留
        assert not is_good_zh_doc("中" * 49 + "1" * 51)  # 0.49 → 丢

    def test_rare_han_counts_as_han(self):
        # 扩展区生僻字（\p{Han} 能匹配，\u4e00-\u9fff 匹配不到）
        assert is_good_zh_doc("\U00020000" * 100)


# ---------------------------------------------------------------- is_good_en_doc

EN_PARA = textwrap.dedent("""\
    Once upon a time there was a little boy named Ben. Ben loved to explore
    the world around him. He saw many amazing things, like beautiful vases
    that were on display in a store. One day, Ben was walking through the
    store when he came across a very special vase.""")


class TestIsGoodEnDoc:
    def test_empty(self):
        assert not is_good_en_doc("")

    def test_real_story(self):
        assert is_good_en_doc(EN_PARA)

    def test_length_boundary(self):
        assert is_good_en_doc("a" * 200)
        assert not is_good_en_doc("a" * 199)

    def test_digit_junk_rejected(self):
        assert not is_good_en_doc("1234567890 " * 30)

    def test_letter_ratio_boundary(self):
        assert is_good_en_doc("a" * 120 + "1" * 80)  # 0.6 → 保留
        assert not is_good_en_doc("a" * 119 + "1" * 81)  # 0.595 → 丢


# ---------------------------------------------------------------- dedupe


class TestDedupe:
    def test_keeps_first_occurrence_in_order(self):
        assert list(dedupe(["a", "b", "a", "c", "b", "a"])) == ["a", "b", "c"]

    def test_unicode(self):
        assert list(dedupe(["中文", "中文", "英文"])) == ["中文", "英文"]

    def test_is_lazy(self):
        # 生成器在第三次取值时才会炸；如果 dedupe 偷懒先转 list，前两次 next 就会炸
        def source():
            yield "x"
            yield "y"
            raise RuntimeError("dedupe 不应该提前消费整个输入流")

        it = dedupe(source())
        assert next(it) == "x"
        assert next(it) == "y"
        with pytest.raises(RuntimeError):
            next(it)


# ---------------------------------------------------------------- 管道工程（已实现，顺便验一下）


def test_han_ratio():
    assert han_ratio("中文abc") == pytest.approx(2 / 5)
    assert han_ratio("   ") == 0.0


def test_stable_bucket_is_deterministic_and_in_range():
    assert stable_bucket("hello") == stable_bucket("hello")
    assert 0 <= stable_bucket("hello") < 100
    assert stable_bucket("hello") != stable_bucket("hello!")  # 大概率不同


def test_iter_tinystories_splits_on_marker(tmp_path):
    f = tmp_path / "ts.txt"
    f.write_text("story one\nline two\n<|endoftext|>\nstory two\n<|endoftext|>\n", encoding="utf-8")
    stories = list(iter_tinystories(f))
    assert stories == ["story one\nline two\n", "story two\n"]


def test_iter_tinystories_max_bytes_drops_partial(tmp_path):
    f = tmp_path / "ts.txt"
    f.write_text("aaa\n<|endoftext|>\nbbb\n<|endoftext|>\nccc\n", encoding="utf-8")
    # 读到第一个分隔符时已超过 3 字节 → 只产出第一个故事
    assert list(iter_tinystories(f, max_bytes=3)) == ["aaa\n"]
