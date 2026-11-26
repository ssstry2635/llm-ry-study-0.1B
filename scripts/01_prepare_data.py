"""M1 第二步：清洗语料 → data/clean/train.jsonl + val.jsonl + stats.json

流程（每种语言各走一遍）：
    原始文档流 → normalize_whitespace → (中文) 繁转简 → 质量过滤 → 精确去重
              → 按内容哈希分桶：1% 进 val，99% 进 train → 写 JSONL

输出一行一篇文档：{"text": "...", "lang": "zh"|"en"}
JSONL 而不是纯文本，是因为纯文本需要一个"文档分隔符"，而任何分隔符都可能在正文里出现。

前置条件：tests/test_data_prep.py 全绿（4 个 TODO 函数已实现）。
用法：
    python scripts/01_prepare_data.py            # 全量
    python scripts/01_prepare_data.py --limit 2000   # 每种语言只处理 2000 篇，用来快速调试
"""
import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.data_prep import (  # noqa: E402
    dedupe,
    is_good_en_doc,
    is_good_zh_doc,
    iter_tinystories,
    iter_wiki_zh,
    normalize_whitespace,
    stable_bucket,
    to_simplified,
)

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

VAL_BUCKETS = 100          # 桶 0 作为 val → 1%
EN_MAX_BYTES = None        # None = TinyStories 全部用；想降低英文占比就改小（字节数）

# token 数粗估系数，M2 训完 tokenizer 后会用真实值替换
ZH_TOKENS_PER_CHAR = 1.2   # 32k byte-level BPE 下，中文大约 1.2 token / 字
EN_CHARS_PER_TOKEN = 4.0   # 英文大约 4 字符 / token
MEASURED_TOKENS_PER_SEC = 24_000  # verify_env.py 估算的 0.1B 训练吞吐


def clean_stream(lang: str, docs: Iterable[str], keep, stats: Counter) -> Iterator[str]:
    """规整 + （中文）繁转简 + 质量过滤，统计写进 stats。"""
    for raw in docs:
        stats[f"{lang}_in"] += 1
        text = normalize_whitespace(raw)
        if lang == "zh":
            text = to_simplified(text)
        if not keep(text):
            stats[f"{lang}_filtered"] += 1
            continue
        stats[f"{lang}_passed"] += 1
        yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每种语言最多处理多少篇（调试用）")
    args = ap.parse_args()

    CLEAN.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    t0 = time.time()

    sources = [
        ("zh", iter_wiki_zh(RAW / "wiki_zh" / "20231101.zh"), is_good_zh_doc),
        ("en", iter_tinystories(RAW / "tinystories" / "TinyStoriesV2-GPT4-train.txt",
                                max_bytes=EN_MAX_BYTES), is_good_en_doc),
    ]

    with open(CLEAN / "train.jsonl", "w", encoding="utf-8") as f_train, \
         open(CLEAN / "val.jsonl", "w", encoding="utf-8") as f_val:
        for lang, docs, keep in sources:
            if args.limit:
                docs = islice(docs, args.limit)
            cleaned = clean_stream(lang, docs, keep, stats)
            for text in tqdm(dedupe(cleaned), desc=f"[{lang}]", unit="doc"):
                split = "val" if stable_bucket(text, VAL_BUCKETS) == 0 else "train"
                f = f_val if split == "val" else f_train
                f.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False) + "\n")
                stats[f"{lang}_{split}_docs"] += 1
                stats[f"{lang}_{split}_chars"] += len(text)

    # ---------------- 统计报告 ----------------
    report = {}
    for lang in ("zh", "en"):
        passed, kept = stats[f"{lang}_passed"], stats[f"{lang}_train_docs"] + stats[f"{lang}_val_docs"]
        chars = stats[f"{lang}_train_chars"] + stats[f"{lang}_val_chars"]
        est_tokens = chars * ZH_TOKENS_PER_CHAR if lang == "zh" else chars / EN_CHARS_PER_TOKEN
        report[lang] = {
            "docs_in": stats[f"{lang}_in"],
            "docs_filtered": stats[f"{lang}_filtered"],
            "docs_dup": passed - kept,
            "docs_kept": kept,
            "chars_kept": chars,
            "est_tokens": int(est_tokens),
            "train_docs": stats[f"{lang}_train_docs"],
            "val_docs": stats[f"{lang}_val_docs"],
        }
    total_tokens = sum(r["est_tokens"] for r in report.values())
    report["total"] = {
        "est_tokens": total_tokens,
        "zh_token_share": round(report["zh"]["est_tokens"] / max(total_tokens, 1), 3),
        "est_hours_one_epoch": round(total_tokens / MEASURED_TOKENS_PER_SEC / 3600, 1),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (CLEAN / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'':6}{'输入':>9}{'过滤':>9}{'重复':>9}{'保留':>9}{'字符数':>13}{'估算token':>13}")
    for lang in ("zh", "en"):
        r = report[lang]
        print(f"{lang:6}{r['docs_in']:>9,}{r['docs_filtered']:>9,}{r['docs_dup']:>9,}"
              f"{r['docs_kept']:>9,}{r['chars_kept']:>13,}{r['est_tokens']:>13,}")
    t = report["total"]
    print("-" * 62)
    print(f"合计估算 {t['est_tokens']:,} token，中文占 {t['zh_token_share']:.0%}；"
          f"0.1B 模型跑一遍约 {t['est_hours_one_epoch']} 小时；清洗耗时 {t['elapsed_sec']} 秒")
    print(f"输出：{CLEAN / 'train.jsonl'}  /  {CLEAN / 'val.jsonl'}  /  stats.json")


if __name__ == "__main__":
    main()
