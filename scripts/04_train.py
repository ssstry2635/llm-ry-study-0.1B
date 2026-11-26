"""M5：训练脚本（nanoGPT 风格，全流程手写：AdamW + warmup/cosine + bf16 + 梯度累积 + 断点续训）。

用法：
    python scripts/04_train.py --config test --max-iters 100      # 迷你模型冒烟测试（约 2 分钟）
    python scripts/04_train.py --config 012b                      # 正式 0.1B 训练（自动从断点续训）
    python scripts/04_train.py --config 012b --fresh              # 推倒重来
日志：logs/train_<配置>.log 与 logs/metrics.csv；断点：checkpoints/<配置>/ckpt.pt
"""
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.bpe import BPETokenizer  # noqa: E402
from llm0.config import GPTConfig, gpt_0_1b, gpt_test  # noqa: E402
from llm0.dataset import estimate_loss, get_batch, open_bin  # noqa: E402
from llm0.model import GPT  # noqa: E402

DATA_DIR = ROOT / "data" / "tokenized"

# 两种预设：test 追求快，012b 是正式目标
PRESETS = {
    "test": dict(
        cfg=gpt_test(), batch_size=32, grad_accum=1, max_iters=300,
        lr=1e-3, warmup=20, eval_interval=50, ckpt_interval=100,
    ),
    "012b": dict(
        cfg=gpt_0_1b(), batch_size=4, grad_accum=16,  # 有效 batch = 4×1024×16 ≈ 6.6 万 token/步
        max_iters=15000,                              # ≈ 10 亿 token（按实测吞吐可再调）
        lr=4e-4, warmup=300, eval_interval=250, ckpt_interval=500,
    ),
}


def get_lr(it: int, peak: float, warmup: int, total: int) -> float:
    """warmup 线性升 → cosine 降到峰值的 10%。"""
    if it < warmup:
        return peak * (it + 1) / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.45 * peak * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(PRESETS), default="012b")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="忽略已有断点从头训")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="梯度检查点：用约 30%% 速度换显存（爆显存时开）")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="token 数据目录（含 train.bin/val.bin）")
    ap.add_argument("--pause-every", type=int, default=0,
                    help="每训练多少步休息一次（0=不休息）。休息发生在存档点，进度零损失")
    ap.add_argument("--pause-sec", type=int, default=600, help="每次休息的秒数（默认 600 = 10 分钟）")
    args = ap.parse_args()
    p = PRESETS[args.config]
    max_iters = args.max_iters or p["max_iters"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = ROOT / "checkpoints" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"train_{args.config}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---------------- 数据 ----------------
    tok_path = ROOT / "data" / "tokenizer" / "tokenizer.json"
    cfg: GPTConfig = p["cfg"]
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        assert tok.vocab_size == cfg.vocab_size, \
            f"词表 {tok.vocab_size} 与模型配置 {cfg.vocab_size} 不一致"
    else:
        log("警告：tokenizer.json 不存在，跳过词表校验（合成数据调试时可接受）")
    data_dir = Path(args.data_dir)
    train_data = open_bin(data_dir / "train.bin")
    val_data = open_bin(data_dir / "val.bin")
    log(f"数据：train {len(train_data):,} / val {len(val_data):,} tokens；词表 {cfg.vocab_size:,}")

    # ---------------- 模型 / 优化器 ----------------
    model = GPT(cfg).to(device)
    if args.grad_ckpt:
        # 梯度检查点：前向时不存激活、反向时重算，用约 30% 速度换大幅省显存
        for block in model.h:
            orig_forward = block.forward
            block.forward = (lambda x, _f=orig_forward: torch.utils.checkpoint.checkpoint(
                _f, x, use_reentrant=False))
    n_params = model.num_params()
    log(f"模型：{n_params / 1e6:.1f}M 参数（不含 embedding {model.num_params(non_embedding=True) / 1e6:.1f}M），"
        f"{cfg.n_layer} 层 × {cfg.n_embd} 维，attn={cfg.attn_impl}")

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=p["lr"], betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu")

    best_val = float("inf")
    start_iter = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt.get("best_val", float("inf"))
        log(f"从断点续训：iter {start_iter}（best_val {best_val:.4f}）")
    elif args.fresh:
        log("指定 --fresh，从头训练")

    metrics_path = log_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    metrics_f = open(metrics_path, "a", newline="", encoding="utf-8")
    metrics = csv.writer(metrics_f)
    if new_file:
        metrics.writerow(["config", "iter", "train_loss", "val_loss", "lr", "tok_per_s", "elapsed_s"])

    # ---------------- 训练循环 ----------------
    seq_len = cfg.seq_len
    tokens_per_step = p["batch_size"] * seq_len * p["grad_accum"]
    log(f"开始训练：{max_iters} 步 × {tokens_per_step:,} token/步 ≈ "
        f"{max_iters * tokens_per_step / 1e9:.2f}B token 总量")
    t_window = time.time()
    tokens_window = 0
    model.train()

    for it in range(start_iter, max_iters):
        lr = get_lr(it, p["lr"], p["warmup"], max_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 梯度累积：小显存跑大 batch 的手法——每个 micro-batch 反向传播累积梯度，
        # 攒够 grad_accum 份才更新一次参数（数学上等价于大 batch 一步）
        optimizer.zero_grad(set_to_none=True)
        for _ in range(p["grad_accum"]):
            x, y = get_batch(train_data, p["batch_size"], seq_len, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, loss = model(x, y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防爆炸
        optimizer.step()

        tokens_window += tokens_per_step
        if it % 20 == 0 or it == max_iters - 1:
            dt = time.time() - t_window
            tok_per_s = tokens_window / max(dt, 1e-9)
            eta_h = (max_iters - it) * tokens_per_step / max(tok_per_s, 1) / 3600
            log(f"iter {it:>6} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tok_per_s / 1000:.1f}k tok/s | 显存 {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
                f" | 预计剩余 {eta_h:.1f}h" if device.type == "cuda" else
                f"iter {it:>6} | loss {loss.item():.4f} | lr {lr:.2e} | {tok_per_s / 1000:.1f}k tok/s")
            t_window, tokens_window = time.time(), 0

        if (it + 1) % p["eval_interval"] == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   batch_size=p["batch_size"], seq_len=seq_len, device=device)
            metrics.writerow([args.config, it + 1, f"{losses['train']:.4f}",
                              f"{losses['val']:.4f}", f"{lr:.2e}",
                              f"{tokens_window:.0f}", f"{time.time():.0f}"])
            metrics_f.flush()
            log(f"== 评估 iter {it + 1}: train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}（ln(V) 基线 {math.log(cfg.vocab_size):.2f}）")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg),
                            "iter": it + 1, "val_loss": best_val},
                           out_dir / "best_model.pt")

        if (it + 1) % p["ckpt_interval"] == 0 or it == max_iters - 1:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": asdict(cfg), "iter": it, "best_val": best_val,
                        "max_iters": max_iters}, ckpt_path)
            log(f"断点已保存：iter {it}")

        # 间歇休息：只在刚存完档的节点睡（进度零损失），顺便释放显存缓存给电脑降温
        if args.pause_every and (it + 1) % args.pause_every == 0 and it != max_iters - 1:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            back = time.strftime("%H:%M", time.localtime(time.time() + args.pause_sec))
            log(f"训练暂停休息 {args.pause_sec // 60} 分钟（GPU 降温），预计 {back} 恢复")
            time.sleep(args.pause_sec)
            log("休息结束，继续训练")

    metrics_f.close()
    log(f"训练结束：{max_iters} 步完成，best val loss {best_val:.4f}。"
        f"生成试试：python scripts/05_generate.py --config {args.config}")


if __name__ == "__main__":
    main()
