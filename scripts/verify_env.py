"""M0 环境自检：确认 Python / PyTorch / CUDA / bf16 / 显存 / 算力全部就绪。

用法：
    python scripts/verify_env.py
全部打印 [OK] 即环境合格；出现 [FAIL] 按提示修复。
"""
import sys
import time

PASS, FAIL = "[OK]  ", "[FAIL]"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    mark = PASS if ok else FAIL
    if not ok:
        failures += 1
    print(f"{mark}{name}" + (f"  — {detail}" if detail else ""))


def main() -> None:
    print("=" * 60)
    print("环境自检：从零手写 tokenizer + transformer 训练 0.1B")
    print("=" * 60)

    check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

    try:
        import torch
    except ImportError as e:
        check("PyTorch 已安装", False, str(e))
        print("\n环境不合格。请先安装 PyTorch（见 README.md）。")
        sys.exit(1)
    check("PyTorch 已安装", True, torch.__version__)

    check("CUDA 可用", torch.cuda.is_available(),
          torch.version.cuda if torch.cuda.is_available() else "未检测到 GPU 版 PyTorch")
    if not torch.cuda.is_available():
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    check("显卡", True, f"{props.name}, 显存 {vram_gb:.1f} GB, 算力 sm_{props.major}{props.minor}")
    check("显存 >= 6GB", vram_gb >= 6, f"{vram_gb:.1f} GB（0.1B bf16 训练下限约 5~6GB）")
    check("bf16 支持", torch.cuda.is_bf16_supported(), "混合精度训练用")

    # 快速算力实测：bf16 大矩阵乘，估算有效 TFLOPS
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):  # 预热
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * n**3 * iters / dt / 1e12
    check("GPU 实测算力", tflops > 3, f"{tflops:.1f} TFLOPS (bf16 matmul)")

    # 估算 0.1B 训练吞吐与耗时（按 matmul 效率的 ~55% 折算训练效率）
    params = 1.1e8
    tok_per_s = tflops * 0.55e12 / (6 * params)
    hours_per_billion = 1e9 / tok_per_s / 3600
    print("-" * 60)
    print(f"估算：0.1B 模型训练吞吐 ≈ {tok_per_s:,.0f} token/s")
    print(f"估算：每 10 亿 token 需 ≈ {hours_per_billion:.1f} 小时")

    print("=" * 60)
    if failures:
        print(f"有 {failures} 项不合格，请修复后重跑。")
        sys.exit(1)
    print("全部通过 ✔  可以开始 M1 数据准备。")


if __name__ == "__main__":
    main()
