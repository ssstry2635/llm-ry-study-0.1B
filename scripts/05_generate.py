"""M6：加载训练好的模型，生成文本。

用法：
    python scripts/05_generate.py --config test --prompt "人工智能"
    python scripts/05_generate.py --config 012b --prompt "从前有一座山" --max-tokens 200
    python scripts/05_generate.py --config 012b --interactive     # 简易对话（每轮独立，无记忆）
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.bpe import BPETokenizer  # noqa: E402
from llm0.config import GPTConfig  # noqa: E402
from llm0.model import GPT  # noqa: E402


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """加载检查点（只信张量与基础类型，weights_only=True 防反序列化攻击）。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def generate_once(model: GPT, tok: BPETokenizer, prompt: str, args, device) -> str:
    ids = tok.encode(prompt) if prompt else [tok.special_id("<|endoftext|>")]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p if args.top_p > 0 else None)
    return tok.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="012b")
    ap.add_argument("--ckpt", default=None, help="默认 checkpoints/<config>/best_model.pt，没有则用 ckpt.pt")
    ap.add_argument("--prompt", default="从前有一座山，")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.0, help="0 = 不启用")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ROOT / "checkpoints" / args.config
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = base / "best_model.pt" if (base / "best_model.pt").exists() else base / "ckpt.pt"
    assert ckpt_path.exists(), f"找不到检查点：{ckpt_path}"

    tok = BPETokenizer.load(ROOT / "data" / "tokenizer" / "tokenizer.json")
    model, ckpt = load_checkpoint(ckpt_path, device)
    print(f"[已加载 {ckpt_path.name}，iter {ckpt.get('iter', '?')}，val {ckpt.get('val_loss', ckpt.get('best_val', '?'))}]")

    if args.interactive:
        print("简易对话（输入 exit 退出；这是预训练模型，只会续写，不懂指令）")
        while True:
            try:
                prompt = input("\n你： ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("exit", "quit", ""):
                break
            text = generate_once(model, tok, prompt, args, device)
            print(f"模型： {text[len(prompt):].strip()}")
    else:
        print("=" * 60)
        print(generate_once(model, tok, args.prompt, args, device))
        print("=" * 60)


if __name__ == "__main__":
    main()
