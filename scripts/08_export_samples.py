"""Save unedited, seeded examples from the local historical best checkpoint.

Run with the training environment: python scripts/08_export_samples.py
Results vary across devices/software even with a fixed seed.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm0.bpe import BPETokenizer
from llm0.config import GPTConfig
from llm0.model import GPT


def main():
    ckpt_path = ROOT / "checkpoints/012b/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(ROOT / "data/tokenizer/tokenizer.json")
    samples = []
    for prompt in ["Once upon a time", "从前有一座山，", "人工智能是一门"]:
        torch.manual_seed(42)
        x = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=120, temperature=.8, top_k=50)
        samples.append({"prompt": prompt, "output": tok.decode(out[0].tolist())})
    report = {"checkpoint": "best_model.pt", "step": ckpt["iter"],
              "seed": 42, "temperature": .8, "top_k": 50, "max_new_tokens": 120,
              "torch": str(torch.__version__), "device": str(device), "samples": samples}
    dest = ROOT / "reports"
    dest.mkdir(exist_ok=True)
    (dest / "generation_samples.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines = ["# 实际续写样例", "", "以下为本地历史最佳权重的原始输出，没有人工改写或筛选。", "",
             "参数：step 14,750；seed=42；temperature=0.8；top-k=50；最多新增 120 tokens。",
             "不同设备和软件版本仍可能产生不同文本。模型是预训练续写模型，没有经过 SFT。", ""]
    for sample in samples:
        lines += ["## 输入："+sample["prompt"], "", "```text", sample["output"], "```", ""]
    lines += ["配置与原始输出见 [generation_samples.json](../reports/generation_samples.json)。", "",
              "这些样例用来展示实际生成行为，不代表基准成绩或可靠知识。", ""]
    (ROOT / "docs/GENERATION_SAMPLES.md").write_text("\n".join(lines), encoding="utf-8")
    files = [ckpt_path, ROOT / "data/tokenizer/tokenizer.json"]
    artifacts = []
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b""):
                digest.update(chunk)
        artifacts.append({"path": file.relative_to(ROOT).as_posix(), "bytes": file.stat().st_size,
                          "sha256": digest.hexdigest()})
    manifest = {"model_parameters": model.num_params(), "best_step": ckpt["iter"],
                "best_val_loss": ckpt["val_loss"], "config": ckpt["config"], "artifacts": artifacts}
    (dest / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Exported three unedited samples and checkpoint/tokenizer SHA-256 manifest")


if __name__ == "__main__":
    main()
