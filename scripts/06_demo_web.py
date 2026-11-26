"""网页演示：在浏览器里玩自己手搓的 0.1B 模型（零第三方依赖，只用标准库）。

用法：
    python scripts/06_demo_web.py            # 打开 http://127.0.0.1:8000
    python scripts/06_demo_web.py --port 8000
演示时模型常驻显存，每次生成只做一次前向循环，响应几秒钟。
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm0.bpe import BPETokenizer  # noqa: E402
from llm0.config import GPTConfig  # noqa: E402
from llm0.model import GPT  # noqa: E402

TOK_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
CKPT_DIR = ROOT / "checkpoints" / "012b"

HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>我的 0.1B 大模型 · 从零手搓</title>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--line:#2a2e3d;--txt:#e8eaf2;--dim:#8b91a7;--acc:#6c8cff;--gold:#e8b34b}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--txt);font-family:system-ui,"Microsoft YaHei",sans-serif;min-height:100vh;display:flex;justify-content:center;padding:24px 16px}
.wrap{width:100%;max-width:820px}
h1{font-size:22px;margin-bottom:4px}
h1 .gold{color:var(--gold)}
.sub{color:var(--dim);font-size:13px;margin-bottom:18px;line-height:1.7}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.badge{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:12px;color:var(--dim)}
.badge b{color:var(--txt);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
textarea{width:100%;min-height:88px;background:#11131c;border:1px solid var(--line);border-radius:8px;color:var(--txt);padding:12px;font-size:15px;line-height:1.6;resize:vertical;font-family:inherit}
textarea:focus{outline:1px solid var(--acc)}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:12px}
button{background:var(--acc);border:0;color:#fff;border-radius:8px;padding:10px 22px;font-size:15px;cursor:pointer;font-weight:600}
button:disabled{opacity:.45;cursor:wait}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--dim);padding:6px 12px;font-size:13px;font-weight:400}
label{font-size:13px;color:var(--dim)}
input[type=range]{accent-color:var(--acc)}
.output{white-space:pre-wrap;line-height:1.9;font-size:15px;min-height:60px}
.output .p{color:var(--gold)}
.meta{color:var(--dim);font-size:12px;margin-top:10px}
.foot{color:var(--dim);font-size:12px;line-height:1.8;margin-top:6px}
</style></head><body><div class="wrap">
<h1>我的 <span class="gold">0.1B</span> 大模型</h1>
<div class="sub">分词器（BPE）与 Transformer 全部从零手写，在笔记本 RTX 4060 上用 9.8 亿 token 训练。<br>它只会「续写」：给一句开头，它接着往下写。事实未经验证，请把它当作会写字的鹦鹉而不是百科全书。</div>
<div class="badges">
<span class="badge">参数量 <b>110,119,680</b></span>
<span class="badge">验证损失 <b id="bval">…</b></span>
<span class="badge">词表 <b>32,768</b>（自训 BPE）</span>
<span class="badge">训练 <b>15,000 步 · 9.8 亿 token</b></span>
</div>
<div class="card">
<textarea id="prompt">人工智能的发展历史可以追溯到</textarea>
<div class="row">
<button id="go" onclick="gen()">✦ 续写</button>
<label>长度 <input type="range" id="len" min="40" max="300" value="140" oninput="lenv.value=this.value"> <b id="lenv">140</b></label>
<label>温度 <input type="range" id="temp" min="0.1" max="1.4" step="0.1" value="0.8" oninput="tempv.value=this.value"> <b id="tempv">0.8</b></label>
</div>
<div class="row" id="exs"></div>
</div>
<div class="card"><div class="output" id="out">输出会出现在这里 ▸</div><div class="meta" id="meta"></div></div>
<div class="foot">架构：Llama 风格（RoPE 旋转位置编码 / RMSNorm / SwiGLU / 注意力因果掩码 / 输入输出权重共享）· 代码见 llm0/ 与 scripts/</div>
</div>
<script>
const EX=[["📖 英文小故事","Once upon a time, a little girl named Lily"],["🏔 中文故事","从前有一座山，山里有一座庙，"],["🧠 科普续写","人工智能的发展历史可以追溯到"],["🔬 学术腔","深度学习模型的表现主要取决于"],["🐱 萌系","The little cat named Spot went to the garden."]];
const exs=document.getElementById("exs");
EX.forEach(([t,p])=>{const b=document.createElement("button");b.className="ghost";b.textContent=t;b.onclick=()=>{document.getElementById("prompt").value=p};exs.appendChild(b)});
async function gen(){
 const p=document.getElementById("prompt").value.trim();
 if(!p){document.getElementById("out").textContent="请先输入一个开头～";return}
 const btn=document.getElementById("go");btn.disabled=true;
 document.getElementById("out").textContent="正在生成…";
 const t0=performance.now();
 try{
  const r=await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({prompt:p,max_tokens:+document.getElementById("len").value,temperature:+document.getElementById("temp").value,top_k:50})});
  const d=await r.json();
  document.getElementById("out").innerHTML='<span class="p"></span>';
  document.getElementById("out").querySelector(".p").textContent=d.prompt;
  document.getElementById("out").append(d.continuation);
  document.getElementById("meta").textContent=`${d.new_tokens} 个新 token · ${d.seconds.toFixed(1)} 秒 · ${d.tokens_per_s.toFixed(0)} token/秒 · 模型 iter ${d.iter}`;
 }catch(e){document.getElementById("out").textContent="生成失败："+e}
 btn.disabled=false;
}
fetch("/api/info").then(r=>r.json()).then(d=>{document.getElementById("bval").textContent=d.val_loss});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(TOK_PATH)
    ckpt_path = CKPT_DIR / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_loss = ckpt.get("val_loss", ckpt.get("best_val", 0))
    print(f"[demo] 模型已加载：{ckpt_path.name}，val {val_loss:.4f}，设备 {device.type}")

    gen_lock = __import__("threading").Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                info = json.dumps({"val_loss": round(float(val_loss), 4),
                                   "iter": ckpt.get("iter", "?")}).encode()
                self._send(200, info, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path != "/api/generate":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    body = raw.decode("utf-8")
                except UnicodeDecodeError:  # 兼容个别终端发来的 GBK 编码
                    body = raw.decode("gbk", errors="replace")
                req = json.loads(body)
                prompt = str(req.get("prompt", "")).strip()[:2000]
                max_tokens = min(int(req.get("max_tokens", 140)), 400)
                temperature = min(max(float(req.get("temperature", 0.8)), 0.05), 2.0)
                top_k = int(req.get("top_k", 50))
                assert prompt, "prompt 不能为空"
            except Exception as e:  # noqa: BLE001
                self._send(400, json.dumps({"error": f"参数错误：{e}"}).encode(), "application/json")
                return
            with gen_lock, torch.inference_mode():
                ids = tok.encode(prompt)
                x = torch.tensor([ids], dtype=torch.long, device=device)
                t0 = time.time()
                out = model.generate(x, max_new_tokens=max_tokens,
                                     temperature=temperature, top_k=top_k, top_p=None)
                dt = time.time() - t0
            text = tok.decode(out[0].tolist())
            resp = json.dumps({
                "prompt": prompt, "continuation": text[len(prompt):],
                "new_tokens": out.size(1) - len(ids), "seconds": round(dt, 2),
                "tokens_per_s": round((out.size(1) - len(ids)) / max(dt, 1e-6), 1),
                "iter": ckpt.get("iter", "?"),
            }, ensure_ascii=False).encode("utf-8")
            self._send(200, resp, "application/json")

        def log_message(self, fmt, *a):  # 静默访问日志
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[demo] 网页演示已启动：http://{args.host}:{args.port}  （Ctrl+C 停止）")
    server.serve_forever()


if __name__ == "__main__":
    main()
