"""M3：GPT decoder-only transformer，从零手写（Llama 风格：RoPE + RMSNorm + SwiGLU）。

== 数据流（一个 forward 的全貌）==
  token id 序列 (B, T)
    → wte 查表变成向量          (B, T, 768)        "把词变成模型能算的数字"
    → 12 × TransformerBlock                            每一块做两件事：
        ① 自注意力：每个位置去"看"前文所有位置，汇总相关信息
        ② 前馈网络：对每个位置独立做一次非线性变换，存储/加工知识
    → 最终 RMSNorm
    → lm_head（与 wte 同一张表）(B, T, 32768)      对每个位置预测下一个 token 的概率
    → 与正确答案比对算交叉熵 loss

== 每个组件的原理见各类的 docstring。==
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm0.config import GPTConfig


class RMSNorm(nn.Module):
    """均方根归一化：比 LayerNorm 少减一个均值，参数更少、更稳定。

    把每个位置的向量除以它的均方根（压回标准尺度），再逐维乘一个可学习的缩放。
    为什么需要归一化：12 层残差累加后数值会越来越大，归一化让每层输入
    保持在稳定的分布里（"pre-norm"：归一化放在子层之前，是深层模型能训起来的关键）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # bf16 下求均方根精度不够，升到 fp32 算完再降回去
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0):
    """预计算 RoPE 用到的 cos/sin 表，形状 (seq_len, head_dim)。

    RoPE（旋转位置编码）的思路：把每个注意力头的向量看成 head_dim/2 个二维平面，
    第 t 个位置的向量在每个平面上旋转 t×频率 的角度。两个向量做点积时，
    旋转的相对角度只取决于它们的距离 —— 于是注意力天然"知道"词与词隔多远，
    而不需要额外学一张位置向量表。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)：第 t 行是各平面的旋转角
    emb = torch.cat([freqs, freqs], dim=-1)   # 复制成两半（另一半转相反角度）
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, n_head, T, head_dim) 的 q 或 k 做旋转。前半维转 +θ、后半维转 −θ 的等价实现。"""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """因果自注意力——transformer 的心脏。

    直觉：句子"我 吃 了 一个 苹果"里，"苹果"是什么意思取决于前文所有词。
    注意力让每个位置生成三个向量：
      q（query，我在找什么）k（key，我能提供什么标签）v（value，我的实际内容）
    每个 q 与所有前文位置的 k 算相似度（点积/√d），softmax 成权重，
    对前文的 v 加权求和 —— 这就是"看前文、取所需"。
    "因果"（causal）：只许看前文不许看后文（否则训练时就是作弊偷答案），
    实现为把注意力矩阵的上三角设成 -inf，softmax 后权重为 0。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.attn_impl = cfg.attn_impl
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        cos, sin = build_rope_cache(cfg.seq_len, self.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T <= self.cos.shape[0], f"序列长度 {T} 超过 RoPE 缓存 {self.cos.shape[0]}"

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B,T,C) → (B, 头数, T, 每头维度)：把向量切成 12 个头，各自独立做注意力
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.attn_impl == "naive":
            # 手写版：每一步都写清楚
            att = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B,H,T,T) 相似度
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~causal_mask, float("-inf"))        # 不许看未来
            att = F.softmax(att, dim=-1)                               # 权重归一化
            y = att @ v                                                # 对前文加权求和
        else:
            # sdpa 版：PyTorch 的融合算子，数学与上面逐行等价（测试对拍过），
            # 但用 FlashAttention 类内核实现——训练快 30%+ 且省显存。
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 12 个头拼回一个向量
        return self.proj(y)


class SwiGLU(nn.Module):
    """前馈网络（FFN）：注意力负责"沟通"，FFN 负责"思考"。

    普通 FFN 是两层矩阵夹一个激活；SwiGLU 用两条支路：
      一条过 silu 门控（决定放行多少），另一条原样通过，逐元素相乘后再投影回去。
    大模型的"知识"大部分就存在这三个矩阵的参数里。
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # 门控支路
        self.w3 = nn.Linear(dim, hidden, bias=False)  # 内容支路
        self.w2 = nn.Linear(hidden, dim, bias=False)  # 投影回原维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """一个 transformer 层 = 注意力 + FFN，各配 pre-norm 残差连接。

    残差（x = x + f(x)）让梯度可以直通底层，12 层堆叠才能训得动。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg.n_embd, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    """整装：embedding → N × Block → norm → lm_head（与 embedding 权重共享）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重共享（weight tying）：预测下一个词用的矩阵和查词向量的矩阵是同一张表。
        # 词表投影占 2300 万参数，共享后省掉一份，小模型收益巨大。
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # 残差分支的输出投影缩小初始化，避免 12 层累加后方差过大（GPT-2 的技巧）
        for name in ("proj", "w2"):
            if isinstance(m, CausalSelfAttention) and name == "proj":
                nn.init.normal_(m.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))
            if isinstance(m, SwiGLU) and name == "w2":
                nn.init.normal_(m.w2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # 共享权重只数一次
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        if targets is not None:  # 训练：所有位置一起算 loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        else:  # 推理：只需要最后一个位置的预测
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: tuple[float, float], device_type: str):
        """AdamW：矩阵参数（2D）做 weight decay，归一化缩放（1D）不做。"""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        extra = {"fused": True} if device_type == "cuda" and torch.cuda.is_available() else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    # ------------------------------------------------ 推理生成（M6 用）

    @torch.no_grad
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float | None = None) -> torch.Tensor:
        """自回归生成：预测下一个 token → 接到序列末尾 → 再预测，重复。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.seq_len else idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:  # 只在概率最高的 top_k 个里采样（截掉长尾乱选）
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:  # 核采样：只在累计概率达 top_p 的最短集合里采
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits)
            probs = F.softmax(logits.float(), dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
