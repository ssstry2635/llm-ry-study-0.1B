"""模型与训练超参配置。

0.1B 配置的参数量验算（无任何 bias）：
  embedding:   768 × 32768            = 25,165,824   (lm_head 与其共享权重)
  每层 attention: qkv 3×768² + proj 768² = 2,359,296
  每层 ffn:      3 × 768 × 2048       = 4,718,592   (SwiGLU 三个矩阵)
  每层 norm:     2 × 768              =     1,536
  → 12 层 × 7,079,424 = 84,953,088；加 embedding 和最终 norm
  → 合计 110,119,680 ≈ 0.11B（测试里有逐位断言）
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    seq_len: int = 1024          # 训练上下文长度（RoPE 缓存按此预计算）
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12             # head_dim = n_embd / n_head = 64
    ffn_hidden: int = 2048       # SwiGLU 中间维度（≈ 8/3 × n_embd 再取整到 64 倍数）
    dropout: float = 0.0
    rope_theta: float = 10000.0
    attn_impl: str = "sdpa"      # "naive"=手写 softmax(QKᵀ/√d)V（教学/对拍）|"sdpa"=等价的融合算子（训练用，快很多）


def gpt_0_1b(vocab_size: int = 32768) -> GPTConfig:
    """目标模型：≈110M 参数。"""
    return GPTConfig(vocab_size=vocab_size, seq_len=1024, n_layer=12,
                     n_embd=768, n_head=12, ffn_hidden=2048, attn_impl="sdpa")


def gpt_test(vocab_size: int = 512) -> GPTConfig:
    """测试用迷你模型：几秒就能跑完前向/反向。"""
    return GPTConfig(vocab_size=vocab_size, seq_len=256, n_layer=4,
                     n_embd=128, n_head=4, ffn_hidden=256, attn_impl="naive")
