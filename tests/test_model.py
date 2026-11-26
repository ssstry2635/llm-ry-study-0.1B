"""M3 模型验收测试。运行：python -m pytest tests/test_model.py -v"""
import math

import pytest
import torch

from llm0.config import gpt_0_1b, gpt_test
from llm0.model import GPT, apply_rope, build_rope_cache

VOCAB = 512


def make_model(cfg=None, device="cpu") -> GPT:
    torch.manual_seed(42)
    model = GPT(cfg or gpt_test(VOCAB)).to(device)
    return model


# ---------------------------------------------------------------- 参数量


def test_param_count_of_012b():
    model = GPT(gpt_0_1b(vocab_size=32768))
    # 见 llm0/config.py 的逐项验算：embedding 25,165,824 + 12层×7,079,424 + 最终norm 768
    assert model.num_params() == 110_119_680
    assert model.num_params(non_embedding=True) == 84_953_856
    # 权重共享：lm_head 和 embedding 必须是同一个张量
    assert model.lm_head.weight is model.wte.weight


# ---------------------------------------------------------------- 前向


def test_init_loss_is_ln_vocab():
    # 随机初始化的模型什么都不知道 → 每个位置均匀预测 32768…512 个词 → loss ≈ ln(512) ≈ 6.24
    model = make_model()
    x = torch.randint(0, VOCAB, (4, 128))
    y = torch.randint(0, VOCAB, (4, 128))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(VOCAB)) < 0.5


def test_output_shape():
    model = make_model()
    x = torch.randint(0, VOCAB, (2, 64))
    logits, loss = model(x, torch.randint(0, VOCAB, (2, 64)))
    assert logits.shape == (2, 64, VOCAB)
    logits_gen, loss_gen = model(x)  # 推理路径只输出最后一个位置
    assert logits_gen.shape == (2, 1, VOCAB)
    assert loss_gen is None


def test_causal_no_peeking_at_future():
    # 把序列后半段换掉，前半段的预测必须纹丝不动（否则就是"偷看答案"）
    model = make_model()
    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        a = torch.randint(0, VOCAB, (1, 32))
        b = a.clone()
        b[0, 20:] = torch.randint(0, VOCAB, (12,))
        # 传 targets 走训练路径，拿全部位置的 logits（不传只返回最后一个位置）
        dummy = torch.zeros_like(a)
        la, _ = model(a, dummy)
        lb, _ = model(b, dummy)
    assert torch.allclose(la[0, :20], lb[0, :20], atol=1e-5)


def test_naive_attention_matches_sdpa():
    # 手写的 softmax(QKᵀ/√d)V 必须和融合算子数值一致（训练用 sdpa 才放心）
    torch.manual_seed(7)
    from llm0.config import GPTConfig
    from llm0.model import CausalSelfAttention

    cfg_naive = GPTConfig(vocab_size=VOCAB, seq_len=64, n_embd=64, n_head=4,
                          ffn_hidden=128, attn_impl="naive")
    cfg_sdpa = GPTConfig(vocab_size=VOCAB, seq_len=64, n_embd=64, n_head=4,
                         ffn_hidden=128, attn_impl="sdpa")
    m1, m2 = CausalSelfAttention(cfg_naive), CausalSelfAttention(cfg_sdpa)
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        y1, y2 = m1(x), m2(x)
    assert torch.allclose(y1, y2, atol=1e-4)


# ---------------------------------------------------------------- RoPE


def test_rope_position_zero_is_identity():
    # 位置 0 的旋转角是 0 → 不旋转，输出应等于输入
    cos, sin = build_rope_cache(8, 16)
    x = torch.randn(2, 3, 1, 16)  # 只有位置 0
    assert torch.allclose(apply_rope(x, cos[:1], sin[:1]), x, atol=1e-6)


def test_rope_relative_position_property():
    # RoPE 的核心性质：注意力分数只依赖相对距离（q 在 t 位、k 在 s 位，
    # 换成 t+d、s+d 分数不变）。构造恒定 q=k 的向量验证。
    head_dim = 16
    cos, sin = build_rope_cache(32, head_dim)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    # 手动把 q 放在位置 2、k 放在位置 5（相对距离 3）
    qr = apply_rope(q, cos[2:3], sin[2:3])
    kr = apply_rope(k, cos[5:6], sin[5:6])
    score_a = (qr * kr).sum()
    # 整体平移到位置 10、13（相对距离仍 3）→ 分数应相同
    qr2 = apply_rope(q, cos[10:11], sin[10:11])
    kr2 = apply_rope(k, cos[13:14], sin[13:14])
    score_b = (qr2 * kr2).sum()
    assert torch.allclose(score_a, score_b, atol=1e-4)


# ---------------------------------------------------------------- 反向传播


def test_overfit_single_batch():
    """最重要的正确性测试：单个 batch 反复训练，loss 必须能压到接近 0。
    压不下去说明前向/反向链路有 bug——这一关不过，训 12 小时也白训。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = make_model(gpt_test(VOCAB), device=device)
    opt = model.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cuda")
    x = torch.randint(0, VOCAB, (8, 128), device=device)
    y = torch.randint(0, VOCAB, (8, 128), device=device)
    first = None
    for _ in range(60):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert first > 5.0           # 起点接近 ln(512)
    assert loss.item() < 0.5     # 60 步内背下这个 batch
