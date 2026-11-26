"""M4：memmap 数据管线——把 token 流当数组用，内存占用与文件大小无关。

data/tokenized/{train,val}.bin 是一维 uint16 数组（vocab 32768 < 65536，两个字节够存），
文档之间用 <|endoftext|> 的 token id 分隔。训练时随机取起始位置、切 seq_len+1 长度的窗口：
  x = tokens[i : i+seq_len]      模型的输入
  y = tokens[i+1 : i+seq_len+1]  正确答案（就是输入整体右移一位）
"预测每个位置的下一个 token"——这就是语言模型的全部训练目标。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def open_bin(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: torch.device):
    """随机位置采样一个 batch（nanoGPT 风格）。比按顺序切窗口的 epochs 更多样。"""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, *, batch_size, seq_len, device, iters=40):
    """周期性评估：train/val 各随机采 iters 个 batch 取平均 loss。"""
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, batch_size, seq_len, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out
