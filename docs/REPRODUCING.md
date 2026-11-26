# 环境与复现指南

## 安装和测试

使用 Python 3.11，先安装 PyTorch，再安装 `requirements.txt`。整理时验证环境为 Python 3.11.9、PyTorch 2.6.0+cu124、NumPy 2.4.6、regex 2026.9.10、tqdm 4.70.1、pytest 9.1.1、pyarrow 25.0.1、opencc 1.4.2。这是整理时快照，原始训练完整依赖未锁定。

```bash
python -m pytest tests/
```

覆盖数据清洗、BPE 与模型正确性；GPU 可用时还运行单 batch 过拟合。整理时本机 **46 项全部通过**。

## 从数据到训练

1. 安装 aria2，在 Windows Git Bash 运行 `bash scripts/00_download_data.sh`。脚本含 `cygpath` / WinGet 路径回退；其他平台建议手动下载到约定位置。
2. 原始文件位于 `data/raw/wiki_zh/20231101.zh/*.parquet` 与 `data/raw/tinystories/TinyStoriesV2-GPT4-train.txt`。
3. 运行 `python scripts/01_prepare_data.py`。`--limit` 用于调试，会重写清洗输出。
4. 已提供配套 `data/tokenizer/tokenizer.json`。若学习重新训练词表，先备份它，再运行 `python scripts/02_train_tokenizer.py`。默认采样前部 150,000,000 字符；换词表后必须重新 token 化并从头训练匹配模型。
5. 运行 `python scripts/03_tokenize_corpus.py`；调试可加 `--workers 1`。输出 `data/tokenized/train.bin` / `val.bin`，格式为本机 uint16。
6. 运行 `python scripts/04_train.py --config 012b`。存在 `checkpoints/012b/ckpt.pt` 时自动恢复；`--fresh` 从头开始，后续会覆盖同目录产物，使用前先备份。

原始、清洗和 token 文件合计需十余 GB 磁盘空间，另需为 checkpoint 预留空间。日志的约 20.4k token/s 是本机特定环境表现。

## 关于 `--config test`

`test` 预设词表是 512，随仓库分词器为 32,768，不能直接拿真实 token 数据使用该预设。最小正确性验证先运行：

```bash
python -m pytest tests/test_model.py -v
```

如需早期训练循环冒烟，应在独立副本内使用 512 词表的合成数据，并使 tokenizer 校验和模型词表一致。合成数据低 loss 不是语言建模成绩。

## 推理

按 [模型卡](MODEL_CARD.md) 获取权重，放入 `checkpoints/012b/`，保留配套 tokenizer：

```bash
python scripts/05_generate.py --config 012b --prompt "Once upon a time" --max-tokens 120
python scripts/05_generate.py --config 012b --interactive
python scripts/06_demo_web.py --port 8000
```

交互模式每轮独立续写，无多轮记忆。网页默认绑定本机回环地址，用标准库提供 HTTP 界面，模型推理仍依赖 PyTorch。

## 复现实验的边界

本仓库保留历史训练实现。梯度累积缩放、日志字段和随机采样问题详见 [训练报告](TRAINING_REPORT.md)。修正代码或改变词表、数据、超参时应另建实验记录，不把新曲线混入本次结果。
