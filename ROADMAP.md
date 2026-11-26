# 当前进度（2026-11-26 整理）

已完成：数据清洗、自训 32k BPE、110.12M GPT、token 管线、15,000 步预训练、命令行与网页续写。现有 46 项测试通过。

- [学习过程](docs/LEARNING_JOURNEY.md)
- [真实训练结果与已知问题](docs/TRAINING_REPORT.md)
- [运行与复现](docs/REPRODUCING.md)

下一阶段：规范梯度累积与日志字段、固定评估协议、分语言评估、KV cache 和小规模 SFT。

---

## 以下是启动时的历史计划

历史勾选状态、性能估算、部分脚本编号与验收目标不代表最终完成结果。实际运行请使用上面的最新文档。

# 从零手写 Tokenizer + Transformer，端到端训练 0.1B 大模型

> 目标：不用 HuggingFace transformers / tokenizers / sentencepiece，
> 只用 PyTorch 的张量运算，亲手写出一个 BPE tokenizer 和一个 GPT 架构，
> 在本机 RTX 4060 Laptop (8GB) 上完成 0.1B 模型的预训练、推理生成。

---

## 硬件与算力预算（本机实测环境）

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8GB 显存 |
| 驱动 | CUDA 13.0（任何 ≤13 的 PyTorch CUDA 版本都能跑） |
| 环境 | 项目内 venv `.venv`（Python 3.11.9），PyTorch 2.x + cu124 |
| 训练方式 | bf16 混合精度 + 梯度累积 +（如需要）梯度检查点 |

**算力账**（心里要有数，这是所有决策的依据）：
- 训练吞吐估算：6 × 参数量 × token数 FLOPs；4060 Laptop **实测 bf16 矩阵乘 28.6 TFLOPS**（2026-09-17，`verify_env.py`）
- 按训练效率 ~55% 折算，0.1B 模型 ≈ 每秒可训 **~2.4万 token**
- **训 10 亿 token ≈ 12 小时挂机**（第一次跑建议先 3~5 亿 token，约 4~6 小时）
- 显存账：权重 0.2GB + AdamW 状态 0.8GB + 梯度 0.2GB ≈ 1.2GB，剩余 ~6GB 全给激活值

## 目标模型配置（≈110M 参数，"0.1B"）

| 超参 | 值 | 说明 |
|---|---|---|
| vocab_size | 32768 | 自己训的 byte-level BPE，中文友好 |
| n_layer | 12 | Transformer block 层数 |
| n_embd | 768 | 隐藏维度 |
| n_head | 12 | 注意力头数（head_dim = 64） |
| 中间层 | 2048 (SwiGLU) | FFN 维度 |
| 位置编码 | RoPE | 旋转位置编码，不用可学习位置向量 |
| 归一化 | RMSNorm (pre-norm) | 比 LayerNorm 参数更少更稳 |
| seq_len | 1024 | 训练上下文长度 |
| lm_head | 与 embedding 共享权重 (tied) | 省 2500 万参数 |

参数量验算：embedding 768×32768 ≈ 25.2M + 12 × (attention 2.36M + SwiGLU 4.72M) ≈ 85M + norm ≈ **110M** ✓

---

## 里程碑总览

每个里程碑都有**明确的验收标准**，达标才进入下一个。

### M0 环境与骨架 ✅
- [x] 项目 venv `.venv`（Python 3.11.9，winget 安装；`D:\anaconda` 是 2021 年的老 3.8，不用）
- [x] PyTorch + CUDA 可用，`scripts/verify_env.py` 全绿
- [x] 项目目录骨架

### M1 数据准备 🔄（进行中）
- 语料：**中文维基** `wikimedia/wikipedia` 20231101.zh（1.6GB parquet，**1,384,748 篇**）+ **TinyStories V2**（2.1GB，GPT-4 生成的英文儿童故事，中位 725 字符）；预计中文占 token 的 75~80%
- 抽样发现（`scripts/inspect_raw.py`）：中文维基 **12.3% 的字符是繁体**（95.8% 文档含繁体）→ 必须 OpenCC 繁转简；
  3.1% 文档 <100 字（"XX可以指："消歧页）；5.3% 文档汉字比例 <0.5（表格/列表）；中位长度 1405 字
- 下载踩坑记录：HF 的 CDN 走代理时**按单条连接限速**（0.5MB/s），`huggingface_hub` 和单线程 curl 都慢；
  换 aria2 一个文件开 8 条连接后 ~9MB/s。另外 bash 的 `set -o pipefail` 会让 `curl -sI | grep` 无匹配时静默退出脚本，要 `|| true`
- 脚本链：`00_download_data.sh`（aria2，可续传）→ `01_prepare_data.py`（清洗）→ `data/clean/{train,val}.jsonl` + `stats.json`
- **作业**：`llm0/data_prep.py` 里 4 个 TODO 函数（normalize_whitespace / is_good_zh_doc / is_good_en_doc / dedupe）
- **验收**：`pytest tests/test_data_prep.py` 全绿；`01_prepare_data.py` 全量跑通并输出 token 预算；抽样肉眼检查 clean 数据

### M2 手写 BPE Tokenizer（第一个核心交付物）
纯 Python 实现 byte-level BPE（GPT-2 同款方案）：
- GPT-2 的正则预分词（中英文都能正确切）
- BPE 合并规则训练算法（增量合并最高频字节对，词表 32k）
- `encode / decode`，round-trip 无损
- 特殊 token：`<|endoftext|>` 等
- **验收**：
  - `tests/test_bpe.py` 通过（round-trip、特殊 token、空串/emoji/生僻字）
  - 中文压缩率：每汉字 ≤1.5 token
  - 100MB 语料训完词表 < 30 分钟（纯 Python 可以慢，但要能跑完）

### M3 手写 Transformer
在 `llm0/model.py` 中逐模块实现并单测：
1. RMSNorm → 2. RoPE → 3. Causal Self-Attention（先 MHA，手写 softmax(QK^T/√d)V）
   → 4. SwiGLU FFN → 5. Block（pre-norm 残差）→ 6. GPT 整体（tied embedding）
- **验收**：
  - 参数量与上表一致（写参数量断言测试）
  - 随机初始化时 loss ≈ ln(32768) ≈ 10.4
  - **过拟合单 batch 测试**：50 步内把一个 batch 的 loss 打到 <0.1（验证整个反向传播链路正确，最重要的一步）
  - 用 `torch.autograd.gradcheck` 抽查自定义算子（RoPE）

### M4 数据管线
- 把语料全部 tokenize 成 token 流，存成 uint16 的 `memmap` 二进制（`data/train.bin` / `val.bin`）
- Dataset：从 memmap 随机窗口切片，(x, y) 错一位即下一个 token 预测
- **验收**：DataLoader 跑通；验证 1 亿字符 ≈ N 亿 token 的换算；dataloader 不成为训练瓶颈（GPU 利用率抽查 >80%）

### M5 训练循环（第二个核心交付物）
`scripts/03_train.py`，nanoGPT 风格：
- AdamW（β=0.9/0.95, wd=0.1 只作用于二维参数）、梯度裁剪 1.0
- warmup + cosine 退火，峰值 lr ≈ 6e-4
- bf16 混合精度 + 梯度累积（micro-batch 8~16 × accum 8 → 有效 batch ≈ 10 万 token/步）
- 定期 val loss 评估、checkpoint 保存/断点续训、日志
- **验收**：
  - 先用 10M 迷你模型跑通全流程 15 分钟，loss 稳定下降
  - 再启动 0.1B 正式训练，val loss 降到 **≤3.5**（中文语料 32k 词表下的合理水平）
  - 训练曲线无明显发散/loss 尖刺

### M6 推理与生成
`scripts/04_generate.py`：
- 自回归采样：temperature / top-k / top-p，rep penalty
- 流式打印生成效果；封装一个简易对话（`<|user|>` 模板拼接）
- **验收**：模型能生成通顺中文段落（0.1B + 10亿 token 预期是"通顺但没常识"，这是正常的）

### M7 进阶（可选，按兴趣选做）
- KV cache（推理提速）
- FlashAttention（`F.scaled_dot_product_attention` 对拍自己手写的 attention）
- GQA（省显存）
- SFT：用指令数据把预训练模型微调成能对话
- torch.compile 加速训练

---

## 目录结构

```
从0开始学大模型/
├── ROADMAP.md          # 本文件：总路线图
├── llm0/               # 主代码包
│   ├── bpe.py          # M2: BPE tokenizer
│   ├── model.py        # M3: GPT transformer
│   ├── dataset.py      # M4: memmap 数据管线
│   ├── config.py       # 模型/训练超参
│   └── utils.py
├── scripts/
│   ├── verify_env.py   # M0: 环境自检
│   ├── 01_train_tokenizer.py  # M2: 训词表
│   ├── 02_tokenize_corpus.py  # M4: 语料转 token 流
│   ├── 03_train.py     # M5: 训练主脚本
│   └── 04_generate.py  # M6: 推理生成
├── tests/              # 每个模块的单元测试
├── data/               # 语料与 token 化数据（体积大，不进 git）
├── checkpoints/        # 模型存档
└── logs/               # 训练日志
```

## 学习原则（很重要）

1. **教学模式：手把手带写**（2026-09-18 起）。每个函数：我讲"要解决什么问题 → 怎么想 → 代码 → 逐行解释"，
   你跑测试、改参数做小实验、用自己的话复述。能复述才算懂，能改对才算会。
2. **每一步都有验收标准**，测试不绿不往下走。深度学习最大的坑是"看起来能跑"。
3. 遇到不懂的概念（RoPE、SwiGLU、为什么 pre-norm……）随时问，我按"直觉 → 公式 → 代码"三步讲。
4. 训练是挂机活，写代码和调试才是学习主体——把单测做厚。

## 当前进度

- **M0~M6 全部完成**（2026-09-22 训练收官）：
  - 0.1B 模型（110,119,680 参数）完成 15000 步训练，消化 **9.8 亿 token**，实际耗时约 21 小时（含两晚中断与休息间隔）
  - **best val loss 1.78**（验收线 ≤3.5，初始基线 10.40）——平均每步把下一个 token 的候选从 32768 压缩到约 6 个
  - 生成验收：英文能写完整连贯的小故事；中文能成段成篇（维基文体），事实靠编、有重复循环——0.1B 的正常水平
- 待选做：**M7 进阶**（KV cache / SFT 指令微调 / 更多中文语料重混 / torch.compile）
