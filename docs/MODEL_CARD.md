# 模型卡：RY Study 0.1B

## 概况

- 作者：[ssstry2635](https://github.com/ssstry2635)。
- 从随机初始化预训练的 decoder-only 自回归模型。
- 参数 110,119,680；12 层 / 768 维 / 12 头；1,024 上下文。
- 自实现 byte-level BPE，32,768 词表，随仓库提供。
- 数据为中文维基百科与 TinyStories，见 [数据说明](DATA.md)。
- 最佳权重 step 14,750，随机采样 val loss 1.7797412872314453；训练完成 15,000 步。

## 权重与完整性

权重已作为 [v0.1.0 Release](https://github.com/ssstry2635/llm-ry-study-0.1B/releases/tag/v0.1.0) 附件公开发布，不写入 Git 历史。直接下载：[best_model.pt](https://github.com/ssstry2635/llm-ry-study-0.1B/releases/download/v0.1.0/best_model.pt)。

`best_model.pt` 为 **440,509,524 bytes**（约 420.1 MiB）。配置、指标和 SHA-256 见 [checkpoint_manifest.json](../reports/checkpoint_manifest.json)。放到 `checkpoints/012b/best_model.pt`，使用本仓库配套 tokenizer。

1.32GB 的 `ckpt.pt` 包括优化器状态，用于续训，推理不需要。检查点使用 `torch.load(..., weights_only=True)` 加载。

## 用途与边界

适合学习分词、因果语言建模、单 GPU 训练和采样。模型没有指令微调或偏好对齐；问答、推理和事实检索能力未经验证。输出可能不通顺、重复或编造，中英文表现可能不同。

只有混合语料随机窗口验证指标，无标准 benchmark、固定全量测试集、多随机种子或独立语言评估。不同词表和数据集的 loss 不能直接比较。

提供代码、配套词表、配置、日志与哈希，原始 RNG 状态未保存。已知问题见 [训练报告](TRAINING_REPORT.md)，运行步骤见 [复现指南](REPRODUCING.md)。
