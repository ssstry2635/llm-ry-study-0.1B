# 实验记录的来源

- `metrics.original.csv`：从历史 `logs/metrics.csv` 复制，保留试跑和重复评估。
- `metrics.training.csv`：由 `scripts/07_plot_training.py` 提取真实语料的正式训练结果；每步保留最后一条。
- `train_012b.log`：历史训练主日志，包含早期合成试跑。
- `tokenizer_train.log`、`tokenize_corpus.log`：历史分词记录，已将机器绝对路径替换为相对路径；多进程 token 化中的「0 字符」是未累计字符计数，不代表空语料。
- `data_stats.json`：清洗输出的统计文件；token 字段为估算。
- `generation_samples.json`：整理时使用历史最佳权重、固定 seed 生成的三个未编辑样例。
- `checkpoint_manifest.json`：本地权重和配套 tokenizer 的字节大小、SHA-256 与权重元数据。

历史 CSV 的 `tok_per_s` / `elapsed_s` 有字段语义错误，勿直接当作吞吐 / 训练耗时。细节见 [训练报告](../docs/TRAINING_REPORT.md)。
