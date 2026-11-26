# 数据来源与处理

| 数据 | 版本 | 清洗后文档数 | 原始说明 |
|---|---|---:|---|
| 中文维基百科 | `wikimedia/wikipedia`，`20231101.zh` | 990,107 | [官方数据卡](https://huggingface.co/datasets/wikimedia/wikipedia)；标注 CC BY-SA 3.0 / GFDL |
| TinyStories | `TinyStoriesV2-GPT4-train.txt` | 2,717,409 | [作者数据卡](https://huggingface.co/datasets/roneneldan/TinyStories)；标注 CDLA Sharing 1.0 |

数据的署名、许可与再分发条件以各来源为准。本仓库保存下载脚本和统计，不镜像原始或清洗全文。

处理顺序：NFC / 空白规整 → 中文繁转简 → 长度与语言比例过滤 → 文档精确去重 → 内容哈希分桶（约 1% 验证集）。

[data_stats.json](../reports/data_stats.json) 中 `est_tokens` 按字符比例估算，中文 65.9% 也是估计，不能当作实际 token 比例。

实际 token 化日志记录训练集 **1,421,745,504 tokens**、验证集 **14,351,504 tokens**。文档末尾插入 `<|endoftext|>`，保存为 uint16；训练时从连续文件随机截取窗口。

清洗文件先写中文再写英文；词表脚本顺序取前部采样，模型训练随机取窗口。目前没有两种语言各自实际 token 数统计。精确去重也不排除文档间长片段重叠。
