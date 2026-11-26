#!/usr/bin/env bash
# M1 第一步：下载原始语料到 data/raw/
#   可重复运行：已完整的文件跳过，下到一半的断点续传，网络抖动自动重试。
#
# 为什么用 aria2 多连接下载：本机实测 HF 的 CDN 走代理时**按单条连接限速**（一条 0.5MB/s），
# 但多开连接各自都有带宽 —— aria2 一个文件开 8 条连接分段下，聚合速度能到几 MB/s。
# 安装：winget install aria2.aria2（已装好；装完需要重开终端才能直接敲 aria2c）
#
# 用法（Git Bash）：
#     bash scripts/00_download_data.sh
# 语料：
#   1. 中文维基百科  wikimedia/wikipedia 的 20231101.zh 子集（6 个 parquet，约 1.6GB）
#   2. 英文 TinyStories V2（GPT-4 生成的儿童小故事，2.1GB；清洗时只取其中一部分）
set -euo pipefail
cd "$(dirname "$0")/.."

find_aria2() {
  # 1) 已在 PATH 里；2) winget 的别名目录；3) winget 包目录（别名有时没建成功）
  command -v aria2c 2>/dev/null && return
  local base
  base=$(cygpath "$LOCALAPPDATA")/Microsoft/WinGet
  [[ -x "$base/Links/aria2c.exe" ]] && { echo "$base/Links/aria2c.exe"; return; }
  find "$base/Packages" -name aria2c.exe 2>/dev/null | head -1
}
ARIA2=$(find_aria2)
[[ -n "$ARIA2" ]] || { echo "找不到 aria2c，请先：winget install aria2.aria2"; exit 1; }
echo "使用 aria2c：$ARIA2"
BASE="${HF_ENDPOINT:-https://huggingface.co}/datasets"

# 每行："仓库内路径  本地路径"
FILES=(
  "wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet  data/raw/wiki_zh/20231101.zh/train-00000-of-00006.parquet"
  "wikimedia/wikipedia/resolve/main/20231101.zh/train-00001-of-00006.parquet  data/raw/wiki_zh/20231101.zh/train-00001-of-00006.parquet"
  "wikimedia/wikipedia/resolve/main/20231101.zh/train-00002-of-00006.parquet  data/raw/wiki_zh/20231101.zh/train-00002-of-00006.parquet"
  "wikimedia/wikipedia/resolve/main/20231101.zh/train-00003-of-00006.parquet  data/raw/wiki_zh/20231101.zh/train-00003-of-00006.parquet"
  "wikimedia/wikipedia/resolve/main/20231101.zh/train-00004-of-00006.parquet  data/raw/wiki_zh/20231101.zh/train-00004-of-00006.parquet"
  "wikimedia/wikipedia/resolve/main/20231101.zh/train-00005-of-00006.parquet  data/raw/wiki_zh/20231101.zh/train-00005-of-00006.parquet"
  "roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt  data/raw/tinystories/TinyStoriesV2-GPT4-train.txt"
)

for entry in "${FILES[@]}"; do
  read -r remote local <<<"$entry"
  url="$BASE/$remote"
  mkdir -p "$(dirname "$local")"

  # 跟随重定向到 CDN 后的 Content-Length 就是文件真实大小。
  # HEAD 偶尔会失败/拿不到该头，此时 expected 为空 → 不跳过，交给 aria2 -c 续传判断。
  # 注意 set -o pipefail 下 grep 无匹配会让整个赋值失败并退出脚本，所以要 || true。
  expected=$(curl -sIL --max-time 20 "$url" | grep -i '^content-length:' | tail -1 | tr -dc '0-9' || true)
  have=$(stat -c %s "$local" 2>/dev/null || echo 0)
  if [[ -n "$expected" && "$have" == "$expected" ]]; then
    echo "[跳过] $local  已完整 ($((have / 1048576)) MB)"
    continue
  fi

  echo "[下载] $local  已有 $((have / 1048576)) MB / 共 $((${expected:-0} / 1048576)) MB"
  "$ARIA2" -c -x 8 -s 8 -k 8M \
    --max-tries=0 --retry-wait=10 --timeout=60 \
    --file-allocation=none --auto-file-renaming=false \
    --console-log-level=warn --summary-interval=30 \
    -d "$(dirname "$local")" -o "$(basename "$local")" "$url"
  echo "[完成] $local  ($(($(stat -c %s "$local") / 1048576)) MB)"
done

echo "下载完成 ✔  下一步：python scripts/01_prepare_data.py"
