#!/usr/bin/env bash
set -euo pipefail

ROOT=interview-questions/llm-interview
DATA_RAW=${1:-data/sft.jsonl}
DATA_CLEAN=./cleaned.json
BASE=${2:-gpt2}
OUT=./runs

echo "[1/5] 数据清洗"
python $ROOT/data/scripts/advanced_cleaning.py $DATA_RAW > $DATA_CLEAN || true

echo "[2/5] SFT 训练（DeepSpeed）"
MODEL=$BASE DATA=$DATA_CLEAN OUT=$OUT/sft python $ROOT/training/examples/train_sft_deepspeed.py

echo "[3/5] LoRA 训练（8bit）"
MODEL=$BASE DATA=$DATA_CLEAN OUT=$OUT/lora python $ROOT/finetuning/examples/train_lora_peft.py

echo "[4/5] 合并导出"
python $ROOT/finetuning/examples/export_merge_lora.py --base $BASE --adapter $OUT/lora --out $OUT/merged

echo "[5/5] vLLM 推理"
MERGED_MODEL=$OUT/merged python $ROOT/inference/examples/vllm_min_infer.py

echo "Done. 输出目录: $OUT"
