PY=python
ROOT=interview-questions/llm-interview
DATA?=data/sft.jsonl
BASE?=gpt2
OUT?=runs
MERGED?=$(OUT)/merged

.PHONY: all clean sft lora qlora merge vllm

all: clean sft lora merge vllm

clean:
	@mkdir -p $(OUT)
	@echo "[OK] OUT dir: $(OUT)"

sft:
	MODEL=$(BASE) DATA=$(DATA) OUT=$(OUT)/sft $(PY) $(ROOT)/training/examples/train_sft_deepspeed.py

lora:
	MODEL=$(BASE) DATA=$(DATA) OUT=$(OUT)/lora $(PY) $(ROOT)/finetuning/examples/train_lora_peft.py

qlora:
	MODEL=$(BASE) DATA=$(DATA) OUT=$(OUT)/qlora $(PY) $(ROOT)/finetuning/examples/train_qlora_peft.py

merge:
	$(PY) $(ROOT)/finetuning/examples/export_merge_lora.py --base $(BASE) --adapter $(OUT)/lora --out $(MERGED)

vllm:
	MERGED_MODEL=$(MERGED) $(PY) $(ROOT)/inference/examples/vllm_min_infer.py
