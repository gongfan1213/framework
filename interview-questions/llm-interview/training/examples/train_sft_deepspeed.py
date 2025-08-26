import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


@dataclass
class SFTSample:
    instruction: str
    input: str
    output: str


def build_prompt(sample: SFTSample) -> str:
    sys = "你是资深助理"
    if sample.input:
        return f"<|system|>{sys}\n<|user|>{sample.instruction}\n{sample.input}\n<|assistant|>{sample.output}"
    else:
        return f"<|system|>{sys}\n<|user|>{sample.instruction}\n<|assistant|>{sample.output}"


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def preprocess(tokenizer, dataset: List[Dict], max_len=2048):
    texts = []
    for item in dataset:
        sample = SFTSample(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
            output=item.get("output", ""),
        )
        texts.append(build_prompt(sample))
    tokenized = tokenizer(texts, truncation=True, max_length=max_len)

    # 仅计算assistant部分损失：简单示例用全label，可按模板mask
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    model_name = os.environ.get("MODEL", "gpt2")
    data_path = os.environ.get("DATA", "samples.jsonl")
    output_dir = os.environ.get("OUT", "./sft-out")
    ds_config = os.environ.get("DS_CONFIG", "./ds_zero3.json")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset(data_path)
    tokenized = preprocess(tokenizer, dataset)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        deepspeed=ds_config,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
