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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


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
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    model_name = os.environ.get("MODEL", "gpt2")
    data_path = os.environ.get("DATA", "samples.jsonl")
    output_dir = os.environ.get("OUT", "./qlora-out")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4bit QLoRA 加载
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )

    # LoRA 头
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    dataset = load_dataset(data_path)
    tokenized = preprocess(tokenizer, dataset)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=tokenized,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
