import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(base_model, adapter_path, out_dir, dtype="bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=getattr(__import__('torch'), dtype))

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    peft_model = peft_model.merge_and_unload()

    peft_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    merge_and_save(args.base, args.adapter, args.out, args.dtype)
