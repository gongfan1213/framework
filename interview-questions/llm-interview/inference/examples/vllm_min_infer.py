import os
from vllm import LLM, SamplingParams

# 假设已用 export_merge_lora.py 将 LoRA/QLoRA 合并到 base 权重
# base_dir 指向合并后的权重目录
base_dir = os.environ.get("MERGED_MODEL", "./merged")

llm = LLM(model=base_dir, tensor_parallel_size=int(os.environ.get("TP", 1)), trust_remote_code=True)

prompt = "请用三句话解释什么是注意力机制。"
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
outputs = llm.generate([prompt], params)
print(outputs[0].outputs[0].text)
