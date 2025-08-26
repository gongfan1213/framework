# QLoRA（4/8bit）实践

## 一、核心思路
- 使用NF4/Int8对基础权重量化，仅训练LoRA低秩增量
- 兼顾显存占用与微调效果

## 二、环境与加载
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import __version__ as bnb_ver

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    device_map="auto",
)
```

## 三、训练注意事项
- 勿对量化权重做权重衰减；prefer AdamW with fused kernels
- 关注数值稳定：bf16 计算，grad_norm clip
- 较小 lr（5e-5↓），较长 warmup

## 四、评估与对比
- 与全参/LoRA 对比 ppl、任务指标
- 关注长文本稳性与数值溢出告警

## 五、常见坑
- CPU offload 导致吞吐骤降
- 量化与检查点版本不一致
- 推理需确保相同量化加载参数
