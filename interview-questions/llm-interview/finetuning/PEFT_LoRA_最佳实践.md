# PEFT / LoRA 最佳实践

## 一、为何选择PEFT
- 显著降低可训练参数与显存开销
- 迁移快、易部署，仅加载Adapter权重

## 二、LoRA 关键超参
- rank(r): 4/8/16（任务复杂度越高可适当增大）
- alpha: 一般 16~64，控制缩放
- target modules: attn q/k/v/o、mlp proj 等
- dropout: 0~0.1，防过拟合

## 三、实战模板（Transformers + PEFT）
```python
from peft import LoraConfig, get_peft_model

def build_lora(model, r=8, alpha=32, dropout=0.05, target=["q_proj","v_proj"]):
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target, task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)
```

## 四、冻结策略与梯度检查点
- 冻结除 LoRA 层外的其余参数；开启 gradient_checkpointing 以省显存

## 五、混合策略
- LoRA + Adapters（例如 Prefix/IA3）
- LoRA + QLoRA（配合 4bit 量化）

## 六、面试问答
- Q: 为何只在注意力层加LoRA？
  - A: 性价比最高，必要时对MLP扩展以覆盖语义变形
- Q: LoRA偏置项如何处理？
  - A: 通常不训练bias；需要时可启用lora_bias=all
- Q: 如何做多域适配？
  - A: 多Adapter并行/路由加载；或adapter fusion 做加权融合
