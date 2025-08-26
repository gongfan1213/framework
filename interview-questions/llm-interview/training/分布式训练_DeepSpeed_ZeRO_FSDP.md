# 分布式训练实践：DeepSpeed / ZeRO / FSDP

## 一、并行策略总览
- 数据并行（DP）、张量并行（TP）、流水并行（PP）、混合并行（3D并行）
- 选择依据：模型大小、GPU内存、集群拓扑、吞吐/延迟目标

## 二、ZeRO 优化阶段
- Stage-1：优化器状态分片
- Stage-2：再分片梯度
- Stage-3：再分片权重（可offload到CPU/NVMe）
- 面试点：为何 ZeRO-3 能显著降低峰值显存？答：参数/梯度/优化器三者全部分片

## 三、DeepSpeed 配置示例（ZeRO-3 + Offload）
```json
{
  "train_batch_size": 128,
  "gradient_accumulation_steps": 8,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_param": {"device": "cpu", "pin_memory": true},
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "optimizer": {"type": "AdamW", "params": {"lr": 2e-5, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}},
  "scheduler": {"type": "CosineAnnealing", "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-5, "warmup_num_steps": 1000}}
}
```

## 四、FSDP 实践要点（PyTorch）
```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

policy = transformer_auto_wrap_policy

model = build_model()
model = FSDP(model, auto_wrap_policy=policy, mixed_precision=torch.bfloat16)
```
- Auto wrap 仅包裹大层（如注意力/MLP块），减少通信
- 参数扁平化、逐层重建；配合 activation checkpointing

## 五、Checkpoint 与容错
- 保存策略：每N步局部+全局；异步保存
- 容错：启用 `torchrun --max-restarts`；DeepSpeed 异常重试

## 六、性能诊断
- 关注：通信等待（NCCL timeline）、显存碎片、梯度同步热点
- 工具：nsys、torch.profiler、deepspeed --dump_state

## 七、面试要点
- 解释 ZeRO-1/2/3 的差异与适用场景
- FSDP 与 ZeRO-3 的取舍（生态、易用性、offload支持）
- 如何在大模型上实现 3D 并行（TP+PP+DP）
