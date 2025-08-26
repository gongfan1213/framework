# 量化实战：GPTQ / AWQ / SmoothQuant

## 一、动机
- 降低显存/存储，提升推理吞吐；在可接受的精度损失下运行大模型

## 二、方法对比
- GPTQ：离线权重量化，最小化量化误差（Hessian近似）；适合离线压缩
- AWQ：Activation-aware Weight Quantization，关注激活分布，鲁棒性更好
- SmoothQuant：将激活峰值平滑到权重，改善激活量化效果

## 三、GPTQ 实践
```python
# 伪代码：使用 AutoGPTQ 量化
from autogptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config={
  "bits": 4,
  "group_size": 128,
  "desc_act": True
})
model.quantize(dataset="calib.jsonl")
model.save_quantized("./gptq-4bit")
```
- 关键：校准数据代表性；选择 group_size 与 per-channel 选项

## 四、AWQ 实践
```python
# 伪代码：使用 AWQ 量化
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(wbits=4, groupsize=128, calib_data="calib.jsonl")
model.save_quantized("./awq-4bit")
```
- 关注激活 outlier 的处理；较优的鲁棒性

## 五、SmoothQuant 流程
1) 统计激活峰值，计算平滑系数 s
2) 将部分幅度迁移到权重：`W' = W * diag(s)`，`A' = A / s`
3) 再进行激活/权重量化

## 六、评估指标
- Perplexity（WikiText2/PP）
- 下游任务指标与人评
- 延迟/吞吐/GPU利用率、模型体积

## 七、部署注意
- 内核支持：cutlass/flash-attn、量化matmul kernel
- 编译与运行时（TensorRT-LLM、vLLM、LMDeploy）
- 与KV缓存/分页注意力的兼容

## 八、面试要点
- 比较 GPTQ 与 AWQ 的差异与适用场景
- 解释 SmoothQuant 如何改善激活量化
- 如何选择校准集与量化超参
