### 05 工程落地：TensorRT-LLM / vLLM / bitsandbytes / AutoGPTQ

#### vLLM
- 支持AWQ/GPTQ部分格式与PagedAttention；
- 参数：`--quantization` 选项（随版本变化），配合`--gpu-memory-utilization`与批处理策略；
- 适合快速生产部署与OpenAI兼容API对接。

#### TensorRT-LLM
- 对INT8/FP8有成熟支持；
- 通过校准与构图（builder）生成引擎；
- 极致性能取决于内核与拓扑（NVLink/PCIe）。

#### bitsandbytes（bnb）
- 轻量化加载INT8/4权重（如NF4），适合在Python侧快速验证；
- 需注意内核版本与兼容性。

#### AutoGPTQ
- 方便离线生成GPTQ权重；
- 与Transformers生态良好集成；
- 需配合对应推理内核（vLLM/CTRansformers/自研）。


