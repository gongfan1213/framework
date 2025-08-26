### 05 量化与加速引擎：TensorRT-LLM、vLLM、FasterTransformer

本篇对主流推理加速方案进行对比与实践要点提炼。

#### 量化基础
- 目的：降低显存/带宽占用、提高吞吐，代价是精度与质量可能下降。
- 路线：
  - 训练后量化（PTQ）：GPTQ、AWQ、RPTQ；简单易用，效果受数据分布影响。
  - 量化感知训练（QAT）：需要再训练，质量更好，成本高。
  - 数据类型：INT8/INT4/NF4/FP8；权重/激活可分别选择策略。

#### TensorRT-LLM（NVIDIA）
- 优势：与NVIDIA硬件生态深度结合，内核高度优化，支持Speculative、PagedKV、FP8等。
- 要点：构图与引擎编译、Profile校准、分块与流水、插件内核、KV分页、并行策略。
- 适合：NVIDIA GPU为主的生产集群，追求极致吞吐与低延迟。

#### vLLM
- 优势：PagedAttention带来的高效KV管理；OpenAI兼容接口；生态活跃。
- 要点：动态批处理、连续批处理、前缀缓存与并行；部署简便，扩展灵活。
- 适合：通用生产环境，快速获得高吞吐与较低延迟。

#### FasterTransformer（FT）
- 优势：成熟C++/CUDA内核库，支持多模型；长期工程积累。
- 要点：需要较多工程集成；适合自定义深度优化、与已有C++服务融合。

#### 选择建议
- 优先vLLM获得工程友好和高性价比；对极致性能/NVIDIA栈，考虑TensorRT-LLM；C++深度定制可选FT。


