### 11 实战：TensorRT-LLM 部署与示例

适用于追求极致吞吐与低延迟的 NVIDIA GPU 环境（如 A800、H20、L20）。

#### 环境与准备
- 建议使用官方 Docker 镜像（包含 CUDA/CUDNN/TensorRT）：
```bash
docker run --gpus all -it --rm \
  -p 8001:8001 \
  nvcr.io/nvidia/tritonserver:24.07-py3 bash
```
- 安装 TensorRT-LLM 与依赖（亦可使用集成镜像）：参考官方文档。

#### 构建引擎（以 Llama-2 为例，思路同 Qwen/GLM）
```bash
trt-llm-build --checkpoint_dir /weights/llama-7b \
  -- dtype float16 --tp_size 1 --pp_size 1 \
  --max_input_len 4096 --max_output_len 1024 \
  --use_paged_kv_cache
```

#### 通过 Triton Serving 暴露服务（推荐）
```bash
tritonserver --model-repository=/models --http-port=8001 --grpc-port=8002
```

#### Python 客户端（gRPC/HTTP）
```python
import requests

url = "http://127.0.0.1:8001/v2/models/llm/generate"
payload = {"text_input": "解释Speculative Decoding的思路，50字内。", "max_tokens": 128}
r = requests.post(url, json=payload, timeout=30)
print(r.json())
```

#### 性能要点
- 打开 paged KV；开启 FP8/FP16 混精；配合张量并行；
- 合理 batch 与并发，避免尾延迟过高；
- 结合 Nsight 与 triton metrics 做持续优化。


