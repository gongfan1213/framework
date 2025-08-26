### 10 实战：vLLM 部署与示例（Python 栈）

本篇给出基于 vLLM 的快速落地方案，包含服务端启动、客户端调用、批处理与参数要点。

#### 环境准备（conda）
```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm==0.5.3 transformers==4.42.0 accelerate==0.33.0 uvicorn fastapi sse-starlette
```

#### 服务端：启动 OpenAI 兼容 API（SSE 流式）
```bash
python -m vllm.entrypoints.api_server \
  --model Qwen/Qwen2-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen2-7b \
  --host 0.0.0.0 --port 8000
```
- 说明：
  - `--gpu-memory-utilization` 控制显存水位；
  - `--max-model-len` 与上下文上限相关；
  - 默认启用 PagedAttention 与持续批处理。

#### 客户端：Python（OpenAI SDK 兼容）
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen2-7b",
    messages=[{"role": "user", "content": "介绍下KV缓存的作用，用三句话。"}],
    temperature=0.3,
    stream=False,
)
print(resp.choices[0].message.content)
```

#### 客户端：SSE 流式打印
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

with client.chat.completions.stream(
    model="qwen2-7b",
    messages=[{"role": "user", "content": "请逐字输出一句中文诗句。"}],
    temperature=0.7,
) as stream:
    for event in stream:
        if event.type == "token":
            print(event.token, end="", flush=True)
```

#### 批处理与并发建议
- vLLM 默认支持动态/持续批处理；提升吞吐但会拉高 P99，按 SLA 调参：
  - 控制 `--max-num-seqs`、`--gpu-memory-utilization` 与并发连接数；
  - 对于强交互场景，降低最大批并结合优先级队列；
  - 通过监控 TTFT 与 tokens/s 动态优化。

#### 常见参数
- `--dtype`：auto/bfloat16/float16；
- `--enforce-eager`：调试；
- `--max-num-batched-tokens`：限制批内token数；
- `--swap-space`：开启 CPU/NVMe 交换空间（超长上下文）。


