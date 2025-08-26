### 08 VLM 直连与 vLLM 对接（实战）

本篇展示如何以“图像+文本”原生喂给支持多模态的VLM（如 Qwen2-VL）并通过 vLLM 的 OpenAI 接口调用。

#### vLLM 启动（示例）
```bash
python -m vllm.entrypoints.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name qwen2-vl \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8000
```

#### Python 客户端：图像+文本
```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

def encode_image(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()

img_b64 = encode_image("invoice.jpg")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "请从票据中读取抬头与金额，输出JSON"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    ]}
]

resp = client.chat.completions.create(model="qwen2-vl", messages=messages, temperature=0.2)
print(resp.choices[0].message.content)
```

#### 注意
- vLLM 多模态消息体需使用 `content` 的数组结构，元素包含 `text` 与 `image_url`；
- 根据具体VLM的协议适配字段；
- 大图建议先等比缩放，合理控制上下文长度。


