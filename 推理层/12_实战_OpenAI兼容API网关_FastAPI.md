### 12 实战：OpenAI 兼容 API 网关（FastAPI）

目标：在网关层实现鉴权、限流、路由到后端 vLLM/TensorRT-LLM，暴露 OpenAI 兼容接口。

#### 依赖
```bash
conda create -n gateway python=3.10 -y
conda activate gateway
pip install fastapi uvicorn httpx==0.27.0 pydantic==2.* aiolimiter
```

#### 代码：main.py（最小可用网关）
```python
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx
from aiolimiter import AsyncLimiter

OPENAI_BASE = os.getenv("BACKEND_BASE", "http://127.0.0.1:8000/v1")  # 指向 vLLM
API_KEYS = set(os.getenv("API_KEYS", "test-key").split(","))
limiter = AsyncLimiter(permits=50, time_period=1.0)  # 50 RPS 示例

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.3
    stream: bool | None = False

def check_key(request: Request):
    key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/v1/chat/completions")
async def chat_completions(req: Request, body: ChatCompletionRequest):
    check_key(req)
    async with limiter:
        async with httpx.AsyncClient(timeout=60) as client:
            if body.stream:
                async def gen():
                    async with client.stream("POST", f"{OPENAI_BASE}/chat/completions", json=body.model_dump()) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                return StreamingResponse(gen(), media_type="text/event-stream")
            else:
                r = await client.post(f"{OPENAI_BASE}/chat/completions", json=body.model_dump())
                return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

#### 启动
```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

#### 说明
- 生产可扩展：租户级限流、AB灰度、观测埋点、超时与重试策略、重写采样参数白名单。


