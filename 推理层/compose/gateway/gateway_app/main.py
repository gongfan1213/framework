import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx
from aiolimiter import AsyncLimiter

OPENAI_BASE = os.getenv("BACKEND_BASE", "http://vllm:8000/v1")
API_KEYS = set(os.getenv("API_KEYS", "test-key").split(","))
limiter = AsyncLimiter(permits=50, time_period=1.0)

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


