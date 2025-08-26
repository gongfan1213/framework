import os
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

model_name = os.environ.get("MODEL", "facebook/opt-1.3b")
max_model_len = int(os.environ.get("MAX_LEN", 4096))

llm = LLM(
    model=model_name,
    tensor_parallel_size=int(os.environ.get("TP", 1)),
    max_model_len=max_model_len,
    trust_remote_code=True,
)

class GenReq(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256

@app.post("/generate")
async def generate(req: GenReq):
    params = SamplingParams(
        temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens
    )
    outputs = llm.generate([req.prompt], params)
    return {"text": outputs[0].outputs[0].text}

# vLLM 内置连续批处理与KV缓存，无需额外管理
# 运行：uvicorn vllm_server:app --host 0.0.0.0 --port 8000
