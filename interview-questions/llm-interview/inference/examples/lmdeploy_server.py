import os
from fastapi import FastAPI
from pydantic import BaseModel
from lmdeploy import pipeline

app = FastAPI()

model_name = os.environ.get("MODEL", "internlm/internlm2-1_8b")
pipe = pipeline(model_name, backend_config={'session_len': int(os.environ.get('MAX_LEN', 4096))})

class GenReq(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256

@app.post("/generate")
async def generate(req: GenReq):
    gen_cfg = dict(temperature=req.temperature, top_p=req.top_p, max_new_tokens=req.max_tokens)
    resp = pipe([req.prompt], gen_cfg)
    return {"text": resp[0].text}

# 运行：uvicorn lmdeploy_server:app --host 0.0.0.0 --port 8001
# 对比：vLLM 内置连续批处理、KV缓存；LMDeploy 具备推理图优化与多框架支持
