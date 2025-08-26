### 13 实战：压测脚本（SSE/HTTP, Python asyncio）

用于评估网关或 vLLM/TensorRT-LLM 的性能（TTFT、吞吐）

#### 依赖
```bash
pip install httpx==0.27.0 anyio rich
```

#### 压测脚本：load_test.py
```python
import anyio
import httpx
import time
from rich.console import Console
from statistics import mean

console = Console()

BASE_URL = "http://127.0.0.1:9000/v1"  # 指向网关
API_KEY = "test-key"
CONCURRENCY = 32
REQUESTS = 200

prompt = "用三句话说明KV缓存的好处。"

async def one_req(client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": "qwen2-7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = await client.post(f"{BASE_URL}/chat/completions", json=data, headers=headers)
    tt = time.perf_counter() - t0
    return tt, r.status_code

async def main():
    latencies = []
    ok = 0
    async with httpx.AsyncClient(timeout=60) as client:
        async with anyio.create_task_group() as tg:
            sem = anyio.Semaphore(CONCURRENCY)

            async def worker(i):
                nonlocal ok
                async with sem:
                    tt, code = await one_req(client)
                    latencies.append(tt)
                    if code == 200:
                        ok += 1

            for i in range(REQUESTS):
                tg.start_soon(worker, i)

    latencies.sort()
    p50 = latencies[int(0.5*len(latencies))]
    p95 = latencies[int(0.95*len(latencies))]
    p99 = latencies[int(0.99*len(latencies))-1]
    console.print({
        "ok": ok,
        "total": REQUESTS,
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "avg": round(mean(latencies), 3),
    })

if __name__ == "__main__":
    anyio.run(main)
```

#### 使用
```bash
python load_test.py
```


