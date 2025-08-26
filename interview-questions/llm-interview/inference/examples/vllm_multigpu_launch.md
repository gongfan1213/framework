# vLLM 多卡启动与部署示例（TP/PP）

## 1. 多卡本地启动（Tensor Parallel）
```bash
export MODEL=facebook/opt-6.7b
export TP=2
uvicorn interview-questions/llm-interview/inference/examples/vllm_server:app --host 0.0.0.0 --port 8000
```
- 说明：在 `vllm_server.py` 中通过 `tensor_parallel_size=$TP` 启用TP。

## 2. Pipeline Parallel（实验性思路）
- vLLM 现以 TP 为主；如需 PP，可考虑模型分片加载或使用 LMDeploy 的流水并行配置。

## 3. Dockerfile（简化示例）
```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install vllm fastapi uvicorn[standard]
WORKDIR /app
COPY . /app
ENV MODEL=facebook/opt-1.3b TP=1
CMD ["bash", "-lc", "uvicorn interview-questions/llm-interview/inference/examples/vllm_server:app --host 0.0.0.0 --port 8000"]
```

## 4. docker-compose（多卡）
```yaml
version: '3.8'
services:
  vllm:
    build: .
    image: vllm-svc:latest
    environment:
      - MODEL=facebook/opt-6.7b
      - TP=2
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    runtime: nvidia
    shm_size: 2g
```
- 备注：需设置 `NVIDIA_VISIBLE_DEVICES` 或在 swarm/k8s 指定GPU。
