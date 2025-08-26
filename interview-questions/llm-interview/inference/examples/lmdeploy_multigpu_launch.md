# LMDeploy 多卡启动与部署示例（TP/PP）

## 1. 多卡启动（TP）
```bash
export MODEL=internlm/internlm2-7b
export MAX_LEN=4096
# 启动 FastAPI 服务（参考 lmdeploy_server.py）
uvicorn interview-questions/llm-interview/inference/examples/lmdeploy_server:app --host 0.0.0.0 --port 8001
```
- 说明：LMDeploy pipeline 内部可根据后端对多卡做并行；也可使用其命令行/engine进行显式并行配置。

## 2. Pipeline Parallel（示意）
- 通过 LMDeploy engine 配置流水并行与张量并行（参考官方文档），此处给出 docker-compose 方式。

## 3. Dockerfile（简化）
```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install lmdeploy fastapi uvicorn[standard]
WORKDIR /app
COPY . /app
ENV MODEL=internlm/internlm2-7b MAX_LEN=4096
CMD ["bash", "-lc", "uvicorn interview-questions/llm-interview/inference/examples/lmdeploy_server:app --host 0.0.0.0 --port 8001"]
```

## 4. docker-compose（多卡）
```yaml
version: '3.8'
services:
  lmdeploy:
    build: .
    image: lmdeploy-svc:latest
    environment:
      - MODEL=internlm/internlm2-7b
      - MAX_LEN=4096
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    runtime: nvidia
    shm_size: 2g
```
- 提示：如需 TP/PP 细粒度控制，请参考 LMDeploy 官方对 engine 的多并行配置参数。
