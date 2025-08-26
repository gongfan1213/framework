### 02 Docker GPU 与多阶段构建、Compose 实战

#### GPU 容器运行
- 安装 NVIDIA Container Toolkit；
- 运行：`docker run --gpus all image:tag`；
- Compose：
```yaml
services:
  infer:
    image: my/infer:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

#### 多阶段构建示例
```Dockerfile
FROM python:3.10-slim as build
WORKDIR /w
COPY requirements.txt .
RUN pip wheel -r requirements.txt -w /wheels

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app
COPY --from=build /wheels /wheels
RUN apt-get update && apt-get install -y python3-pip && pip install /wheels/* && rm -rf /var/lib/apt/lists/*
COPY . .
CMD ["python", "server.py"]
```

#### Compose 编排 vLLM + 网关（参考推理层/compose）
- 见本仓`推理层/compose/`，可扩展加上 Grafana/Alertmanager。


