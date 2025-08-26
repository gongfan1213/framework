### 01 Docker 基础与 AI 镜像最佳实践

#### 基础概念
- 镜像（Image）、容器（Container）、仓库（Registry）、分层（Layer）、入口（ENTRYPOINT/CMD）。

#### 最佳实践
- 选择合适基础镜像：`nvidia/cuda`（GPU）/`python:slim`（CPU）；
- 多阶段构建：编译依赖与运行环境分离；
- 固定版本与可重复构建：锁定`requirements.txt`/`pip-tools`；
- 非root运行、健康检查、只读根文件系统、最小化攻击面；
- 缓存优化：合理分层（先复制lock文件再`pip install`）。

#### 示例：Python 推理服务镜像
```Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
USER 10001
EXPOSE 8000
CMD ["python", "server.py"]
```


