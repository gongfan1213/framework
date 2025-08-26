### 15 一键部署（docker-compose）说明

本目录提供 vLLM + 网关（OpenAI 兼容）+ Prometheus 的一键部署示例。

#### 目录结构
```text
推理层/compose/
  docker-compose.yml
  gateway/
    Dockerfile
    gateway_app/
      main.py
  prometheus/
    prometheus.yml
```

#### 先决条件
- 安装 Docker 及 GPU 驱动与 runtime（NVIDIA Container Toolkit）；
- 具备可用的 NVIDIA GPU；

#### 启动
```bash
cd 推理层/compose
docker compose up -d --build
```

#### 验证
```bash
curl -H "Authorization: Bearer test-key" \
  http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2-7b","messages":[{"role":"user","content":"你好"}]}'
```

访问 Prometheus: http://127.0.0.1:9090

#### 常见调整
- 修改 `docker-compose.yml` 中 vLLM 模型名称、`--max-model-len`、显存利用率；
- 网关环境变量：`BACKEND_BASE`、`API_KEYS`；
- 增加 Grafana（可选）对接 Prometheus。


