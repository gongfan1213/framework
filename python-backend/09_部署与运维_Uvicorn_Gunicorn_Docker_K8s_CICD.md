### 09 部署与运维：Uvicorn/Gunicorn、Docker、K8s、CI/CD

#### 进程模型
- 开发：单进程 uvicorn；
- 生产：gunicorn + uvicorn worker；多副本背后配负载均衡。

#### Docker 化
- 基于 `python:slim` 或 CUDA 镜像；
- 非root、健康检查、只读FS、最小体积；

#### K8s
- 资源请求/限制、HPA、探针、ConfigMap/Secret、Ingress；

#### CI/CD
- 构建→推送镜像→Helm 部署；门禁：测试通过、lint、镜像扫描。


