### 04 K8s 上的推理服务模式与 Helm 部署

#### 服务模式
- 单模型服务：每个Deployment一个模型，简单直观；
- 多模型路由：网关按`model`参数转发到不同后端；
- 弹性副本：高峰扩容、低谷缩容，结合TTFT与P99门槛。

#### Helm Chart 结构（示例）
```text
chart/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    hpa.yaml
```

#### values.yaml 关键项
- 镜像、资源限制（含`nvidia.com/gpu`）、副本数、环境变量（模型名、并行策略）、探针、节点亲和。

#### 上线流程
- `helm install` dev -> stage -> prod；
- 金丝雀：调整`replica`与`weight`；
- 回滚：`helm rollback`。


