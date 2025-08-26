### 03 Kubernetes 基础与 GPU 调度

#### 核心对象
- Pod/Deployment/Service/Ingress/ConfigMap/Secret/HPA；
- 节点选择/亲和性/污点与容忍；

#### GPU 调度
- 安装 NVIDIA Device Plugin；
- 在 Pod 中声明资源：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: infer }
spec:
  replicas: 2
  selector: { matchLabels: { app: infer } }
  template:
    metadata: { labels: { app: infer } }
    spec:
      containers:
      - name: server
        image: my/infer:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
```

#### 存储与配置
- 使用 ConfigMap/Secret 提供模型路由与密钥；
- 使用 PVC/NVMe 做权重本地缓存；

#### 横向扩缩容
- HPA 基于CPU/GPU利用率或自定义 metrics；


