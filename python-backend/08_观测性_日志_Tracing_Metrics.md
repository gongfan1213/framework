### 08 观测性：日志、Tracing、Metrics

#### 日志
- 结构化JSON日志；按请求ID聚合；

#### Tracing（OTel）
- FastAPI中间件注入trace；
- 传递 traceparent 到下游服务；

#### Metrics
- prometheus-client 暴露 `/metrics`；
- 关键指标：请求率、状态码、延迟分位、依赖后端耗时。


