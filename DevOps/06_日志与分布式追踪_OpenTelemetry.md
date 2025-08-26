### 06 日志与分布式追踪（OpenTelemetry）

#### 目标
- 将一次用户请求在网关→编排→推理服务→向量库的全链路打通，定位瓶颈与错误。

#### 实施
- SDK：使用 OTel SDK（Python/Node）埋点；
- 传递 TraceContext（W3C traceparent）跨服务；
- 收集器：otel-collector 导出到 Tempo/Jaeger；
- 日志：结构化日志（JSON）并关联 trace_id/span_id。

#### 看板
- 关键Span：分词、前向、采样、网络I/O、检索；
- 异常检测：慢Span榜单、错误堆栈聚合。


