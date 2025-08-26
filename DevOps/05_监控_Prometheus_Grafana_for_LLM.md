### 05 监控：Prometheus + Grafana for LLM

#### 监控对象
- 网关：QPS、TTFT、P50/95/99、错误率、流式断开率；
- 推理服务：tokens/s、批大小、GPU利用率、显存水位、队列长度；
- 向量检索：查询耗时、召回量、索引命中；

#### Prometheus 抓取
- 为服务暴露 `/metrics`（OpenMetrics）；
- Prometheus `scrape_configs` 指向服务地址；

#### Grafana 看板
- 常用图表：
  - 延迟分位（TTFT/P95/99）
  - tokens/s 与并发
  - GPU 显存/利用率热力图
  - 错误/超时/熔断趋势

#### 报警建议
- P95/99 超阈值；
- 错误率/超时率突增；
- 显存水位过高与持续队列积压。


