### 10 Grafana 看板JSON 与 Prometheus 报警规则

#### 看板JSON（片段示例）
```json
{
  "title": "LLM Inference",
  "panels": [
    {"type": "timeseries", "title": "TTFT P95", "targets": [{"expr": "histogram_quantile(0.95, sum(rate(ttft_seconds_bucket[5m])) by (le))"}]},
    {"type": "timeseries", "title": "Tokens/s", "targets": [{"expr": "sum(rate(tokens_generated_total[1m]))"}]}
  ]
}
```

#### 报警规则（PrometheusRule 片段）
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: llm-alerts
spec:
  groups:
  - name: llm
    rules:
    - alert: HighP95Latency
      expr: histogram_quantile(0.95, sum(rate(ttft_seconds_bucket[5m])) by (le)) > 1.5
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "P95 TTFT high"
```


