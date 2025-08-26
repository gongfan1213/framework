# Agent工程实践面试题 - 性能与系统监控

## 1. 性能与系统监控能力分析

### 1.1 详细分析项目的性能与系统监控能力

**面试题：请详细分析你的项目中性能与系统监控能力，是否评估过Agent的响应延迟、token成本、memory检索效率？是否使用tracing / caching / LangSmith / PromptLayer等工具进行优化？**

**答案要点：**

**1. 性能监控架构：**
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimization_engine = OptimizationEngine()
        self.alert_manager = AlertManager()
        
        # 监控维度
        self.monitoring_dimensions = {
            "response_latency": ResponseLatencyMonitor(),
            "token_cost": TokenCostMonitor(),
            "memory_efficiency": MemoryEfficiencyMonitor(),
            "throughput": ThroughputMonitor(),
            "error_rate": ErrorRateMonitor()
        }
    
    def start_monitoring(self):
        # 启动性能监控
        for dimension_name, monitor in self.monitoring_dimensions.items():
            monitor.start()
    
    def collect_metrics(self, agent_id, operation_type, metrics_data):
        # 收集性能指标
        timestamp = time.time()
        
        for dimension_name, monitor in self.monitoring_dimensions.items():
            if dimension_name in metrics_data:
                monitor.record_metric(agent_id, operation_type, metrics_data[dimension_name], timestamp)
    
    def analyze_performance(self, agent_id=None, time_range=None):
        # 分析性能
        return self.performance_analyzer.analyze(agent_id, time_range)
    
    def optimize_performance(self, optimization_targets):
        # 性能优化
        return self.optimization_engine.optimize(optimization_targets)
```

**2. 响应延迟监控：**
```python
class ResponseLatencyMonitor:
    def __init__(self):
        self.latency_data = {}
        self.thresholds = {
            "critical": 5000,  # 5秒
            "warning": 2000,   # 2秒
            "optimal": 500     # 500毫秒
        }
    
    def record_metric(self, agent_id, operation_type, latency, timestamp):
        # 记录延迟指标
        if agent_id not in self.latency_data:
            self.latency_data[agent_id] = []
        
        self.latency_data[agent_id].append({
            "operation_type": operation_type,
            "latency": latency,
            "timestamp": timestamp
        })
        
        # 检查阈值
        self._check_thresholds(agent_id, latency, operation_type)
    
    def get_latency_stats(self, agent_id, time_range=None):
        # 获取延迟统计
        if agent_id not in self.latency_data:
            return None
        
        data = self.latency_data[agent_id]
        
        if time_range:
            start_time = time.time() - time_range
            data = [d for d in data if d["timestamp"] >= start_time]
        
        if not data:
            return None
        
        latencies = [d["latency"] for d in data]
        
        return {
            "count": len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "mean": sum(latencies) / len(latencies),
            "median": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99": sorted(latencies)[int(len(latencies) * 0.99)]
        }
    
    def _check_thresholds(self, agent_id, latency, operation_type):
        # 检查延迟阈值
        if latency > self.thresholds["critical"]:
            self._trigger_alert("critical", agent_id, operation_type, latency)
        elif latency > self.thresholds["warning"]:
            self._trigger_alert("warning", agent_id, operation_type, latency)
    
    def _trigger_alert(self, level, agent_id, operation_type, latency):
        # 触发告警
        alert = {
            "level": level,
            "agent_id": agent_id,
            "operation_type": operation_type,
            "latency": latency,
            "threshold": self.thresholds[level],
            "timestamp": time.time()
        }
        
        # 发送告警
        self.alert_manager.send_alert(alert)
```

**3. Token成本监控：**
```python
class TokenCostMonitor:
    def __init__(self):
        self.token_usage = {}
        self.cost_rates = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # 每1K tokens
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3": {"input": 0.015, "output": 0.075}
        }
        self.budget_limits = {
            "daily": 100,    # 每日预算
            "monthly": 2000  # 每月预算
        }
    
    def record_metric(self, agent_id, operation_type, token_data, timestamp):
        # 记录token使用情况
        if agent_id not in self.token_usage:
            self.token_usage[agent_id] = []
        
        # 计算成本
        model = token_data.get("model", "gpt-3.5-turbo")
        input_tokens = token_data.get("input_tokens", 0)
        output_tokens = token_data.get("output_tokens", 0)
        
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        usage_record = {
            "operation_type": operation_type,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "timestamp": timestamp
        }
        
        self.token_usage[agent_id].append(usage_record)
        
        # 检查预算
        self._check_budget_limits(agent_id, cost)
    
    def _calculate_cost(self, model, input_tokens, output_tokens):
        # 计算token成本
        if model not in self.cost_rates:
            return 0
        
        rates = self.cost_rates[model]
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        
        return input_cost + output_cost
    
    def get_cost_stats(self, agent_id=None, time_range=None):
        # 获取成本统计
        if agent_id:
            agents = [agent_id]
        else:
            agents = list(self.token_usage.keys())
        
        total_cost = 0
        total_tokens = 0
        model_usage = {}
        
        for agent in agents:
            if agent in self.token_usage:
                data = self.token_usage[agent]
                
                if time_range:
                    start_time = time.time() - time_range
                    data = [d for d in data if d["timestamp"] >= start_time]
                
                for record in data:
                    total_cost += record["cost"]
                    total_tokens += record["total_tokens"]
                    
                    model = record["model"]
                    if model not in model_usage:
                        model_usage[model] = {"cost": 0, "tokens": 0}
                    model_usage[model]["cost"] += record["cost"]
                    model_usage[model]["tokens"] += record["total_tokens"]
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "model_usage": model_usage,
            "avg_cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0
        }
    
    def _check_budget_limits(self, agent_id, cost):
        # 检查预算限制
        daily_cost = self._get_daily_cost(agent_id)
        monthly_cost = self._get_monthly_cost(agent_id)
        
        if daily_cost > self.budget_limits["daily"]:
            self._trigger_budget_alert("daily", agent_id, daily_cost)
        elif monthly_cost > self.budget_limits["monthly"]:
            self._trigger_budget_alert("monthly", agent_id, monthly_cost)
    
    def _get_daily_cost(self, agent_id):
        # 获取每日成本
        today = datetime.now().date()
        daily_cost = 0
        
        if agent_id in self.token_usage:
            for record in self.token_usage[agent_id]:
                record_date = datetime.fromtimestamp(record["timestamp"]).date()
                if record_date == today:
                    daily_cost += record["cost"]
        
        return daily_cost
    
    def _get_monthly_cost(self, agent_id):
        # 获取每月成本
        current_month = datetime.now().month
        current_year = datetime.now().year
        monthly_cost = 0
        
        if agent_id in self.token_usage:
            for record in self.token_usage[agent_id]:
                record_date = datetime.fromtimestamp(record["timestamp"])
                if record_date.month == current_month and record_date.year == current_year:
                    monthly_cost += record["cost"]
        
        return monthly_cost
```

**4. 内存检索效率监控：**
```python
class MemoryEfficiencyMonitor:
    def __init__(self):
        self.memory_metrics = {}
        self.retrieval_patterns = {}
        self.cache_hit_rates = {}
    
    def record_metric(self, agent_id, operation_type, memory_data, timestamp):
        # 记录内存检索指标
        if agent_id not in self.memory_metrics:
            self.memory_metrics[agent_id] = []
        
        metric_record = {
            "operation_type": operation_type,
            "retrieval_time": memory_data.get("retrieval_time", 0),
            "cache_hit": memory_data.get("cache_hit", False),
            "result_count": memory_data.get("result_count", 0),
            "query_complexity": memory_data.get("query_complexity", "simple"),
            "memory_size": memory_data.get("memory_size", 0),
            "timestamp": timestamp
        }
        
        self.memory_metrics[agent_id].append(metric_record)
        
        # 更新缓存命中率
        self._update_cache_hit_rate(agent_id, memory_data.get("cache_hit", False))
    
    def get_efficiency_stats(self, agent_id, time_range=None):
        # 获取效率统计
        if agent_id not in self.memory_metrics:
            return None
        
        data = self.memory_metrics[agent_id]
        
        if time_range:
            start_time = time.time() - time_range
            data = [d for d in data if d["timestamp"] >= start_time]
        
        if not data:
            return None
        
        retrieval_times = [d["retrieval_time"] for d in data]
        cache_hits = [d["cache_hit"] for d in data]
        result_counts = [d["result_count"] for d in data]
        
        return {
            "total_operations": len(data),
            "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
            "cache_hit_rate": sum(cache_hits) / len(cache_hits),
            "avg_result_count": sum(result_counts) / len(result_counts),
            "efficiency_score": self._calculate_efficiency_score(data)
        }
    
    def _calculate_efficiency_score(self, data):
        # 计算效率分数
        avg_retrieval_time = sum(d["retrieval_time"] for d in data) / len(data)
        cache_hit_rate = sum(d["cache_hit"] for d in data) / len(data)
        
        # 时间分数（越短越好）
        time_score = max(0, 1 - (avg_retrieval_time / 1000))  # 假设1秒为基准
        
        # 缓存分数
        cache_score = cache_hit_rate
        
        # 综合分数
        return (time_score * 0.6 + cache_score * 0.4)
    
    def _update_cache_hit_rate(self, agent_id, cache_hit):
        # 更新缓存命中率
        if agent_id not in self.cache_hit_rates:
            self.cache_hit_rates[agent_id] = {"hits": 0, "total": 0}
        
        self.cache_hit_rates[agent_id]["total"] += 1
        if cache_hit:
            self.cache_hit_rates[agent_id]["hits"] += 1
```

**5. 性能优化工具集成：**
```python
class PerformanceOptimizationTools:
    def __init__(self):
        self.tracing_tool = TracingTool()
        self.caching_tool = CachingTool()
        self.langsmith_tool = LangSmithTool()
        self.promptlayer_tool = PromptLayerTool()
        self.optimization_strategies = {
            "tracing": self._apply_tracing_optimization,
            "caching": self._apply_caching_optimization,
            "langsmith": self._apply_langsmith_optimization,
            "promptlayer": self._apply_promptlayer_optimization
        }
    
    def apply_optimization(self, optimization_type, config):
        # 应用性能优化
        if optimization_type in self.optimization_strategies:
            return self.optimization_strategies[optimization_type](config)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    def _apply_tracing_optimization(self, config):
        # 应用分布式追踪优化
        return self.tracing_tool.setup_tracing(config)
    
    def _apply_caching_optimization(self, config):
        # 应用缓存优化
        return self.caching_tool.setup_caching(config)
    
    def _apply_langsmith_optimization(self, config):
        # 应用LangSmith优化
        return self.langsmith_tool.setup_monitoring(config)
    
    def _apply_promptlayer_optimization(self, config):
        # 应用PromptLayer优化
        return self.promptlayer_tool.setup_monitoring(config)

class TracingTool:
    def __init__(self):
        self.tracer = None
        self.spans = {}
    
    def setup_tracing(self, config):
        # 设置分布式追踪
        service_name = config.get("service_name", "agent-system")
        endpoint = config.get("endpoint", "http://localhost:14268/api/traces")
        
        # 初始化Jaeger tracer
        self.tracer = opentracing.init_tracer(service_name, config={
            'sampler': {'type': 'const', 'param': True},
            'local_agent': {'reporting_host': endpoint}
        })
        
        return {"status": "success", "tracer": self.tracer}
    
    def start_span(self, operation_name, tags=None):
        # 开始追踪span
        if self.tracer:
            span = self.tracer.start_span(operation_name, tags=tags or {})
            span_id = str(uuid.uuid4())
            self.spans[span_id] = span
            return span_id
        return None
    
    def finish_span(self, span_id, tags=None):
        # 结束追踪span
        if span_id in self.spans:
            span = self.spans[span_id]
            if tags:
                for key, value in tags.items():
                    span.set_tag(key, value)
            span.finish()
            del self.spans[span_id]

class CachingTool:
    def __init__(self):
        self.cache_layers = {
            "memory": MemoryCache(),
            "redis": RedisCache(),
            "distributed": DistributedCache()
        }
        self.cache_strategies = {
            "lru": LRUCacheStrategy(),
            "ttl": TTLCacheStrategy(),
            "write_through": WriteThroughStrategy(),
            "write_back": WriteBackStrategy()
        }
    
    def setup_caching(self, config):
        # 设置缓存系统
        cache_type = config.get("type", "memory")
        strategy = config.get("strategy", "lru")
        
        cache_layer = self.cache_layers.get(cache_type)
        cache_strategy = self.cache_strategies.get(strategy)
        
        if not cache_layer or not cache_strategy:
            raise ValueError(f"Invalid cache configuration: {cache_type}, {strategy}")
        
        # 配置缓存
        cache_layer.configure(config.get("config", {}))
        cache_strategy.configure(config.get("strategy_config", {}))
        
        return {
            "status": "success",
            "cache_layer": cache_layer,
            "cache_strategy": cache_strategy
        }

class LangSmithTool:
    def __init__(self):
        self.client = None
        self.project_name = None
    
    def setup_monitoring(self, config):
        # 设置LangSmith监控
        api_key = config.get("api_key")
        project_name = config.get("project_name", "agent-system")
        
        if not api_key:
            raise ValueError("LangSmith API key is required")
        
        # 初始化LangSmith客户端
        self.client = langsmith.Client(api_key=api_key)
        self.project_name = project_name
        
        return {"status": "success", "client": self.client}
    
    def log_run(self, run_data):
        # 记录LangSmith运行数据
        if self.client:
            run = self.client.create_run(
                project_name=self.project_name,
                **run_data
            )
            return run
        return None

class PromptLayerTool:
    def __init__(self):
        self.api_key = None
        self.project_name = None
    
    def setup_monitoring(self, config):
        # 设置PromptLayer监控
        api_key = config.get("api_key")
        project_name = config.get("project_name", "agent-system")
        
        if not api_key:
            raise ValueError("PromptLayer API key is required")
        
        self.api_key = api_key
        self.project_name = project_name
        
        return {"status": "success"}
    
    def log_prompt(self, prompt_data):
        # 记录PromptLayer数据
        if self.api_key:
            # 这里实现具体的PromptLayer API调用
            pass
```

**6. 系统监控仪表板：**
```python
class SystemMonitoringDashboard:
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.metrics_aggregator = MetricsAggregator()
        self.visualization_engine = VisualizationEngine()
        self.alert_manager = AlertManager()
    
    def generate_dashboard(self, time_range="24h"):
        # 生成监控仪表板
        metrics = self.metrics_aggregator.aggregate_metrics(time_range)
        
        dashboard_data = {
            "overview": self._generate_overview(metrics),
            "performance": self._generate_performance_charts(metrics),
            "costs": self._generate_cost_charts(metrics),
            "efficiency": self._generate_efficiency_charts(metrics),
            "alerts": self._get_active_alerts()
        }
        
        return dashboard_data
    
    def _generate_overview(self, metrics):
        # 生成概览数据
        return {
            "total_agents": len(metrics.get("agents", [])),
            "total_requests": metrics.get("total_requests", 0),
            "avg_response_time": metrics.get("avg_response_time", 0),
            "total_cost": metrics.get("total_cost", 0),
            "error_rate": metrics.get("error_rate", 0),
            "system_health": self._calculate_system_health(metrics)
        }
    
    def _generate_performance_charts(self, metrics):
        # 生成性能图表数据
        return {
            "response_time_trend": self._get_response_time_trend(metrics),
            "throughput_trend": self._get_throughput_trend(metrics),
            "error_rate_trend": self._get_error_rate_trend(metrics),
            "agent_performance": self._get_agent_performance(metrics)
        }
    
    def _generate_cost_charts(self, metrics):
        # 生成成本图表数据
        return {
            "cost_trend": self._get_cost_trend(metrics),
            "model_usage": self._get_model_usage(metrics),
            "agent_costs": self._get_agent_costs(metrics),
            "budget_utilization": self._get_budget_utilization(metrics)
        }
    
    def _generate_efficiency_charts(self, metrics):
        # 生成效率图表数据
        return {
            "memory_efficiency": self._get_memory_efficiency(metrics),
            "cache_hit_rates": self._get_cache_hit_rates(metrics),
            "resource_utilization": self._get_resource_utilization(metrics)
        }
    
    def _calculate_system_health(self, metrics):
        # 计算系统健康度
        health_factors = {
            "response_time": self._normalize_response_time(metrics.get("avg_response_time", 0)),
            "error_rate": 1 - metrics.get("error_rate", 0),
            "availability": metrics.get("availability", 1.0),
            "efficiency": metrics.get("efficiency_score", 0.5)
        }
        
        # 加权平均
        weights = {"response_time": 0.3, "error_rate": 0.3, "availability": 0.2, "efficiency": 0.2}
        health_score = sum(health_factors[factor] * weights[factor] for factor in health_factors)
        
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.7:
            return "good"
        elif health_score >= 0.5:
            return "fair"
        else:
            return "poor"
```

**7. 性能优化建议：**
```python
class PerformanceOptimizationAdvisor:
    def __init__(self):
        self.optimization_rules = {
            "high_latency": self._suggest_latency_optimization,
            "high_cost": self._suggest_cost_optimization,
            "low_efficiency": self._suggest_efficiency_optimization,
            "high_error_rate": self._suggest_error_optimization
        }
    
    def generate_optimization_suggestions(self, performance_data):
        # 生成优化建议
        suggestions = []
        
        # 分析性能问题
        issues = self._identify_performance_issues(performance_data)
        
        # 生成建议
        for issue in issues:
            if issue["type"] in self.optimization_rules:
                suggestion = self.optimization_rules[issue["type"]](issue)
                suggestions.append(suggestion)
        
        return suggestions
    
    def _identify_performance_issues(self, performance_data):
        # 识别性能问题
        issues = []
        
        # 检查响应延迟
        if performance_data.get("avg_response_time", 0) > 2000:
            issues.append({
                "type": "high_latency",
                "severity": "high",
                "current_value": performance_data["avg_response_time"],
                "threshold": 2000
            })
        
        # 检查成本
        if performance_data.get("total_cost", 0) > 100:
            issues.append({
                "type": "high_cost",
                "severity": "medium",
                "current_value": performance_data["total_cost"],
                "threshold": 100
            })
        
        # 检查效率
        if performance_data.get("efficiency_score", 1.0) < 0.7:
            issues.append({
                "type": "low_efficiency",
                "severity": "medium",
                "current_value": performance_data["efficiency_score"],
                "threshold": 0.7
            })
        
        return issues
    
    def _suggest_latency_optimization(self, issue):
        # 延迟优化建议
        return {
            "issue_type": "high_latency",
            "severity": issue["severity"],
            "description": f"响应延迟过高: {issue['current_value']}ms",
            "suggestions": [
                "启用缓存机制减少重复计算",
                "优化数据库查询和索引",
                "使用异步处理非关键操作",
                "实施负载均衡分散请求",
                "优化模型调用减少token使用"
            ],
            "priority": "high"
        }
    
    def _suggest_cost_optimization(self, issue):
        # 成本优化建议
        return {
            "issue_type": "high_cost",
            "severity": issue["severity"],
            "description": f"成本过高: ${issue['current_value']}",
            "suggestions": [
                "使用更便宜的模型进行简单任务",
                "实施token使用限制和预算控制",
                "优化prompt减少token消耗",
                "启用结果缓存避免重复调用",
                "使用批量处理减少API调用次数"
            ],
            "priority": "medium"
        }
    
    def _suggest_efficiency_optimization(self, issue):
        # 效率优化建议
        return {
            "issue_type": "low_efficiency",
            "severity": issue["severity"],
            "description": f"系统效率低: {issue['current_value']}",
            "suggestions": [
                "优化内存检索算法",
                "提高缓存命中率",
                "实施智能任务调度",
                "优化资源分配策略",
                "使用更高效的序列化格式"
            ],
            "priority": "medium"
        }
    
    def _suggest_error_optimization(self, issue):
        # 错误率优化建议
        return {
            "issue_type": "high_error_rate",
            "severity": issue["severity"],
            "description": f"错误率过高: {issue['current_value']}%",
            "suggestions": [
                "实施重试机制和错误恢复",
                "改进错误处理和日志记录",
                "增加输入验证和参数检查",
                "实施熔断器模式防止级联失败",
                "优化网络连接和超时设置"
            ],
            "priority": "high"
        }
```