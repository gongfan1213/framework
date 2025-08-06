# LangGraph 多Agents 搭建教程 - 第4部分：生产环境部署

## 概述

第4部分将深入探讨LangGraph在生产环境中的部署、监控、扩展等关键话题。我们将从开发环境到生产环境的完整流程，包括容器化、负载均衡、监控告警、性能优化等。

## 4.1 生产环境架构设计

### 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   LangGraph     │
│   (Nginx/HAProxy│────│   (FastAPI)     │────│   Applications  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   PostgreSQL    │    │   Monitoring    │
│   (Checkpoints) │    │   (Checkpoints) │    │   (Prometheus)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

#### 1. API网关层
```python
# api_gateway.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import Dict, Any

app = FastAPI(title="LangGraph API Gateway")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkflowRequest(BaseModel):
    workflow_type: str
    input_data: Dict[str, Any]
    user_id: str
    session_id: str

class WorkflowResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    execution_time: float
    session_id: str

@app.post("/workflow/execute", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """执行工作流"""
    try:
        start_time = time.time()
        
        # 根据工作流类型选择相应的应用
        app_instance = get_workflow_app(request.workflow_type)
        
        # 执行工作流
        result = await app_instance.ainvoke(request.input_data)
        
        execution_time = time.time() - start_time
        
        return WorkflowResponse(
            status="success",
            result=result,
            execution_time=execution_time,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 2. 工作流管理器
```python
# workflow_manager.py
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.cache import RedisCache
import redis
import psycopg2
from typing import Dict, Any

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        # 初始化数据库连接
        self.db_config = {
            "host": "localhost",
            "database": "langgraph",
            "user": "langgraph_user",
            "password": "secure_password"
        }
        
        # 初始化Redis缓存
        self.redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # 初始化检查点保存器
        self.checkpointer = PostgresSaver.from_conn_string(
            "postgresql://langgraph_user:secure_password@localhost/langgraph"
        )
        
        # 初始化缓存
        self.cache = RedisCache(redis_client=self.redis_client)
        
        # 工作流注册表
        self.workflows = {}
    
    def register_workflow(self, name: str, workflow: StateGraph):
        """注册工作流"""
        compiled_workflow = workflow.compile(
            checkpointer=self.checkpointer,
            cache=self.cache
        )
        self.workflows[name] = compiled_workflow
    
    async def execute_workflow(self, name: str, input_data: Dict[str, Any], config: Dict[str, Any]):
        """执行工作流"""
        if name not in self.workflows:
            raise ValueError(f"Workflow {name} not found")
        
        workflow = self.workflows[name]
        result = await workflow.ainvoke(input_data, config=config)
        return result
    
    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        status = self.redis_client.get(f"workflow_status:{session_id}")
        return {"session_id": session_id, "status": status}
    
    def cleanup_session(self, session_id: str):
        """清理会话数据"""
        self.redis_client.delete(f"workflow_status:{session_id}")
```

## 4.2 容器化部署

### Docker配置

#### 1. 应用Dockerfile
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. requirements.txt
```txt
# requirements.txt
langgraph==0.0.40
langchain==0.1.0
langchain-openai==0.0.5
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
psycopg2-binary==2.9.9
prometheus-client==0.19.0
pydantic==2.5.0
```

#### 3. Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  api-gateway:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=langgraph
      - POSTGRES_USER=langgraph_user
      - POSTGRES_PASSWORD=secure_password
    depends_on:
      - redis
      - postgres
    networks:
      - langgraph-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - langgraph-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=langgraph
      - POSTGRES_USER=langgraph_user
      - POSTGRES_PASSWORD=secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - langgraph-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api-gateway
    networks:
      - langgraph-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - langgraph-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - langgraph-network

volumes:
  redis_data:
  postgres_data:
  grafana_data:

networks:
  langgraph-network:
    driver: bridge
```

#### 4. Nginx配置
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream langgraph_backend {
        server api-gateway:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://langgraph_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://langgraph_backend/health;
        }
    }
}
```

## 4.3 监控与告警

### 1. 性能监控

#### Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'langgraph-api'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

#### 应用指标收集
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Request
import time

# 定义指标
REQUEST_COUNT = Counter('langgraph_requests_total', 'Total requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('langgraph_request_duration_seconds', 'Request duration', ['endpoint'])
WORKFLOW_EXECUTION_TIME = Histogram('langgraph_workflow_execution_seconds', 'Workflow execution time', ['workflow_type'])
ACTIVE_SESSIONS = Gauge('langgraph_active_sessions', 'Active sessions')
ERROR_COUNT = Counter('langgraph_errors_total', 'Total errors', ['error_type'])

class MetricsMiddleware:
    """指标中间件"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # 记录请求开始
            REQUEST_COUNT.labels(
                endpoint=scope["path"],
                method=scope["method"]
            ).inc()
            
            # 处理请求
            await self.app(scope, receive, send)
            
            # 记录请求时长
            duration = time.time() - start_time
            REQUEST_DURATION.labels(endpoint=scope["path"]).observe(duration)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """指标中间件"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method
    ).inc()
    
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """暴露指标端点"""
    return generate_latest()
```

### 2. 日志管理

#### 结构化日志
```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 添加处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_workflow_start(self, workflow_type: str, session_id: str, input_data: Dict[str, Any]):
        """记录工作流开始"""
        log_data = {
            "event": "workflow_start",
            "workflow_type": workflow_type,
            "session_id": session_id,
            "input_data": input_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_workflow_complete(self, workflow_type: str, session_id: str, result: Dict[str, Any], duration: float):
        """记录工作流完成"""
        log_data = {
            "event": "workflow_complete",
            "workflow_type": workflow_type,
            "session_id": session_id,
            "result": result,
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, workflow_type: str, session_id: str, error: str, stack_trace: str = None):
        """记录错误"""
        log_data = {
            "event": "workflow_error",
            "workflow_type": workflow_type,
            "session_id": session_id,
            "error": error,
            "stack_trace": stack_trace,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.error(json.dumps(log_data))

# 使用示例
logger = StructuredLogger("langgraph")

@app.post("/workflow/execute")
async def execute_workflow(request: WorkflowRequest):
    """执行工作流"""
    try:
        start_time = time.time()
        
        # 记录开始
        logger.log_workflow_start(
            request.workflow_type,
            request.session_id,
            request.input_data
        )
        
        # 执行工作流
        result = await workflow_manager.execute_workflow(
            request.workflow_type,
            request.input_data,
            {"session_id": request.session_id}
        )
        
        duration = time.time() - start_time
        
        # 记录完成
        logger.log_workflow_complete(
            request.workflow_type,
            request.session_id,
            result,
            duration
        )
        
        return WorkflowResponse(
            status="success",
            result=result,
            execution_time=duration,
            session_id=request.session_id
        )
        
    except Exception as e:
        # 记录错误
        logger.log_error(
            request.workflow_type,
            request.session_id,
            str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. 告警配置

#### Grafana告警规则
```yaml
# grafana_alerts.yml
groups:
  - name: langgraph_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(langgraph_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: SlowWorkflowExecution
        expr: histogram_quantile(0.95, rate(langgraph_workflow_execution_seconds_bucket[5m])) > 30
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Slow workflow execution"
          description: "95th percentile execution time is {{ $value }} seconds"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
```

## 4.4 负载均衡与扩展

### 1. 水平扩展

#### 自动扩缩容配置
```yaml
# kubernetes_deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-api
  template:
    metadata:
      labels:
        app: langgraph-api
    spec:
      containers:
      - name: langgraph-api
        image: langgraph-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-service
spec:
  selector:
    app: langgraph-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### HPA (Horizontal Pod Autoscaler)
```yaml
# hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langgraph-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langgraph-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2. 会话管理

#### 分布式会话存储
```python
# session_manager.py
import redis
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class DistributedSessionManager:
    """分布式会话管理器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_ttl = 3600  # 1小时
    
    def create_session(self, session_id: str, initial_data: Dict[str, Any]) -> bool:
        """创建会话"""
        session_data = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "data": initial_data,
            "status": "active"
        }
        
        return self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps(session_data)
        )
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话数据"""
        session_data = self.redis.get(f"session:{session_id}")
        
        if session_data:
            session = json.loads(session_data)
            # 更新最后访问时间
            session["last_accessed"] = datetime.utcnow().isoformat()
            self.redis.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session)
            )
            return session
        
        return None
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话数据"""
        session = self.get_session(session_id)
        
        if session:
            session["data"].update(data)
            session["last_accessed"] = datetime.utcnow().isoformat()
            
            return self.redis.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session)
            )
        
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        return bool(self.redis.delete(f"session:{session_id}"))
    
    def get_active_sessions(self) -> list:
        """获取活跃会话列表"""
        pattern = "session:*"
        session_keys = self.redis.keys(pattern)
        
        active_sessions = []
        for key in session_keys:
            session_data = self.redis.get(key)
            if session_data:
                session = json.loads(session_data)
                if session.get("status") == "active":
                    active_sessions.append(session)
        
        return active_sessions
```

## 4.5 安全配置

### 1. 认证与授权

#### JWT认证
```python
# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证令牌"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/workflow/execute")
async def execute_workflow(
    request: WorkflowRequest,
    current_user: str = Depends(verify_token)
):
    """执行工作流（需要认证）"""
    # 添加用户信息到请求
    request.input_data["user_id"] = current_user
    
    # 执行工作流
    result = await workflow_manager.execute_workflow(
        request.workflow_type,
        request.input_data,
        {"session_id": request.session_id, "user_id": current_user}
    )
    
    return WorkflowResponse(
        status="success",
        result=result,
        execution_time=0.0,
        session_id=request.session_id
    )
```

### 2. 速率限制

#### 速率限制中间件
```python
# rate_limiter.py
from fastapi import HTTPException
import redis
import time
from typing import Dict, Tuple

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.limits = {
            "default": {"requests": 100, "window": 3600},  # 100请求/小时
            "workflow": {"requests": 10, "window": 3600},   # 10工作流/小时
            "api": {"requests": 1000, "window": 3600}       # 1000API调用/小时
        }
    
    def check_rate_limit(self, key: str, limit_type: str = "default") -> bool:
        """检查速率限制"""
        limit = self.limits.get(limit_type, self.limits["default"])
        
        current_time = int(time.time())
        window_start = current_time - limit["window"]
        
        # 获取当前时间窗口内的请求数
        requests = self.redis.zcount(key, window_start, current_time)
        
        if requests >= limit["requests"]:
            return False
        
        # 添加当前请求
        self.redis.zadd(key, {str(current_time): current_time})
        self.redis.expire(key, limit["window"])
        
        return True
    
    def get_remaining_requests(self, key: str, limit_type: str = "default") -> int:
        """获取剩余请求数"""
        limit = self.limits.get(limit_type, self.limits["default"])
        current_time = int(time.time())
        window_start = current_time - limit["window"]
        
        requests = self.redis.zcount(key, window_start, current_time)
        return max(0, limit["requests"] - requests)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """速率限制中间件"""
    client_ip = request.client.host
    user_id = request.headers.get("X-User-ID", "anonymous")
    
    # 检查IP限制
    ip_key = f"rate_limit:ip:{client_ip}"
    if not rate_limiter.check_rate_limit(ip_key, "api"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for IP"
        )
    
    # 检查用户限制
    user_key = f"rate_limit:user:{user_id}"
    if not rate_limiter.check_rate_limit(user_key, "workflow"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for user"
        )
    
    response = await call_next(request)
    
    # 添加剩余请求数到响应头
    remaining_ip = rate_limiter.get_remaining_requests(ip_key, "api")
    remaining_user = rate_limiter.get_remaining_requests(user_key, "workflow")
    
    response.headers["X-RateLimit-Remaining-IP"] = str(remaining_ip)
    response.headers["X-RateLimit-Remaining-User"] = str(remaining_user)
    
    return response
```

## 4.6 备份与恢复

### 1. 数据备份策略

#### PostgreSQL备份脚本
```bash
#!/bin/bash
# backup_postgres.sh

# 配置
DB_HOST="localhost"
DB_NAME="langgraph"
DB_USER="langgraph_user"
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_DIR/langgraph_$DATE.sql

# 压缩备份文件
gzip $BACKUP_DIR/langgraph_$DATE.sql

# 删除7天前的备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: langgraph_$DATE.sql.gz"
```

#### Redis备份脚本
```bash
#!/bin/bash
# backup_redis.sh

# 配置
REDIS_HOST="localhost"
REDIS_PORT="6379"
BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行Redis备份
redis-cli -h $REDIS_HOST -p $REDIS_PORT BGSAVE

# 等待备份完成
sleep 10

# 复制RDB文件
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# 压缩备份文件
gzip $BACKUP_DIR/redis_$DATE.rdb

# 删除7天前的备份
find $BACKUP_DIR -name "*.rdb.gz" -mtime +7 -delete

echo "Redis backup completed: redis_$DATE.rdb.gz"
```

### 2. 恢复脚本

#### 数据恢复脚本
```python
# restore_data.py
import psycopg2
import redis
import subprocess
import os
from datetime import datetime

class DataRestore:
    """数据恢复工具"""
    
    def __init__(self, db_config: dict, redis_config: dict):
        self.db_config = db_config
        self.redis_config = redis_config
    
    def restore_postgres(self, backup_file: str):
        """恢复PostgreSQL数据"""
        try:
            # 连接数据库
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = True
            
            # 执行恢复
            with open(backup_file, 'r') as f:
                sql_content = f.read()
            
            cursor = conn.cursor()
            cursor.execute(sql_content)
            cursor.close()
            conn.close()
            
            print(f"PostgreSQL restore completed: {backup_file}")
            
        except Exception as e:
            print(f"PostgreSQL restore failed: {e}")
    
    def restore_redis(self, backup_file: str):
        """恢复Redis数据"""
        try:
            # 停止Redis
            subprocess.run(["systemctl", "stop", "redis"], check=True)
            
            # 复制备份文件
            subprocess.run(["cp", backup_file, "/var/lib/redis/dump.rdb"], check=True)
            
            # 启动Redis
            subprocess.run(["systemctl", "start", "redis"], check=True)
            
            print(f"Redis restore completed: {backup_file}")
            
        except Exception as e:
            print(f"Redis restore failed: {e}")
    
    def verify_restore(self):
        """验证恢复结果"""
        try:
            # 验证PostgreSQL
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM langgraph_checkpoints")
            checkpoint_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            # 验证Redis
            r = redis.Redis(**self.redis_config)
            session_count = len(r.keys("session:*"))
            
            print(f"Restore verification:")
            print(f"  - PostgreSQL checkpoints: {checkpoint_count}")
            print(f"  - Redis sessions: {session_count}")
            
        except Exception as e:
            print(f"Restore verification failed: {e}")

# 使用示例
if __name__ == "__main__":
    db_config = {
        "host": "localhost",
        "database": "langgraph",
        "user": "langgraph_user",
        "password": "secure_password"
    }
    
    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0
    }
    
    restore = DataRestore(db_config, redis_config)
    
    # 恢复数据
    restore.restore_postgres("/backups/postgres/langgraph_20240101_120000.sql")
    restore.restore_redis("/backups/redis/redis_20240101_120000.rdb")
    
    # 验证恢复
    restore.verify_restore()
```

## 4.7 性能优化

### 1. 缓存策略

#### 多级缓存
```python
# cache_manager.py
import redis
from functools import wraps
import hashlib
import json
from typing import Any, Optional

class MultiLevelCache:
    """多级缓存管理器"""
    
    def __init__(self):
        # 内存缓存（L1）
        self.memory_cache = {}
        
        # Redis缓存（L2）
        self.redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        # 先查内存缓存
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 再查Redis缓存
        value = self.redis_client.get(key)
        if value:
            # 解析JSON并存入内存缓存
            parsed_value = json.loads(value)
            self.memory_cache[key] = parsed_value
            return parsed_value
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存值"""
        # 存入内存缓存
        self.memory_cache[key] = value
        
        # 存入Redis缓存
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(value)
        )
    
    def delete(self, key: str):
        """删除缓存"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        self.redis_client.delete(key)
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        self.redis_client.flushdb()

def cache_result(ttl: int = 3600):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hashlib.md5(str(args) + str(kwargs).encode()).hexdigest()}"
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# 使用示例
cache_manager = MultiLevelCache()

@cache_result(ttl=1800)  # 30分钟缓存
def expensive_workflow_agent(state):
    """昂贵的Agent操作"""
    # 模拟复杂计算
    import time
    time.sleep(2)
    
    return {"result": "expensive_computation_result"}
```

### 2. 异步优化

#### 异步Agent执行
```python
# async_agent.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

class AsyncAgentExecutor:
    """异步Agent执行器"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def execute_agent(self, agent_func, state: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行Agent"""
        async with self.semaphore:
            # 在线程池中执行同步Agent函数
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                agent_func,
                state
            )
            return result
    
    async def execute_parallel_agents(self, agents: list, state: Dict[str, Any]) -> list:
        """并行执行多个Agent"""
        tasks = []
        for agent_func in agents:
            task = self.execute_agent(agent_func, state)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# 使用示例
async def parallel_workflow_execution():
    """并行工作流执行"""
    executor = AsyncAgentExecutor(max_workers=4)
    
    # 定义多个Agent
    agents = [
        research_agent,
        analysis_agent,
        summary_agent,
        report_agent
    ]
    
    # 并行执行
    results = await executor.execute_parallel_agents(agents, initial_state)
    
    return results
```

## 4.8 部署检查清单

### 生产环境检查清单

```python
# deployment_checklist.py
import psycopg2
import redis
import requests
import time
from typing import Dict, List

class DeploymentChecklist:
    """部署检查清单"""
    
    def __init__(self):
        self.checks = []
    
    def check_database_connection(self) -> bool:
        """检查数据库连接"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="langgraph",
                user="langgraph_user",
                password="secure_password"
            )
            conn.close()
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def check_redis_connection(self) -> bool:
        """检查Redis连接"""
        try:
            r = redis.Redis(host="localhost", port=6379, db=0)
            r.ping()
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False
    
    def check_api_health(self) -> bool:
        """检查API健康状态"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"API health check failed: {e}")
            return False
    
    def check_workflow_execution(self) -> bool:
        """检查工作流执行"""
        try:
            test_data = {
                "workflow_type": "test",
                "input_data": {"message": "test"},
                "user_id": "test_user",
                "session_id": "test_session"
            }
            
            response = requests.post(
                "http://localhost:8000/workflow/execute",
                json=test_data,
                timeout=30
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Workflow execution test failed: {e}")
            return False
    
    def check_metrics_endpoint(self) -> bool:
        """检查指标端点"""
        try:
            response = requests.get("http://localhost:8000/metrics", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Metrics endpoint check failed: {e}")
            return False
    
    def run_all_checks(self) -> Dict[str, bool]:
        """运行所有检查"""
        checks = {
            "Database Connection": self.check_database_connection(),
            "Redis Connection": self.check_redis_connection(),
            "API Health": self.check_api_health(),
            "Workflow Execution": self.check_workflow_execution(),
            "Metrics Endpoint": self.check_metrics_endpoint()
        }
        
        # 打印检查结果
        print("\n=== Deployment Checklist ===")
        for check_name, result in checks.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check_name}: {status}")
        
        return checks

# 使用示例
if __name__ == "__main__":
    checklist = DeploymentChecklist()
    results = checklist.run_all_checks()
    
    # 如果有失败的检查，退出
    if not all(results.values()):
        print("\n❌ Some checks failed. Please fix issues before deployment.")
        exit(1)
    else:
        print("\n✅ All checks passed. Deployment ready!")
```

## 总结

第4部分详细介绍了LangGraph在生产环境中的部署和运维：

1. **生产环境架构**：API网关、负载均衡、监控系统
2. **容器化部署**：Docker、Docker Compose、Kubernetes
3. **监控与告警**：Prometheus、Grafana、结构化日志
4. **负载均衡与扩展**：水平扩展、会话管理、自动扩缩容
5. **安全配置**：JWT认证、速率限制、访问控制
6. **备份与恢复**：数据备份策略、恢复脚本、验证机制
7. **性能优化**：多级缓存、异步执行、资源管理
8. **部署检查清单**：完整的部署验证流程

这些内容为LangGraph应用的生产环境部署提供了完整的解决方案，确保系统的可靠性、可扩展性和可维护性。

通过这4个部分的教程，您应该已经掌握了使用LangGraph构建复杂多Agent系统的完整技能，从基础概念到生产部署的全流程。 