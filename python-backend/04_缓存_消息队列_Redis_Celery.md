### 04 缓存与消息队列：Redis + Celery/RQ

#### Redis 缓存
```python
import aioredis
redis = await aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
await redis.setex("user:1", 300, "{...}")
```

#### 分布式锁（Lua简版）
```python
ok = await redis.set("lock:task", "1", ex=10, nx=True)
if ok:
    try:
        ... # do work
    finally:
        await redis.delete("lock:task")
```

#### Celery 任务队列
```python
from celery import Celery
celery_app = Celery("app", broker="redis://localhost:6379/1", backend="redis://localhost:6379/2")

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=2, max_retries=5)
def send_email(to: str):
    ...
```

#### 定时任务
- Celery beat / APScheduler；

#### 最佳实践
- 缓存雪崩/击穿：随机过期、互斥锁；
- 幂等：设置去重Key；
- 监控队列堆积与失败重试。


