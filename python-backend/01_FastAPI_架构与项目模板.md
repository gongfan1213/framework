### 01 FastAPI 架构与项目模板

#### 目录结构（建议）
```text
app/
  core/           # 配置、安全、中间件、依赖注入
  api/            # 路由分层（v1/v2）
    v1/
      endpoints/
        users.py
        items.py
  models/         # ORM模型
  schemas/        # Pydantic模型
  services/       # 业务服务层
  repositories/   # 数据访问层
  workers/        # 异步任务（Celery/RQ）
  main.py         # 应用入口
migrations/       # Alembic
tests/
```

#### 最小应用
```python
# app/main.py
from fastapi import FastAPI

app = FastAPI(title="MyAPI")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

#### 路由分层与依赖注入
```python
# app/api/v1/endpoints/items.py
from fastapi import APIRouter, Depends
from app.core.deps import get_db

router = APIRouter()

@router.get("/{item_id}")
async def get_item(item_id: int, db=Depends(get_db)):
    return {"id": item_id}

# app/api/v1/__init__.py
from fastapi import APIRouter
from .endpoints import items

api_router = APIRouter()
api_router.include_router(items.router, prefix="/items", tags=["items"])

# app/main.py
from fastapi import FastAPI
from app.api.v1 import api_router

app = FastAPI()
app.include_router(api_router, prefix="/api/v1")
```

#### 中间件与CORS、GZip
```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

#### Uvicorn/Gunicorn 启动
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
gunicorn -k uvicorn.workers.UvicornWorker -w 4 app.main:app
```


