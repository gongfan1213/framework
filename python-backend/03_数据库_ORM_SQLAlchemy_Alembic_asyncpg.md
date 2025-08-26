### 03 数据库与ORM：SQLAlchemy + Alembic + asyncpg

#### 选择
- 同步：psycopg2 + SQLAlchemy ORM（经典可靠）；
- 异步：asyncpg + SQLAlchemy 2.0 Async（高并发I/O更友好）。

#### 异步SQLAlchemy快速示例
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/app"
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

#### Alembic 迁移
```bash
alembic init migrations
# 编辑 alembic.ini 的 sqlalchemy.url，或在 env.py 中从设置读取
alembic revision -m "create users"
alembic upgrade head
```

#### 事务与模式
- 使用 `async with session.begin():` 包裹写操作；
- 谨慎长事务；读已提交级别足够大多数Web场景。

#### 性能与连接池
- 连接池大小与应用并发相匹配；
- 慢SQL分析：EXPLAIN/索引优化；
- 批量写入合并事务，减少往返。


