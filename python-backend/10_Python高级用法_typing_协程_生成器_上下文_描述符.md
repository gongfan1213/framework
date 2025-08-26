### 10 Python 高级用法：typing、协程、生成器、上下文、描述符

#### typing 进阶
- 类型变量、泛型类、Protocol（结构化子类型）、TypedDict、Literal、Annotated；

#### 协程与生成器
- async/await、异步生成器（`async for`）、`yield from`；

#### 上下文管理器
```python
from contextlib import contextmanager

@contextmanager
def open_conn():
    conn = acquire()
    try:
        yield conn
    finally:
        release(conn)
```

#### 描述符与数据模型
- `__get__ / __set__ / __delete__` 控制属性访问；
- dataclass 与 `__post_init__`；


