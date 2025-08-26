### 12 pgvector 与向量检索

#### 安装与启用
```sql
create extension if not exists vector;
```

#### 表设计
```sql
create table docs (
  id bigserial primary key,
  content text,
  embedding vector(1536)
);
create index on docs using ivfflat (embedding vector_cosine_ops) with (lists = 100);
```

#### 相似度查询
```sql
select id, content from docs
order by embedding <#> '[0.1, 0.2, ...]'::vector
limit 5;
```

#### 实战建议
- 选择合适的维度与度量（cosine/L2/inner product）；
- 预先归一化向量；
- 索引参数（lists/probes）与召回-延迟权衡。


