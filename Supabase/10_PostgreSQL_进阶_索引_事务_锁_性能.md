### 10 PostgreSQL 进阶：索引/事务/锁/性能

#### 索引
- B-Tree（默认，等值/范围）、GIN（JSONB/全文检索）、GiST（地理/相似度）、BRIN（超大顺序表）。
```sql
create index idx_todos_user on todos(user_id);
create index idx_todos_title_trgm on todos using gin (title gin_trgm_ops);
```

#### 事务与隔离级别
- READ COMMITTED / REPEATABLE READ / SERIALIZABLE；MVCC避免读写阻塞。

#### 锁
- 行级锁（FOR UPDATE）、表级锁（DDL）；尽量缩小锁范围与持锁时间。

#### 性能调优
- EXPLAIN/EXPLAIN ANALYZE 分析计划；
- 合理索引与避免函数在索引列上导致失效；
- 批量写入与分区；
- Autovacuum/Analyze 配置；
- 连接池（pgbouncer）与并发控制。


