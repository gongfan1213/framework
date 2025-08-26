### 11 PostgreSQL 迁移/备份/工具链

#### 迁移
- SQL文件或迁移框架（Prisma/Migration工具/Hasura/Knex）；
- 版本化、可回滚、上线前在影子库验证。

#### 备份与恢复
- pg_dump/pg_restore；
- 逻辑复制/物理备份（WAL）；
- 定期校验恢复流程。

#### 常用工具
- psql、pgAdmin、DBeaver、Azure Data Studio、Supabase Studio；
- 连接池 pgbouncer；
- 监控：pg_stat_statements、pgBadger。


