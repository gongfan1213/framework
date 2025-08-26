### 01 Supabase 概览与架构

#### Supabase 是什么
- 基于 PostgreSQL 的开源后端即服务（BaaS）：数据库、认证、存储、实时、Edge Functions、仪表盘与SDK一体化。

#### 架构组件
- Database（PostgreSQL）：核心存储，支持扩展（pgvector/ltree等）。
- Auth：基于GoTrue，支持邮箱、OAuth、OTP、SAML；与RLS深度结合。
- Storage：对象存储（S3兼容API风格），基于Postgres元数据管理。
- Realtime：基于逻辑复制/广播通道，将数据库变更推到客户端。
- Edge Functions：Deno/TypeScript运行时的无服务器函数。
- Studio：Web控制台（表结构、RLS、策略、日志、SQL Editor）。

#### 典型应用模式
- 全栈：前端直连 SDK（RLS 保证安全），或通过自建服务中转；
- LLM 应用：配合 pgvector 存向量索引，Realtime 用于消息/协作；
- 多租户：RLS + JWT Claims，实现租户级隔离与额度控制。


