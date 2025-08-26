### 05 认证（Auth）与 RLS（行级安全）

#### Auth 基础
- 支持邮箱/密码、Magic Link、OAuth（Google/GitHub等）、OTP/SMS、SAML；
- JWT 中会带有 `sub`（用户id）及自定义Claims；

#### RLS 概念
- 在表上启用 RLS 后，所有访问必须满足策略（Policy）；
- 常见策略：
  - 仅本人可读写：`auth.uid() = user_id`
  - 仅租户内可读：`auth.jwt() ->> 'tenant_id' = tenant_id`

#### 示例策略
```sql
alter table public.todos enable row level security;
create policy "user can read own" on public.todos for select using (auth.uid() = user_id);
create policy "user can write own" on public.todos for insert with check (auth.uid() = user_id);
```

#### 最佳实践
- 先写无RLS版业务逻辑，再逐步收紧策略；
- 使用服务端密钥（service_role）做管理任务，避免走RLS；
- 对策略做单元测试（SQL+SDK）。


