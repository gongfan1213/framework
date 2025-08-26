### 09 PostgreSQL 基础

#### 基本数据类型
- 数值/字符/时间/布尔/JSONB/数组/枚举/地理/UUID；

#### 基本操作
```sql
create table todos (
  id bigserial primary key,
  user_id uuid not null,
  title text not null,
  done boolean default false,
  created_at timestamptz default now()
);

insert into todos (user_id, title) values ('00000000-0000-0000-0000-000000000000', 'demo');
select * from todos where user_id = '00000000-0000-0000-0000-000000000000';
update todos set done = true where id = 1;
delete from todos where id = 1;
```

#### 视图与物化视图
```sql
create view v_open_todos as select * from todos where done = false;
create materialized view mv_stats as select user_id, count(*) c from todos group by 1;
refresh materialized view mv_stats;
```


