### 03 SDK（JS/TS）使用

#### 安装
```bash
npm i @supabase/supabase-js
```

#### 初始化客户端
```ts
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)
```

#### 基本CRUD
```ts
// 查询
const { data, error } = await supabase
  .from('todos')
  .select('*')
  .eq('user_id', user.id)

// 插入
await supabase.from('todos').insert({ title: 'demo', user_id: user.id })

// 更新
await supabase.from('todos').update({ done: true }).eq('id', 1)

// 删除
await supabase.from('todos').delete().eq('id', 1)
```

#### 认证
```ts
const { data: { user }, error } = await supabase.auth.signInWithPassword({
  email: 'a@b.com',
  password: 'xxxx'
})
```

#### Realtime 订阅
```ts
const channel = supabase.channel('room1')
  .on('postgres_changes', { event: '*', schema: 'public', table: 'todos' }, payload => {
    console.log('change', payload)
  })
  .subscribe()
```


