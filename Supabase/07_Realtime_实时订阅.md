### 07 Realtime 实时订阅

#### 原理
- 基于Postgres逻辑复制与WS通道；监听表/频道（broadcast）。

#### JS 订阅变更
```ts
const channel = supabase.channel('todos-ch')
  .on('postgres_changes', { event: '*', schema: 'public', table: 'todos' }, payload => {
    console.log('change', payload)
  })
  .subscribe()
```

#### 广播频道
```ts
supabase.channel('room1').send({ type: 'broadcast', event: 'msg', payload: { text: 'hi' } })
```


