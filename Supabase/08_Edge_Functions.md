### 08 Edge Functions

#### 概念
- 基于 Deno 的边缘函数，适合处理 Webhook、签名、服务端逻辑。

#### 开发与部署
```bash
supabase functions new hello
supabase functions serve
supabase functions deploy hello
```

#### 调用
```ts
const { data, error } = await supabase.functions.invoke('hello', { body: { name: 'world' } })
```


