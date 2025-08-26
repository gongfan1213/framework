### 06 存储（Storage）

#### 概念
- Bucket（桶）+ Object（对象）；
- 访问策略可与RLS/Policy结合（存路径元数据到表并用RLS控制）。

#### JS 示例
```ts
const { data, error } = await supabase.storage.from('avatars').upload('u1/a.png', file)
const { data: pub } = supabase.storage.from('avatars').getPublicUrl('u1/a.png')
```

#### Python 示例
```python
supabase.storage.from_("avatars").upload("u1/a.png", open("a.png", "rb"))
url = supabase.storage.from_("avatars").get_public_url("u1/a.png")
```


