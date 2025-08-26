### 04 SDK（Python）使用

#### 安装
```bash
pip install supabase==2.5.1
```

#### 初始化
```python
from supabase import create_client, Client
import os

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_ANON_KEY"]
supabase: Client = create_client(url, key)
```

#### CRUD
```python
res = supabase.table('todos').select('*').eq('user_id', 'u1').execute()
supabase.table('todos').insert({"title": "demo", "user_id": "u1"}).execute()
supabase.table('todos').update({"done": True}).eq('id', 1).execute()
supabase.table('todos').delete().eq('id', 1).execute()
```

#### 认证
```python
auth = supabase.auth.sign_in_with_password({"email": "a@b.com", "password": "xxxx"})
print(auth.user)
```


