import os
from lmdeploy import pipeline

base_dir = os.environ.get("MERGED_MODEL", "./merged")
pipe = pipeline(base_dir)

prompt = "请用三句话解释什么是注意力机制。"
res = pipe([prompt], gen_config=dict(temperature=0.7, top_p=0.9, max_new_tokens=128))
print(res[0].text)
