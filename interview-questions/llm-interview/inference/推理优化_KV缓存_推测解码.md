# 推理优化：KV 缓存与推测解码（Speculative Decoding）

## 一、KV 缓存（Key/Value Cache）
- 目的：自回归生成时复用历史注意力键值，避免重复计算
- 关键：正确维护 past_key_values 的形状与位置编码

### 1) 伪代码
```python
def forward_step(model, input_ids, past_key_values=None):
    outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
    logits = outputs.logits[:, -1]
    return logits, outputs.past_key_values
```

### 2) 常见问题
- 长序列越界：旋转位置编码（RoPE）需同步偏移
- 多批次拼接：注意不同样本的缓存对齐

## 二、推测解码（Draft + Target）
- 思想：用小模型快速“草拟”多步token，由大模型一次性验证与接受
- 收益：减少大模型前向调用次数，显著提升吞吐

### 1) 流程
1. Draft 模型生成 k 个候选token
2. Target 模型在同一批次上验证这些token
3. 按接受-回退机制推进序列

### 2) 伪代码
```python
def speculative_decode(draft, target, prompt, k=4, max_new_tokens=128):
    seq = prompt
    past_t = past_d = None
    for _ in range(max_new_tokens):
        # 1) draft 预测多步
        d_tokens = []
        for _ in range(k):
            logits_d, past_d = draft(seq[:, -1:], past_d)
            next_d = logits_d.softmax(-1).multinomial(1)
            seq = torch.cat([seq, next_d], dim=-1)
            d_tokens.append(next_d)
        
        # 2) target 验证
        logits_t, past_t = target(seq[:, -k-1:], past_t)
        accept_prefix = compute_accept_prefix(d_tokens, logits_t)
        
        if accept_prefix == 0:
            # 回退，仅推进1步（按 target 分布采样）
            next_t = logits_t.softmax(-1).multinomial(1)
            seq = torch.cat([seq[:, :-k], next_t], dim=-1)
        else:
            # 接受前缀，保留余下未验证部分
            seq = torch.cat([seq[:, :-k], torch.cat(d_tokens[:accept_prefix], dim=1)], dim=-1)
    return seq
```

### 3) 实践要点
- Draft 选择：蒸馏/量化的小模型；与 Target 分布接近
- 同步位置编码、温度/Top-p；避免分布漂移
- 监控接受率与回退率，动态调整 k

## 三、并发与批处理
- 按相同 step 的请求分桶（prefix batching）
- 静态/动态批次合并（vLLM/LMDeploy 实现思路）

## 四、面试要点
- 解释 KV 缓存带来的复杂度变化：从 O(n^2) 降到增量 O(n)
- 推测解码的正确性与收益边界（草拟质量不足会拖慢）
- 实际系统如何实现“连续批处理 + KV 管理”
