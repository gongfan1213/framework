# SFT（监督微调）全流程与实战

## 一、SFT 总览
- 目标：让预训练模型对齐特定任务/风格
- 核心环节：数据→模板→分词→训练配置→验证→导出→评估
- 常见陷阱：数据泄漏、过拟合、样式漂移、毒性放大

## 二、数据与模板
- 数据格式：instruction/input/output 或 role-based 多轮对话
- 模板示例（ChatML）：
```text
<|system|> 你是资深助理
<|user|> {instruction}\n{input}
<|assistant|>
```
- 面试问：如何避免泄漏？
  - 划分严格（时间切分/域切分），加入重复检验与反作弊去重（SimHash/MinHash）

## 三、Tokenizer 与长度策略
- 统一分词器与特殊标记；统计长度分布，设定 max_length、packing（多样本打包）
- 面试问：为何需要packing？
  - 提升吞吐，减少padding浪费；需注意loss对齐与label_mask

## 四、损失定义与label mask
```python
# 假设labels中padding位置为-100，仅计算有效token损失
loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
```
- 指令场景仅计算 assistant 段落的label，user/system 段落mask为-100

## 五、训练配置（单机/多卡）
- 优化器：AdamW(betas=(0.9,0.95))；lr: 1e-5~5e-5；warmup_ratio: 0.03；cosine decay
- 正则：label_smoothing、gradient_clip、weight_decay(0.1)
- AMP：bf16 或 fp16；gradient_accumulation 控制有效batch
- 面试问：如何选 lr？
  - LR range test；监控 val ppl 与样式一致性

## 六、评估与早停
- 自动评估：任务指标（EM/F1/ROUGE/BLEU）、安全指标（toxicity）、风格对齐（embedding 相似度）
- 早停：监控 val loss/metric、设置 patience

## 七、导出与推理
- 保存 adapter 或全量权重；推理时构造与训练一致的模板
- 冷启动评测：金标集 + 人工打分 + A/B

## 八、面试高频追问
- 数据倾斜如何处理？采样重加权、温度混合、困难样本挖掘
- SFT 如何与 RLHF/DPO 协同？SFT 稳定语义，RL 对齐偏好
- 如何控制“幻觉”？检索增强、工具强约束、答案结构化
