# 对齐算法：DPO / ORPO / KTO 详解

## 一、问题背景
- RLHF 训练复杂、昂贵；偏好优化类方法直接用偏好对训练，无需显式奖励模型

## 二、DPO（Direct Preference Optimization）
- 目标：最大化 chosen 相对 rejected 的对数几率差，带 KL 正则
- 损失：
```math
L = - E_{(x,y^+,y^-)} [\log \sigma(\beta (\log \pi(y^+|x)-\log \pi(y^-|x)) - \log \pi_0(y^+|x)+\log \pi_0(y^-|x))]
```
- 关键：参考模型 \pi_0 冻结；β 控制强度

## 三、ORPO（Odds Ratio Preference Optimization）
- 直接优化胜算比，训练稳定，离线易用
- 面试点：为何 ORPO 在数据噪声下更稳？因其对比分数形式更鲁棒

## 四、KTO（Kahneman-Tversky Optimization）
- 借鉴前景理论，对好/坏样本施加不同权重，提升安全与偏好一致性

## 五、实现要点（伪代码）
```python
def dpo_loss(logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected, beta=0.1):
    # margin = beta*(logp_c - logp_r) - (logp_ref_c - logp_ref_r)
    margin = beta * (logp_chosen - logp_rejected) - (logp_ref_chosen - logp_ref_rejected)
    return -torch.log(torch.sigmoid(margin)).mean()
```

## 六、实战建议
- 数据：严格对齐格式；覆盖“安全与合规”负样本
- 冻结参考模型；小学习率 + 短轮数
- 评估：人评 + 任务集 + 安全评测（Jailbreak、Toxicity）
