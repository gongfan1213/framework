### 03 PTQ 方法：GPTQ / AWQ / SmoothQuant / RPTQ

#### GPTQ（Gradient Post-Training Quantization）
- 思路：近似最小化权重量化带来的输出误差，分块求解，效果稳健；
- 特点：常用于权重INT4/INT3；推理需对应内核；
- 要点：合适的块大小、分组尺寸，以及良好的校准样本。

#### AWQ（Activation-aware Weight Quantization）
- 思路：考虑激活分布对权重量化的影响，保护重要通道；
- 特点：在视觉/语言模型上表现良好；
- 要点：通道重要性评估与门控策略。

#### SmoothQuant
- 思路：对激活和权重做平衡缩放，降低激活动态范围以便INT8量化；
- 特点：对激活INT8更友好，部署简单；
- 要点：平衡系数选择与层间一致性。

#### RPTQ（Round-to-nearest PTQ/Robust PTQ 等变体）
- 目标：提高低比特（INT4）稳定性，通过鲁棒目标与改进舍入策略降低误差。

#### 实战建议
- 7B/13B优先尝试AWQ/GPTQ；INT8激活配SmoothQuant；
- 先小规模评估不同方法在你的数据分布上的效果，再大规模量产。


