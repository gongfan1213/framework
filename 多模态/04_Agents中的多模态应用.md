### 04 多模态在 Agents 开发中的应用

#### 典型能力
- 视觉感知：让Agent“看得见”，用于UI自动化、机器人抓取、图表理解。
- 视觉定位与操作：根据截图定位元素并调用鼠标/键盘/API完成任务。
- 文档理解：图像+文本联合抽取（发票/工单/表格）。
- 工具编排：截图->OCR->关键信息->搜索/数据库->汇总。

#### 架构模式
- 感知-计划-执行（Perceive-Plan-Act）：
  1) 感知：VLM对截图/照片/视频帧解释；
  2) 计划：LLM形成多步计划；
  3) 执行：工具调用/控制端（RPA/浏览器/API）。

#### 提示工程示例（Python伪代码）
```python
system = "你是多模态智能体，能够理解截图并完成网页表格填写。"
user = {
  "image": "<base64_image>",
  "instruction": "读取发票金额并填入系统的金额输入框。"
}
# Agent 步骤：VLM识别金额 -> 定位输入框坐标 -> 浏览器自动化API填入 -> 校验截图
```

#### 工具链
- 截图/录屏：Selenium/Playwright + 屏幕捕捉；
- OCR/检测：PaddleOCR/PP-Structure、GroundingDINO/YOLO；
- 编排：LangGraph/LangChain/RPA脚本；
- 评测：任务成功率、耗时、人工对比。


