### LangGraph + Chrome 多模态 Agent 实战

本示例展示如何使用 LangGraph 编排一个具备“看图-计划-执行”能力的多模态 Agent：
- 感知：使用 Playwright 控制 Chromium/Chrome 截图；
- OCR：PaddleOCR 提取页面关键信息；
- 规划与对话：调用 OpenAI 兼容接口（可对接本仓中的 vLLM 网关）；
- 执行：根据计划调用浏览器动作（点击、输入、跳转）。

#### 目录
```text
agents-langgraph-chrome/
  requirements.txt
  run.py
  graph.py
  prompts.py
  tools/
    browser.py
    ocr.py
```

#### 环境准备（conda）
```bash
conda create -n lg_chrome python=3.10 -y
conda activate lg_chrome
pip install -r requirements.txt
python -m playwright install chromium
```

若使用本仓的 vLLM 网关，确保已启动 compose（参见`推理层/15_一键部署_docker_compose_说明.md`）。

#### 运行
```bash
export OPENAI_BASE_URL=http://127.0.0.1:9000/v1
export OPENAI_API_KEY=test-key
python run.py --url https://example.com --goal "读取首页主标题并返回"
```

输出将打印：
- OCR 摘要
- LLM 规划步骤
- 执行结果与最终答案

#### 注意
- 首版使用 OCR 将截图转文本后再由 LLM 规划，避免不同VLM入参差异；
- 如需直接走多模态VLM（图像+文本），可在`tools/browser.py`中将截图base64传给你的VLM，并调整`graph.py`的感知节点。


