# LangGraph 工具调用详解 - 第5部分：工具集成与调用

## 概述

第5部分将深入探讨LangGraph中的工具调用机制，包括静态工具调用、动态工具选择、工具链集成等。基于最新的LangGraph源码，详细分析工具系统的架构设计和实现方式。

## 5.1 工具系统架构

### 5.1.1 工具系统设计理念

LangGraph的工具系统基于以下设计理念：

1. **节点即工具**：工具被实现为工作流中的节点
2. **状态驱动**：工具调用通过状态变化驱动
3. **类型安全**：基于Pydantic的类型系统
4. **可组合性**：工具可以组合成复杂的工具链

### 5.1.2 工具系统核心组件

```python
# langgraph/libs/langgraph/langgraph/graph/state.py
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """状态图，支持工具节点集成"""
    
    def add_node(self, node, *, defer=False, metadata=None):
        """添加节点，支持工具集成"""
        if hasattr(node, 'tools'):
            # 节点包含工具
            self.nodes[name] = node
        else:
            # 普通节点
            self.nodes[name] = node
```

## 5.2 静态工具调用

### 5.2.1 基本工具定义

静态工具调用是指在工作流中预定义的工具，具有固定的功能和参数。

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import requests
import json
import re

# 定义工具上下文
@dataclass
class ToolContext:
    api_keys: Dict[str, str]
    tool_permissions: List[str]
    rate_limits: Dict[str, int]

# 定义工具状态
class ToolState(TypedDict):
    user_query: str
    tool_results: Annotated[List[dict], "results"]
    selected_tools: List[str]
    tool_errors: List[dict]
    final_response: str

# 基础工具定义
@task
def web_search_tool(query: str, api_key: str = None) -> dict:
    """网络搜索工具"""
    try:
        # 模拟网络搜索
        search_results = [
            {"title": f"搜索结果1: {query}", "url": "https://example1.com", "snippet": f"关于{query}的详细信息"},
            {"title": f"搜索结果2: {query}", "url": "https://example2.com", "snippet": f"{query}的相关内容"},
            {"title": f"搜索结果3: {query}", "url": "https://example3.com", "snippet": f"{query}的最新动态"}
        ]
        
        return {
            "tool_name": "web_search",
            "query": query,
            "results": search_results,
            "status": "success",
            "count": len(search_results)
        }
    except Exception as e:
        return {
            "tool_name": "web_search",
            "query": query,
            "error": str(e),
            "status": "error"
        }

@task
def calculator_tool(expression: str) -> dict:
    """计算器工具"""
    try:
        # 安全计算，只允许基本数学运算
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含不允许的字符")
        
        result = eval(expression)
        
        return {
            "tool_name": "calculator",
            "expression": expression,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "tool_name": "calculator",
            "expression": expression,
            "error": str(e),
            "status": "error"
        }

@task
def weather_tool(city: str, api_key: str = None) -> dict:
    """天气查询工具"""
    try:
        # 模拟天气API调用
        weather_data = {
            "city": city,
            "temperature": "25°C",
            "condition": "晴天",
            "humidity": "60%",
            "wind_speed": "10 km/h"
        }
        
        return {
            "tool_name": "weather",
            "city": city,
            "data": weather_data,
            "status": "success"
        }
    except Exception as e:
        return {
            "tool_name": "weather",
            "city": city,
            "error": str(e),
            "status": "error"
        }

@task
def translation_tool(text: str, source_lang: str, target_lang: str) -> dict:
    """翻译工具"""
    try:
        # 模拟翻译API调用
        translations = {
            ("你好", "zh", "en"): "Hello",
            ("Hello", "en", "zh"): "你好",
            ("世界", "zh", "en"): "World",
            ("World", "en", "zh"): "世界"
        }
        
        key = (text, source_lang, target_lang)
        translated_text = translations.get(key, f"[翻译: {text}]")
        
        return {
            "tool_name": "translation",
            "original": text,
            "translated": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "status": "success"
        }
    except Exception as e:
        return {
            "tool_name": "translation",
            "original": text,
            "error": str(e),
            "status": "error"
        }
```

### 5.2.2 工具注册和管理

```python
class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools = {}
        self.tool_metadata = {}
    
    def register_tool(self, name: str, tool_func, metadata: dict = None):
        """注册工具"""
        self.tools[name] = tool_func
        self.tool_metadata[name] = metadata or {}
    
    def get_tool(self, name: str):
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self.tools.keys())
    
    def get_tool_metadata(self, name: str) -> dict:
        """获取工具元数据"""
        return self.tool_metadata.get(name, {})

# 创建工具注册表
tool_registry = ToolRegistry()

# 注册工具
tool_registry.register_tool("web_search", web_search_tool, {
    "description": "网络搜索工具",
    "parameters": ["query"],
    "rate_limit": 10
})

tool_registry.register_tool("calculator", calculator_tool, {
    "description": "数学计算工具",
    "parameters": ["expression"],
    "rate_limit": 100
})

tool_registry.register_tool("weather", weather_tool, {
    "description": "天气查询工具",
    "parameters": ["city"],
    "rate_limit": 50
})

tool_registry.register_tool("translation", translation_tool, {
    "description": "翻译工具",
    "parameters": ["text", "source_lang", "target_lang"],
    "rate_limit": 30
})
```

### 5.2.3 静态工具调用工作流

```python
@entrypoint(checkpointer=InMemorySaver())
def static_tool_workflow(
    user_query: str,
    context: ToolContext
) -> dict:
    """静态工具调用工作流"""
    
    # 1. 分析用户查询，确定需要的工具
    required_tools = []
    
    if any(word in user_query.lower() for word in ["搜索", "查找", "查询"]):
        required_tools.append("web_search")
    
    if any(word in user_query.lower() for word in ["计算", "算", "数学"]):
        required_tools.append("calculator")
    
    if any(word in user_query.lower() for word in ["天气", "温度", "气候"]):
        required_tools.append("weather")
    
    if any(word in user_query.lower() for word in ["翻译", "英文", "中文"]):
        required_tools.append("translation")
    
    # 2. 执行工具调用
    tool_results = []
    tool_errors = []
    
    for tool_name in required_tools:
        tool_func = tool_registry.get_tool(tool_name)
        if tool_func:
            try:
                # 根据工具类型提取参数
                if tool_name == "web_search":
                    # 提取搜索查询
                    query = user_query.replace("搜索", "").replace("查找", "").strip()
                    result = tool_func(query)
                elif tool_name == "calculator":
                    # 提取数学表达式
                    import re
                    expression_match = re.search(r'计算\s*(.+)', user_query)
                    if expression_match:
                        expression = expression_match.group(1)
                        result = tool_func(expression)
                    else:
                        continue
                elif tool_name == "weather":
                    # 提取城市名
                    import re
                    city_match = re.search(r'天气\s*(.+)', user_query)
                    if city_match:
                        city = city_match.group(1)
                        result = tool_func(city)
                    else:
                        continue
                elif tool_name == "translation":
                    # 提取翻译文本
                    import re
                    text_match = re.search(r'翻译\s*(.+)', user_query)
                    if text_match:
                        text = text_match.group(1)
                        result = tool_func(text, "zh", "en")
                    else:
                        continue
                
                tool_results.append(result)
                
            except Exception as e:
                tool_errors.append({
                    "tool_name": tool_name,
                    "error": str(e)
                })
    
    # 3. 生成最终响应
    if tool_results:
        response_parts = []
        for result in tool_results:
            if result["status"] == "success":
                if result["tool_name"] == "web_search":
                    response_parts.append(f"搜索结果：{result['results'][0]['snippet']}")
                elif result["tool_name"] == "calculator":
                    response_parts.append(f"计算结果：{result['result']}")
                elif result["tool_name"] == "weather":
                    weather_data = result["data"]
                    response_parts.append(f"{weather_data['city']}天气：{weather_data['temperature']}，{weather_data['condition']}")
                elif result["tool_name"] == "translation":
                    response_parts.append(f"翻译结果：{result['translated']}")
        
        final_response = "；".join(response_parts)
    else:
        final_response = "抱歉，我无法理解您的请求。"
    
    return {
        "user_query": user_query,
        "selected_tools": required_tools,
        "tool_results": tool_results,
        "tool_errors": tool_errors,
        "final_response": final_response
    }

# 使用示例
def run_static_tools():
    """运行静态工具调用"""
    context = ToolContext(
        api_keys={"weather": "your_api_key"},
        tool_permissions=["web_search", "calculator", "weather", "translation"],
        rate_limits={"web_search": 10, "calculator": 100, "weather": 50, "translation": 30}
    )
    
    config = {"configurable": {"thread_id": "user_123"}}
    
    # 测试不同的工具调用
    queries = [
        "搜索人工智能的最新发展",
        "计算 15 * 8 + 23",
        "查询北京的天气",
        "翻译 你好世界"
    ]
    
    for query in queries:
        result = static_tool_workflow.invoke(query, config=config)
        print(f"查询: {query}")
        print(f"结果: {result['final_response']}")
        print("---")
```

## 5.3 动态工具调用

### 5.3.1 动态工具选择机制

动态工具调用是指根据用户查询的内容和上下文，动态选择合适的工具进行调用。

```python
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing import TypedDict, Annotated, List, Dict, Any, Literal

# 定义动态工具上下文
@dataclass
class DynamicToolContext:
    available_tools: List[str]
    tool_capabilities: Dict[str, List[str]]
    user_preferences: Dict[str, Any]
    context_history: List[dict]

# 定义动态工具状态
class DynamicToolState(TypedDict):
    user_query: str
    query_intent: str
    candidate_tools: List[str]
    selected_tools: List[str]
    tool_execution_order: List[str]
    tool_results: Annotated[List[dict], "results"]
    confidence_scores: List[float]
    final_response: str

# 意图分析器
def intent_analyzer(state: DynamicToolState, runtime: Runtime[DynamicToolContext]) -> dict:
    """分析用户查询意图"""
    query = state.get("user_query", "")
    
    # 意图分类
    intents = {
        "information_search": ["搜索", "查找", "查询", "了解", "知道"],
        "calculation": ["计算", "算", "数学", "等于", "结果"],
        "weather_query": ["天气", "温度", "气候", "下雨", "晴天"],
        "translation": ["翻译", "英文", "中文", "语言"],
        "data_analysis": ["分析", "统计", "数据", "图表"],
        "code_generation": ["代码", "编程", "函数", "算法"],
        "image_processing": ["图片", "图像", "照片", "处理"],
        "file_operation": ["文件", "保存", "读取", "下载"]
    }
    
    detected_intent = "general"
    confidence = 0.0
    
    for intent, keywords in intents.items():
        matches = sum(1 for keyword in keywords if keyword in query)
        if matches > 0:
            detected_intent = intent
            confidence = min(matches / len(keywords), 1.0)
            break
    
    return {
        "query_intent": detected_intent,
        "confidence": confidence
    }

# 工具选择器
def tool_selector(state: DynamicToolState, runtime: Runtime[DynamicToolContext]) -> dict:
    """动态选择工具"""
    intent = state.get("query_intent", "")
    available_tools = runtime.context.available_tools
    
    # 意图到工具的映射
    intent_tool_mapping = {
        "information_search": ["web_search", "knowledge_base"],
        "calculation": ["calculator", "math_solver"],
        "weather_query": ["weather", "climate"],
        "translation": ["translation", "language_detector"],
        "data_analysis": ["data_analyzer", "chart_generator"],
        "code_generation": ["code_generator", "code_reviewer"],
        "image_processing": ["image_processor", "image_analyzer"],
        "file_operation": ["file_handler", "file_converter"]
    }
    
    # 选择候选工具
    candidate_tools = intent_tool_mapping.get(intent, [])
    
    # 过滤可用的工具
    available_candidates = [tool for tool in candidate_tools if tool in available_tools]
    
    # 如果没有找到合适的工具，使用通用工具
    if not available_candidates:
        available_candidates = ["general_assistant"]
    
    return {
        "candidate_tools": available_candidates,
        "selected_tools": available_candidates[:2]  # 最多选择2个工具
    }

# 工具执行器
def tool_executor(state: DynamicToolState, runtime: Runtime[DynamicToolContext]) -> dict:
    """执行选定的工具"""
    selected_tools = state.get("selected_tools", [])
    query = state.get("user_query", "")
    tool_results = []
    
    for tool_name in selected_tools:
        try:
            # 获取工具函数
            tool_func = tool_registry.get_tool(tool_name)
            if tool_func:
                # 根据工具类型执行
                if tool_name == "web_search":
                    result = tool_func(query)
                elif tool_name == "calculator":
                    import re
                    expression_match = re.search(r'计算\s*(.+)', query)
                    if expression_match:
                        result = tool_func(expression_match.group(1))
                    else:
                        continue
                elif tool_name == "weather":
                    import re
                    city_match = re.search(r'天气\s*(.+)', query)
                    if city_match:
                        result = tool_func(city_match.group(1))
                    else:
                        continue
                elif tool_name == "translation":
                    import re
                    text_match = re.search(r'翻译\s*(.+)', query)
                    if text_match:
                        result = tool_func(text_match.group(1), "zh", "en")
                    else:
                        continue
                else:
                    # 通用工具
                    result = {
                        "tool_name": tool_name,
                        "query": query,
                        "response": f"使用{tool_name}处理查询：{query}",
                        "status": "success"
                    }
                
                tool_results.append(result)
            
        except Exception as e:
            tool_results.append({
                "tool_name": tool_name,
                "error": str(e),
                "status": "error"
            })
    
    return {"tool_results": tool_results}

# 响应生成器
def response_generator(state: DynamicToolState, runtime: Runtime[DynamicToolContext]) -> dict:
    """生成最终响应"""
    tool_results = state.get("tool_results", [])
    query = state.get("user_query", "")
    
    if not tool_results:
        return {"final_response": "抱歉，我无法处理您的请求。"}
    
    # 合并工具结果
    response_parts = []
    for result in tool_results:
        if result["status"] == "success":
            if result["tool_name"] == "web_search":
                response_parts.append(f"搜索结果：{result.get('results', [{}])[0].get('snippet', '')}")
            elif result["tool_name"] == "calculator":
                response_parts.append(f"计算结果：{result.get('result', '')}")
            elif result["tool_name"] == "weather":
                weather_data = result.get("data", {})
                response_parts.append(f"{weather_data.get('city', '')}天气：{weather_data.get('temperature', '')}，{weather_data.get('condition', '')}")
            elif result["tool_name"] == "translation":
                response_parts.append(f"翻译结果：{result.get('translated', '')}")
            else:
                response_parts.append(result.get("response", ""))
    
    final_response = "；".join(response_parts) if response_parts else "处理完成"
    
    return {"final_response": final_response}

# 构建动态工具工作流
def create_dynamic_tool_workflow() -> StateGraph:
    """创建动态工具工作流"""
    
    workflow = StateGraph(
        state_schema=DynamicToolState,
        context_schema=DynamicToolContext
    )
    
    # 添加节点
    workflow.add_node("intent_analyzer", intent_analyzer)
    workflow.add_node("tool_selector", tool_selector)
    workflow.add_node("tool_executor", tool_executor)
    workflow.add_node("response_generator", response_generator)
    
    # 设置流程
    workflow.set_entry_point("intent_analyzer")
    workflow.add_edge("intent_analyzer", "tool_selector")
    workflow.add_edge("tool_selector", "tool_executor")
    workflow.add_edge("tool_executor", "response_generator")
    workflow.set_finish_point("response_generator")
    
    return workflow

# 使用示例
def run_dynamic_tools():
    """运行动态工具调用"""
    context = DynamicToolContext(
        available_tools=["web_search", "calculator", "weather", "translation", "general_assistant"],
        tool_capabilities={
            "web_search": ["信息检索", "知识查询"],
            "calculator": ["数学计算", "表达式求值"],
            "weather": ["天气查询", "气候信息"],
            "translation": ["语言翻译", "文本转换"],
            "general_assistant": ["通用对话", "问题回答"]
        },
        user_preferences={"language": "zh", "response_style": "detailed"},
        context_history=[]
    )
    
    workflow = create_dynamic_tool_workflow()
    app = workflow.compile()
    
    # 测试动态工具选择
    test_queries = [
        "搜索机器学习的最新进展",
        "计算 25 * 16 / 4 + 10",
        "查询上海的天气情况",
        "翻译 人工智能技术"
    ]
    
    for query in test_queries:
        result = app.invoke(
            {"user_query": query},
            context=context
        )
        print(f"查询: {query}")
        print(f"意图: {result.get('query_intent', '')}")
        print(f"选择工具: {result.get('selected_tools', [])}")
        print(f"响应: {result.get('final_response', '')}")
        print("---")
```

## 5.4 工具链集成

### 5.4.1 复杂工具链设计

工具链是指将多个工具组合成一个复杂的工作流程，实现更高级的功能。

```python
from langgraph.func import entrypoint, task
from typing import TypedDict, Annotated, List, Dict, Any, Literal

# 定义工具链上下文
@dataclass
class ToolChainContext:
    chain_config: Dict[str, Any]
    execution_mode: Literal["sequential", "parallel", "conditional"]
    error_handling: str
    timeout_settings: Dict[str, int]

# 定义工具链状态
class ToolChainState(TypedDict):
    input_data: str
    chain_steps: List[str]
    step_results: Annotated[List[dict], "results"]
    intermediate_data: Dict[str, Any]
    chain_status: Literal["running", "completed", "failed", "paused"]
    final_output: str
    error_log: List[dict]

# 工具链步骤定义
@task
def data_preprocessor(input_data: str) -> dict:
    """数据预处理工具"""
    try:
        # 清理和标准化输入数据
        cleaned_data = input_data.strip().lower()
        
        # 提取关键信息
        import re
        keywords = re.findall(r'\b\w+\b', cleaned_data)
        
        return {
            "step_name": "data_preprocessor",
            "input": input_data,
            "output": {
                "cleaned_data": cleaned_data,
                "keywords": keywords,
                "word_count": len(keywords)
            },
            "status": "success"
        }
    except Exception as e:
        return {
            "step_name": "data_preprocessor",
            "error": str(e),
            "status": "error"
        }

@task
def information_gatherer(keywords: List[str]) -> dict:
    """信息收集工具"""
    try:
        # 模拟信息收集
        gathered_info = []
        for keyword in keywords[:3]:  # 限制关键词数量
            info = {
                "keyword": keyword,
                "sources": [f"source1_{keyword}", f"source2_{keyword}"],
                "summary": f"关于{keyword}的详细信息"
            }
            gathered_info.append(info)
        
        return {
            "step_name": "information_gatherer",
            "input": keywords,
            "output": {
                "gathered_info": gathered_info,
                "total_sources": len(gathered_info) * 2
            },
            "status": "success"
        }
    except Exception as e:
        return {
            "step_name": "information_gatherer",
            "error": str(e),
            "status": "error"
        }

@task
def content_analyzer(gathered_info: List[dict]) -> dict:
    """内容分析工具"""
    try:
        # 分析收集到的信息
        analysis_results = []
        for info in gathered_info:
            analysis = {
                "keyword": info["keyword"],
                "sentiment": "positive",  # 模拟情感分析
                "relevance_score": 0.85,  # 模拟相关性评分
                "key_points": [f"要点1: {info['keyword']}", f"要点2: {info['keyword']}相关"]
            }
            analysis_results.append(analysis)
        
        return {
            "step_name": "content_analyzer",
            "input": gathered_info,
            "output": {
                "analysis_results": analysis_results,
                "average_sentiment": "positive",
                "average_relevance": 0.85
            },
            "status": "success"
        }
    except Exception as e:
        return {
            "step_name": "content_analyzer",
            "error": str(e),
            "status": "error"
        }

@task
def report_generator(analysis_results: List[dict], original_input: str) -> dict:
    """报告生成工具"""
    try:
        # 生成综合报告
        report_sections = []
        
        # 摘要部分
        report_sections.append(f"基于输入'{original_input}'的分析报告")
        
        # 详细分析
        for analysis in analysis_results:
            section = f"""
            {analysis['keyword']}分析:
            - 情感倾向: {analysis['sentiment']}
            - 相关性: {analysis['relevance_score']:.2f}
            - 关键要点: {', '.join(analysis['key_points'])}
            """
            report_sections.append(section)
        
        final_report = "\n".join(report_sections)
        
        return {
            "step_name": "report_generator",
            "input": analysis_results,
            "output": {
                "final_report": final_report,
                "section_count": len(report_sections),
                "word_count": len(final_report.split())
            },
            "status": "success"
        }
    except Exception as e:
        return {
            "step_name": "report_generator",
            "error": str(e),
            "status": "error"
        }

# 工具链工作流
@entrypoint(checkpointer=InMemorySaver())
def tool_chain_workflow(
    input_data: str,
    context: ToolChainContext
) -> dict:
    """工具链工作流"""
    
    # 1. 数据预处理
    preprocess_result = data_preprocessor(input_data)
    if preprocess_result["status"] == "error":
        return {
            "chain_status": "failed",
            "error_log": [preprocess_result],
            "final_output": "数据预处理失败"
        }
    
    intermediate_data = preprocess_result["output"]
    
    # 2. 信息收集
    gather_result = information_gatherer(intermediate_data["keywords"])
    if gather_result["status"] == "error":
        return {
            "chain_status": "failed",
            "error_log": [gather_result],
            "final_output": "信息收集失败"
        }
    
    intermediate_data.update(gather_result["output"])
    
    # 3. 内容分析
    analysis_result = content_analyzer(intermediate_data["gathered_info"])
    if analysis_result["status"] == "error":
        return {
            "chain_status": "failed",
            "error_log": [analysis_result],
            "final_output": "内容分析失败"
        }
    
    intermediate_data.update(analysis_result["output"])
    
    # 4. 报告生成
    report_result = report_generator(intermediate_data["analysis_results"], input_data)
    if report_result["status"] == "error":
        return {
            "chain_status": "failed",
            "error_log": [report_result],
            "final_output": "报告生成失败"
        }
    
    return {
        "chain_status": "completed",
        "input_data": input_data,
        "intermediate_data": intermediate_data,
        "final_output": report_result["output"]["final_report"],
        "step_results": [preprocess_result, gather_result, analysis_result, report_result],
        "error_log": []
    }

# 使用示例
def run_tool_chain():
    """运行工具链"""
    context = ToolChainContext(
        chain_config={
            "max_steps": 10,
            "retry_count": 3,
            "parallel_execution": False
        },
        execution_mode="sequential",
        error_handling="stop_on_error",
        timeout_settings={
            "preprocessor": 30,
            "gatherer": 60,
            "analyzer": 45,
            "generator": 30
        }
    )
    
    config = {"configurable": {"thread_id": "chain_123"}}
    
    # 测试工具链
    test_inputs = [
        "人工智能和机器学习的发展趋势",
        "区块链技术在金融领域的应用",
        "云计算和边缘计算的比较分析"
    ]
    
    for input_data in test_inputs:
        result = tool_chain_workflow.invoke(input_data, config=config)
        print(f"输入: {input_data}")
        print(f"状态: {result['chain_status']}")
        print(f"输出: {result['final_output'][:200]}...")
        print("---")
```

## 5.5 工具调用最佳实践

### 5.5.1 错误处理策略

```python
def robust_tool_executor(tool_name: str, tool_func, *args, **kwargs) -> dict:
    """健壮的工具执行器"""
    try:
        # 执行工具
        result = tool_func(*args, **kwargs)
        
        # 验证结果
        if not isinstance(result, dict):
            result = {"result": result}
        
        result["tool_name"] = tool_name
        result["status"] = "success"
        
        return result
        
    except Exception as e:
        # 错误处理
        return {
            "tool_name": tool_name,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

def retry_tool_executor(tool_name: str, tool_func, max_retries: int = 3, *args, **kwargs) -> dict:
    """带重试的工具执行器"""
    for attempt in range(max_retries):
        try:
            result = tool_func(*args, **kwargs)
            result["tool_name"] = tool_name
            result["status"] = "success"
            result["attempts"] = attempt + 1
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "tool_name": tool_name,
                    "status": "error",
                    "error": str(e),
                    "attempts": max_retries
                }
            else:
                import time
                time.sleep(1)  # 重试前等待
```

### 5.5.2 性能优化

```python
def cached_tool_executor(tool_name: str, tool_func, cache_key: str = None, *args, **kwargs) -> dict:
    """带缓存的工具执行器"""
    if cache_key is None:
        cache_key = f"{tool_name}_{hash(str(args) + str(kwargs))}"
    
    # 检查缓存
    cache = getattr(cached_tool_executor, '_cache', {})
    if cache_key in cache:
        result = cache[cache_key].copy()
        result["cached"] = True
        return result
    
    # 执行工具
    result = tool_func(*args, **kwargs)
    result["tool_name"] = tool_name
    result["status"] = "success"
    result["cached"] = False
    
    # 缓存结果
    if not hasattr(cached_tool_executor, '_cache'):
        cached_tool_executor._cache = {}
    cached_tool_executor._cache[cache_key] = result.copy()
    
    return result

def parallel_tool_executor(tools: List[tuple]) -> List[dict]:
    """并行工具执行器"""
    import concurrent.futures
    import asyncio
    
    async def execute_tool(tool_name: str, tool_func, *args, **kwargs):
        try:
            result = tool_func(*args, **kwargs)
            result["tool_name"] = tool_name
            result["status"] = "success"
            return result
        except Exception as e:
            return {
                "tool_name": tool_name,
                "status": "error",
                "error": str(e)
            }
    
    async def execute_all():
        tasks = []
        for tool_name, tool_func, args, kwargs in tools:
            task = execute_tool(tool_name, tool_func, *args, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    # 运行异步任务
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(execute_all())
    finally:
        loop.close()
    
    return results
```

### 5.5.3 监控和日志

```python
import logging
import time
from functools import wraps

def tool_monitor(tool_name: str):
    """工具监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = logging.getLogger(f"tool.{tool_name}")
            
            try:
                logger.info(f"开始执行工具: {tool_name}")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(f"工具执行成功: {tool_name}, 耗时: {execution_time:.2f}秒")
                
                # 添加监控信息
                if isinstance(result, dict):
                    result["execution_time"] = execution_time
                    result["tool_name"] = tool_name
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"工具执行失败: {tool_name}, 错误: {str(e)}, 耗时: {execution_time:.2f}秒")
                
                return {
                    "tool_name": tool_name,
                    "status": "error",
                    "error": str(e),
                    "execution_time": execution_time
                }
        
        return wrapper
    return decorator

# 使用监控装饰器
@tool_monitor("web_search")
def monitored_web_search(query: str) -> dict:
    """带监控的网络搜索工具"""
    return web_search_tool(query)

@tool_monitor("calculator")
def monitored_calculator(expression: str) -> dict:
    """带监控的计算器工具"""
    return calculator_tool(expression)
```

## 5.6 总结

### 5.6.1 工具调用模式总结

1. **静态工具调用**
   - 预定义工具集
   - 固定的工具选择逻辑
   - 适合简单场景

2. **动态工具调用**
   - 基于意图的工具选择
   - 上下文感知的工具调用
   - 适合复杂场景

3. **工具链集成**
   - 多工具组合
   - 复杂工作流程
   - 适合高级应用

### 5.6.2 最佳实践要点

1. **错误处理**
   - 完善的异常捕获
   - 重试机制
   - 降级策略

2. **性能优化**
   - 缓存机制
   - 并行执行
   - 资源管理

3. **监控和日志**
   - 执行时间监控
   - 错误日志记录
   - 性能指标收集

4. **类型安全**
   - 基于Pydantic的类型定义
   - 输入验证
   - 输出格式化

通过这些工具调用机制，可以构建出高效、可靠、可扩展的LangGraph应用系统，满足各种复杂的业务需求。 