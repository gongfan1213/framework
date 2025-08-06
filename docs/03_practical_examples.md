# LangGraph 多Agents 搭建教程 - 第3部分：实际案例

## 概述

第3部分将通过实际案例，展示如何使用LangGraph构建复杂的多Agent系统。我们将从简单到复杂，逐步构建智能客服、研究助手、数据分析等实际应用。

## 3.1 智能客服系统

### 系统架构

构建一个多层次的智能客服系统，包含接待、分类、专家、总结等Agent。

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.channels import LastValue
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
import json

# 定义状态
class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: str
    issue_type: Literal["technical", "billing", "general", "urgent"]
    priority: Literal["low", "medium", "high", "critical"]
    assigned_agent: str
    resolution_status: Literal["pending", "in_progress", "resolved", "escalated"]
    customer_satisfaction: float
    conversation_summary: str
```

### Agent 实现

#### 1. 接待Agent (Greeter)
```python
def greeter_agent(state: CustomerServiceState) -> dict:
    """接待Agent：欢迎客户并收集基本信息"""
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    
    if not messages:
        # 初始欢迎
        welcome_msg = f"您好！我是智能客服助手，很高兴为您服务。您的客户ID是：{customer_id}。请告诉我您需要什么帮助？"
        return {
            "messages": [("assistant", welcome_msg)],
            "resolution_status": "pending"
        }
    
    # 分析客户问题
    last_message = messages[-1][1].lower()
    
    # 简单的问题分类
    if any(word in last_message for word in ["技术", "故障", "错误", "bug", "technical"]):
        issue_type = "technical"
    elif any(word in last_message for word in ["账单", "费用", "付款", "billing", "payment"]):
        issue_type = "billing"
    elif any(word in last_message for word in ["紧急", "urgent", "critical"]):
        issue_type = "urgent"
    else:
        issue_type = "general"
    
    # 优先级判断
    if issue_type == "urgent":
        priority = "critical"
    elif issue_type == "technical":
        priority = "high"
    elif issue_type == "billing":
        priority = "medium"
    else:
        priority = "low"
    
    return {
        "issue_type": issue_type,
        "priority": priority,
        "messages": [("assistant", f"我理解您的问题类型是：{issue_type}，优先级：{priority}。正在为您转接专业客服...")]
    }
```

#### 2. 分类Agent (Classifier)
```python
def classifier_agent(state: CustomerServiceState) -> dict:
    """分类Agent：根据问题类型和优先级分配专家"""
    issue_type = state.get("issue_type", "general")
    priority = state.get("priority", "low")
    
    # 根据问题类型和优先级分配专家
    if issue_type == "technical":
        if priority in ["high", "critical"]:
            assigned_agent = "senior_tech_support"
        else:
            assigned_agent = "tech_support"
    elif issue_type == "billing":
        assigned_agent = "billing_specialist"
    elif issue_type == "urgent":
        assigned_agent = "urgent_support"
    else:
        assigned_agent = "general_support"
    
    return {
        "assigned_agent": assigned_agent,
        "messages": [("assistant", f"已为您分配{assigned_agent}专家，请稍候...")]
    }
```

#### 3. 技术专家Agent (Tech Support)
```python
def tech_support_agent(state: CustomerServiceState) -> dict:
    """技术专家Agent：处理技术问题"""
    messages = state.get("messages", [])
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # 构建技术支持的上下文
    context = """
    你是技术专家，专门处理技术问题。请提供专业、准确的技术支持。
    如果问题无法在线解决，请提供升级建议。
    """
    
    # 调用LLM
    response = llm.invoke([
        ("system", context),
        *messages
    ])
    
    # 检查是否需要升级
    if "无法解决" in response.content or "需要升级" in response.content:
        resolution_status = "escalated"
        escalation_msg = "由于问题复杂，已为您升级到高级技术支持。"
    else:
        resolution_status = "in_progress"
        escalation_msg = ""
    
    return {
        "messages": [("assistant", response.content + escalation_msg)],
        "resolution_status": resolution_status
    }
```

#### 4. 账单专家Agent (Billing Specialist)
```python
def billing_specialist_agent(state: CustomerServiceState) -> dict:
    """账单专家Agent：处理账单问题"""
    messages = state.get("messages", [])
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    context = """
    你是账单专家，专门处理账单和付款问题。请提供准确的账单信息和建议。
    """
    
    response = llm.invoke([
        ("system", context),
        *messages
    ])
    
    return {
        "messages": [("assistant", response.content)],
        "resolution_status": "in_progress"
    }
```

#### 5. 总结Agent (Summarizer)
```python
def summarizer_agent(state: CustomerServiceState) -> dict:
    """总结Agent：生成对话总结和满意度评估"""
    messages = state.get("messages", [])
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # 生成对话总结
    summary_prompt = f"""
    请总结以下客服对话的关键信息：
    {messages}
    
    总结要点：
    1. 问题类型
    2. 解决方案
    3. 处理状态
    4. 客户满意度（1-10分）
    """
    
    summary_response = llm.invoke([("user", summary_prompt)])
    
    # 提取满意度评分
    import re
    satisfaction_match = re.search(r'满意度.*?(\d+)', summary_response.content)
    satisfaction = float(satisfaction_match.group(1)) if satisfaction_match else 7.0
    
    return {
        "conversation_summary": summary_response.content,
        "customer_satisfaction": satisfaction,
        "resolution_status": "resolved",
        "messages": [("assistant", "感谢您的咨询！对话已结束，如有其他问题请随时联系我们。")]
    }
```

### 工作流构建

```python
def build_customer_service_workflow():
    """构建智能客服工作流"""
    workflow = StateGraph(CustomerServiceState)
    
    # 添加节点
    workflow.add_node("greeter", greeter_agent)
    workflow.add_node("classifier", classifier_agent)
    workflow.add_node("tech_support", tech_support_agent)
    workflow.add_node("billing_specialist", billing_specialist_agent)
    workflow.add_node("urgent_support", urgent_support_agent)
    workflow.add_node("general_support", general_support_agent)
    workflow.add_node("summarizer", summarizer_agent)
    
    # 路由函数
    def route_to_expert(state: CustomerServiceState) -> str:
        assigned_agent = state.get("assigned_agent", "general_support")
        return assigned_agent
    
    def route_to_summary(state: CustomerServiceState) -> str:
        resolution_status = state.get("resolution_status", "pending")
        if resolution_status in ["resolved", "escalated"]:
            return "summarizer"
        else:
            return END
    
    # 添加边
    workflow.add_edge("greeter", "classifier")
    workflow.add_conditional_edges("classifier", route_to_expert)
    workflow.add_conditional_edges("tech_support", route_to_summary)
    workflow.add_conditional_edges("billing_specialist", route_to_summary)
    workflow.add_conditional_edges("urgent_support", route_to_summary)
    workflow.add_conditional_edges("general_support", route_to_summary)
    workflow.add_edge("summarizer", END)
    
    # 设置入口点
    workflow.set_entry_point("greeter")
    
    return workflow.compile()

# 使用示例
def run_customer_service():
    """运行客服系统"""
    app = build_customer_service_workflow()
    
    # 模拟客户对话
    initial_state = {
        "messages": [("user", "我的账户登录有问题，一直显示错误信息")],
        "customer_id": "CUST001"
    }
    
    result = app.invoke(initial_state)
    print("客服系统结果:", result)
```

## 3.2 研究助手系统

### 系统架构

构建一个多Agent研究系统，包含研究规划、数据收集、分析、总结等环节。

```python
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_topic: str
    research_plan: dict
    collected_data: list
    analysis_results: dict
    insights: list
    final_report: str
    research_phase: Literal["planning", "data_collection", "analysis", "synthesis", "reporting"]
    quality_score: float
```

### Agent 实现

#### 1. 研究规划Agent (Research Planner)
```python
def research_planner_agent(state: ResearchState) -> dict:
    """研究规划Agent：制定研究计划"""
    messages = state.get("messages", [])
    research_topic = state.get("research_topic", "")
    
    llm = ChatOpenAI(model="gpt-4")
    
    planning_prompt = f"""
    请为以下研究主题制定详细的研究计划：
    主题：{research_topic}
    
    请包含：
    1. 研究目标
    2. 研究方法
    3. 数据收集策略
    4. 分析框架
    5. 时间安排
    6. 预期成果
    """
    
    response = llm.invoke([("user", planning_prompt)])
    
    # 解析研究计划
    plan = {
        "objectives": ["目标1", "目标2"],
        "methods": ["方法1", "方法2"],
        "data_sources": ["来源1", "来源2"],
        "timeline": {"planning": "1天", "collection": "3天", "analysis": "2天"},
        "expected_outcomes": ["成果1", "成果2"]
    }
    
    return {
        "research_plan": plan,
        "research_phase": "data_collection",
        "messages": [("assistant", f"研究计划已制定：\n{response.content}")]
    }
```

#### 2. 数据收集Agent (Data Collector)
```python
def data_collector_agent(state: ResearchState) -> dict:
    """数据收集Agent：收集研究数据"""
    research_plan = state.get("research_plan", {})
    data_sources = research_plan.get("data_sources", [])
    
    # 模拟数据收集
    collected_data = []
    
    for source in data_sources:
        # 模拟从不同来源收集数据
        data = {
            "source": source,
            "content": f"从{source}收集的数据",
            "timestamp": "2024-01-01",
            "relevance_score": 0.8
        }
        collected_data.append(data)
    
    return {
        "collected_data": collected_data,
        "research_phase": "analysis",
        "messages": [("assistant", f"已收集{len(collected_data)}条数据，准备进行分析...")]
    }
```

#### 3. 数据分析Agent (Data Analyst)
```python
def data_analyst_agent(state: ResearchState) -> dict:
    """数据分析Agent：分析收集的数据"""
    collected_data = state.get("collected_data", [])
    
    llm = ChatOpenAI(model="gpt-4")
    
    # 构建分析提示
    analysis_prompt = f"""
    请分析以下研究数据：
    {collected_data}
    
    请提供：
    1. 数据质量评估
    2. 关键发现
    3. 趋势分析
    4. 统计摘要
    """
    
    response = llm.invoke([("user", analysis_prompt)])
    
    # 模拟分析结果
    analysis_results = {
        "data_quality": 0.85,
        "key_findings": ["发现1", "发现2", "发现3"],
        "trends": ["趋势1", "趋势2"],
        "statistics": {"sample_size": len(collected_data), "confidence": 0.95}
    }
    
    return {
        "analysis_results": analysis_results,
        "research_phase": "synthesis",
        "messages": [("assistant", f"数据分析完成：\n{response.content}")]
    }
```

#### 4. 洞察生成Agent (Insight Generator)
```python
def insight_generator_agent(state: ResearchState) -> dict:
    """洞察生成Agent：生成研究洞察"""
    analysis_results = state.get("analysis_results", {})
    research_plan = state.get("research_plan", {})
    
    llm = ChatOpenAI(model="gpt-4")
    
    insight_prompt = f"""
    基于以下分析结果，生成深入洞察：
    分析结果：{analysis_results}
    研究目标：{research_plan.get('objectives', [])}
    
    请提供：
    1. 主要洞察
    2. 实践建议
    3. 未来研究方向
    4. 局限性讨论
    """
    
    response = llm.invoke([("user", insight_prompt)])
    
    # 解析洞察
    insights = [
        "洞察1：数据趋势显示...",
        "洞察2：关键发现表明...",
        "洞察3：实践建议包括..."
    ]
    
    return {
        "insights": insights,
        "research_phase": "reporting",
        "messages": [("assistant", f"洞察生成完成：\n{response.content}")]
    }
```

#### 5. 报告生成Agent (Report Generator)
```python
def report_generator_agent(state: ResearchState) -> dict:
    """报告生成Agent：生成最终研究报告"""
    research_plan = state.get("research_plan", {})
    analysis_results = state.get("analysis_results", {})
    insights = state.get("insights", [])
    
    llm = ChatOpenAI(model="gpt-4")
    
    report_prompt = f"""
    请生成完整的研究报告，包含：
    
    研究计划：{research_plan}
    分析结果：{analysis_results}
    洞察：{insights}
    
    报告结构：
    1. 执行摘要
    2. 研究背景
    3. 研究方法
    4. 主要发现
    5. 洞察和建议
    6. 结论
    7. 附录
    """
    
    response = llm.invoke([("user", report_prompt)])
    
    return {
        "final_report": response.content,
        "research_phase": "completed",
        "quality_score": 0.9,
        "messages": [("assistant", "研究报告已生成完成！")]
    }
```

### 研究工作流构建

```python
def build_research_workflow():
    """构建研究工作流"""
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("planner", research_planner_agent)
    workflow.add_node("collector", data_collector_agent)
    workflow.add_node("analyst", data_analyst_agent)
    workflow.add_node("insight_generator", insight_generator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    
    # 顺序连接
    workflow.add_edge("planner", "collector")
    workflow.add_edge("collector", "analyst")
    workflow.add_edge("analyst", "insight_generator")
    workflow.add_edge("insight_generator", "report_generator")
    workflow.add_edge("report_generator", END)
    
    workflow.set_entry_point("planner")
    
    return workflow.compile()

# 使用示例
def run_research_assistant():
    """运行研究助手"""
    app = build_research_workflow()
    
    initial_state = {
        "messages": [("user", "请研究人工智能在医疗领域的应用")],
        "research_topic": "人工智能在医疗领域的应用"
    }
    
    result = app.invoke(initial_state)
    print("研究结果:", result)
```

## 3.3 数据分析系统

### 系统架构

构建一个多Agent数据分析系统，包含数据验证、统计分析、可视化、报告生成等环节。

```python
class DataAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    dataset: dict
    data_quality_report: dict
    statistical_analysis: dict
    visualizations: list
    insights: list
    final_report: str
    analysis_phase: Literal["validation", "exploration", "analysis", "visualization", "reporting"]
    data_quality_score: float
```

### Agent 实现

#### 1. 数据验证Agent (Data Validator)
```python
def data_validator_agent(state: DataAnalysisState) -> dict:
    """数据验证Agent：验证数据质量和完整性"""
    dataset = state.get("dataset", {})
    
    # 模拟数据验证
    validation_results = {
        "completeness": 0.95,
        "accuracy": 0.92,
        "consistency": 0.88,
        "timeliness": 0.90,
        "validity": 0.94
    }
    
    # 计算总体质量分数
    overall_score = sum(validation_results.values()) / len(validation_results)
    
    quality_report = {
        "overall_score": overall_score,
        "details": validation_results,
        "issues": [
            "缺失值：5%",
            "异常值：2%",
            "重复数据：1%"
        ],
        "recommendations": [
            "处理缺失值",
            "清理异常值",
            "去重处理"
        ]
    }
    
    return {
        "data_quality_report": quality_report,
        "data_quality_score": overall_score,
        "analysis_phase": "exploration",
        "messages": [("assistant", f"数据验证完成，质量分数：{overall_score:.2f}")]
    }
```

#### 2. 数据探索Agent (Data Explorer)
```python
def data_explorer_agent(state: DataAnalysisState) -> dict:
    """数据探索Agent：探索数据特征和模式"""
    dataset = state.get("dataset", {})
    quality_report = state.get("data_quality_report", {})
    
    llm = ChatOpenAI(model="gpt-4")
    
    exploration_prompt = f"""
    请分析以下数据集的特征：
    数据集：{dataset}
    质量报告：{quality_report}
    
    请提供：
    1. 数据概览
    2. 主要特征
    3. 数据分布
    4. 相关性分析
    5. 异常检测
    """
    
    response = llm.invoke([("user", exploration_prompt)])
    
    exploration_results = {
        "data_overview": "数据集包含1000条记录，10个变量",
        "key_features": ["特征1", "特征2", "特征3"],
        "distributions": {"正态分布": 3, "偏态分布": 2, "其他": 5},
        "correlations": {"强相关": 2, "中等相关": 3, "弱相关": 5},
        "anomalies": ["异常1", "异常2"]
    }
    
    return {
        "exploration_results": exploration_results,
        "analysis_phase": "analysis",
        "messages": [("assistant", f"数据探索完成：\n{response.content}")]
    }
```

#### 3. 统计分析Agent (Statistical Analyst)
```python
def statistical_analyst_agent(state: DataAnalysisState) -> dict:
    """统计分析Agent：执行统计分析"""
    dataset = state.get("dataset", {})
    exploration_results = state.get("exploration_results", {})
    
    llm = ChatOpenAI(model="gpt-4")
    
    analysis_prompt = f"""
    请对以下数据进行统计分析：
    数据集：{dataset}
    探索结果：{exploration_results}
    
    请执行：
    1. 描述性统计
    2. 假设检验
    3. 回归分析
    4. 聚类分析
    5. 时间序列分析（如果适用）
    """
    
    response = llm.invoke([("user", analysis_prompt)])
    
    statistical_results = {
        "descriptive_stats": {
            "mean": [10.5, 25.3, 15.7],
            "std": [2.1, 5.4, 3.2],
            "min": [5.0, 15.0, 8.0],
            "max": [18.0, 35.0, 25.0]
        },
        "hypothesis_tests": [
            {"test": "t-test", "p_value": 0.03, "significant": True},
            {"test": "chi-square", "p_value": 0.15, "significant": False}
        ],
        "regression_analysis": {
            "r_squared": 0.75,
            "coefficients": [0.5, 0.3, -0.2],
            "p_values": [0.001, 0.05, 0.1]
        },
        "clustering_results": {
            "n_clusters": 3,
            "silhouette_score": 0.65,
            "cluster_sizes": [300, 400, 300]
        }
    }
    
    return {
        "statistical_analysis": statistical_results,
        "analysis_phase": "visualization",
        "messages": [("assistant", f"统计分析完成：\n{response.content}")]
    }
```

#### 4. 可视化Agent (Visualization Agent)
```python
def visualization_agent(state: DataAnalysisState) -> dict:
    """可视化Agent：创建数据可视化"""
    statistical_analysis = state.get("statistical_analysis", {})
    exploration_results = state.get("exploration_results", {})
    
    # 模拟生成可视化
    visualizations = [
        {
            "type": "histogram",
            "title": "数据分布直方图",
            "description": "显示主要变量的分布情况",
            "file_path": "histogram.png"
        },
        {
            "type": "scatter_plot",
            "title": "相关性散点图",
            "description": "显示变量间的相关性",
            "file_path": "scatter.png"
        },
        {
            "type": "box_plot",
            "title": "箱线图",
            "description": "显示数据的中位数和异常值",
            "file_path": "boxplot.png"
        },
        {
            "type": "heatmap",
            "title": "相关性热力图",
            "description": "显示变量间的相关性强度",
            "file_path": "heatmap.png"
        }
    ]
    
    return {
        "visualizations": visualizations,
        "analysis_phase": "reporting",
        "messages": [("assistant", f"已生成{len(visualizations)}个可视化图表")]
    }
```

#### 5. 报告生成Agent (Report Generator)
```python
def analysis_report_generator_agent(state: DataAnalysisState) -> dict:
    """分析报告生成Agent：生成数据分析报告"""
    quality_report = state.get("data_quality_report", {})
    statistical_analysis = state.get("statistical_analysis", {})
    visualizations = state.get("visualizations", [])
    
    llm = ChatOpenAI(model="gpt-4")
    
    report_prompt = f"""
    请生成数据分析报告，包含：
    
    数据质量：{quality_report}
    统计分析：{statistical_analysis}
    可视化：{visualizations}
    
    报告结构：
    1. 执行摘要
    2. 数据概览
    3. 数据质量评估
    4. 探索性数据分析
    5. 统计分析结果
    6. 可视化解释
    7. 主要发现
    8. 建议和结论
    """
    
    response = llm.invoke([("user", report_prompt)])
    
    return {
        "final_report": response.content,
        "analysis_phase": "completed",
        "messages": [("assistant", "数据分析报告已生成完成！")]
    }
```

### 数据分析工作流构建

```python
def build_data_analysis_workflow():
    """构建数据分析工作流"""
    workflow = StateGraph(DataAnalysisState)
    
    # 添加节点
    workflow.add_node("validator", data_validator_agent)
    workflow.add_node("explorer", data_explorer_agent)
    workflow.add_node("analyst", statistical_analyst_agent)
    workflow.add_node("visualizer", visualization_agent)
    workflow.add_node("report_generator", analysis_report_generator_agent)
    
    # 顺序连接
    workflow.add_edge("validator", "explorer")
    workflow.add_edge("explorer", "analyst")
    workflow.add_edge("analyst", "visualizer")
    workflow.add_edge("visualizer", "report_generator")
    workflow.add_edge("report_generator", END)
    
    workflow.set_entry_point("validator")
    
    return workflow.compile()

# 使用示例
def run_data_analysis():
    """运行数据分析系统"""
    app = build_data_analysis_workflow()
    
    # 模拟数据集
    dataset = {
        "columns": ["age", "income", "education"],
        "data": [[25, 50000, "bachelor"], [30, 60000, "master"]],
        "shape": [1000, 3]
    }
    
    initial_state = {
        "messages": [("user", "请分析这个数据集")],
        "dataset": dataset
    }
    
    result = app.invoke(initial_state)
    print("分析结果:", result)
```

## 3.4 复杂系统集成

### 多系统协作

将客服、研究、分析系统集成到一个综合平台中。

```python
class IntegratedSystemState(TypedDict):
    messages: Annotated[list, add_messages]
    system_type: Literal["customer_service", "research", "data_analysis"]
    customer_service_state: dict
    research_state: dict
    analysis_state: dict
    final_output: str
```

### 系统路由Agent

```python
def system_router_agent(state: IntegratedSystemState) -> str:
    """系统路由Agent：根据用户需求路由到相应系统"""
    messages = state.get("messages", [])
    
    if not messages:
        return "greeter"
    
    last_message = messages[-1][1].lower()
    
    # 根据关键词判断系统类型
    if any(word in last_message for word in ["客服", "帮助", "问题", "支持"]):
        return "customer_service"
    elif any(word in last_message for word in ["研究", "分析", "调查", "报告"]):
        return "research"
    elif any(word in last_message for word in ["数据", "统计", "分析", "图表"]):
        return "data_analysis"
    else:
        return "general_assistant"
```

### 集成工作流

```python
def build_integrated_system():
    """构建集成系统"""
    workflow = StateGraph(IntegratedSystemState)
    
    # 添加子系统
    customer_service_app = build_customer_service_workflow()
    research_app = build_research_workflow()
    analysis_app = build_data_analysis_workflow()
    
    # 添加节点
    workflow.add_node("router", system_router_agent)
    workflow.add_node("customer_service", customer_service_app)
    workflow.add_node("research", research_app)
    workflow.add_node("analysis", analysis_app)
    workflow.add_node("coordinator", coordinator_agent)
    
    # 路由
    workflow.add_conditional_edges("router", lambda s: s.get("system_type", "general"))
    workflow.add_edge("customer_service", "coordinator")
    workflow.add_edge("research", "coordinator")
    workflow.add_edge("analysis", "coordinator")
    workflow.add_edge("coordinator", END)
    
    workflow.set_entry_point("router")
    
    return workflow.compile()
```

## 3.5 性能优化与监控

### 性能监控

```python
import time
import logging
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logging.info(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result
    return wrapper

# 应用性能监控
@performance_monitor
def monitored_agent(state):
    """带性能监控的Agent"""
    # Agent逻辑
    return {"messages": [("assistant", "处理完成")]}
```

### 错误恢复机制

```python
def robust_workflow():
    """健壮的工作流，包含错误恢复"""
    
    class RobustState(TypedDict):
        messages: Annotated[list, add_messages]
        status: str
        error_count: int
        retry_limit: int
    
    workflow = StateGraph(RobustState)
    
    def error_handler_agent(state: RobustState) -> dict:
        """错误处理Agent"""
        error_count = state.get("error_count", 0)
        retry_limit = state.get("retry_limit", 3)
        
        if error_count >= retry_limit:
            return {
                "status": "failed",
                "messages": [("assistant", "系统遇到问题，请联系管理员")]
            }
        else:
            return {
                "status": "retry",
                "error_count": error_count + 1,
                "messages": [("assistant", f"正在重试，第{error_count + 1}次")]
            }
    
    # 添加错误处理节点
    workflow.add_node("error_handler", error_handler_agent)
    
    return workflow
```

## 总结

第3部分通过实际案例展示了LangGraph的复杂应用：

1. **智能客服系统**：多层次Agent协作，动态路由，满意度评估
2. **研究助手系统**：完整的研究流程，从规划到报告生成
3. **数据分析系统**：数据验证、分析、可视化、报告生成
4. **系统集成**：多系统协作，统一路由和协调
5. **性能优化**：监控、错误恢复、健壮性设计

这些案例展示了LangGraph在实际应用中的强大能力，为构建复杂的多Agent系统提供了完整的参考。

在下一部分中，我们将深入探讨部署、监控、扩展等生产环境相关的话题。 