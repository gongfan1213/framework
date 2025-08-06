# LangGraph 业务场景与设计模式 - 第3部分：实际应用

## 概述

第3部分将深入探讨LangGraph在实际业务场景中的应用，包括智能客服、研究助手、数据分析、内容创作等常见场景，以及对应的设计模式和解决方案。

## 3.1 智能客服系统

### 3.1.1 场景描述

智能客服系统需要处理用户的各类咨询，包括产品信息、技术支持、订单查询、投诉处理等。系统需要能够：
- 理解用户意图
- 路由到合适的处理Agent
- 提供准确的回答
- 处理复杂对话
- 支持人工介入

### 3.1.2 解决方案架构

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass

# 定义上下文
@dataclass
class CustomerContext:
    customer_id: str
    customer_level: Literal["basic", "premium", "vip"]
    language: str = "zh-CN"
    session_id: str = ""

# 定义状态
class CustomerServiceState(TypedDict):
    messages: Annotated[list, "messages"]
    intent: str
    category: Literal["product", "technical", "order", "complaint", "general"]
    priority: Literal["low", "medium", "high", "urgent"]
    assigned_agent: str
    resolution_status: Literal["pending", "in_progress", "resolved", "escalated"]
    customer_satisfaction: float
    escalation_reason: str

# 任务定义
@task
def intent_classifier(user_message: str) -> dict:
    """意图分类器"""
    # 使用NLP模型分类用户意图
    if any(word in user_message.lower() for word in ["产品", "功能", "介绍"]):
        return {"intent": "product_inquiry", "category": "product"}
    elif any(word in user_message.lower() for word in ["问题", "故障", "错误"]):
        return {"intent": "technical_support", "category": "technical"}
    elif any(word in user_message.lower() for word in ["订单", "购买", "支付"]):
        return {"intent": "order_inquiry", "category": "order"}
    elif any(word in user_message.lower() for word in ["投诉", "不满", "退款"]):
        return {"intent": "complaint", "category": "complaint"}
    else:
        return {"intent": "general_inquiry", "category": "general"}

@task
def priority_assessor(intent: str, customer_level: str) -> dict:
    """优先级评估器"""
    priority_map = {
        "complaint": "high",
        "technical_support": "medium",
        "order_inquiry": "medium",
        "product_inquiry": "low",
        "general_inquiry": "low"
    }
    
    base_priority = priority_map.get(intent, "low")
    
    # VIP客户提升优先级
    if customer_level == "vip":
        if base_priority == "low":
            base_priority = "medium"
        elif base_priority == "medium":
            base_priority = "high"
    
    return {"priority": base_priority}

@task
def agent_router(category: str, priority: str, customer_level: str) -> str:
    """Agent路由器"""
    if priority == "urgent" or customer_level == "vip":
        return "senior_agent"
    elif category == "technical":
        return "technical_agent"
    elif category == "product":
        return "product_agent"
    elif category == "order":
        return "order_agent"
    elif category == "complaint":
        return "senior_agent"
    else:
        return "general_agent"

@task
def product_agent(query: str) -> str:
    """产品咨询Agent"""
    # 查询产品数据库
    return f"产品信息：{query}的相关产品详情..."

@task
def technical_agent(issue: str) -> str:
    """技术支持Agent"""
    # 查询知识库
    return f"技术解决方案：{issue}的解决方法..."

@task
def order_agent(order_query: str) -> str:
    """订单查询Agent"""
    # 查询订单系统
    return f"订单信息：{order_query}的订单状态..."

@task
def senior_agent(complex_query: str) -> str:
    """高级Agent处理复杂问题"""
    return f"高级处理：{complex_query}的详细解决方案..."

@task
def satisfaction_evaluator(response: str, customer_level: str) -> float:
    """满意度评估器"""
    # 基于响应质量和客户等级评估满意度
    base_score = 0.7
    if customer_level == "vip":
        base_score += 0.1
    if len(response) > 100:
        base_score += 0.1
    return min(base_score, 1.0)

# 主工作流
@entrypoint(checkpointer=InMemorySaver())
def customer_service_workflow(
    user_message: str, 
    context: CustomerContext
) -> dict:
    """智能客服工作流"""
    
    # 1. 意图分类
    intent_result = intent_classifier(user_message)
    intent = intent_result["intent"]
    category = intent_result["category"]
    
    # 2. 优先级评估
    priority_result = priority_assessor(intent, context.customer_level)
    priority = priority_result["priority"]
    
    # 3. Agent路由
    agent_type = agent_router(category, priority, context.customer_level)
    
    # 4. 根据Agent类型处理
    if agent_type == "product_agent":
        response = product_agent(user_message)
    elif agent_type == "technical_agent":
        response = technical_agent(user_message)
    elif agent_type == "order_agent":
        response = order_agent(user_message)
    elif agent_type == "senior_agent":
        response = senior_agent(user_message)
    else:
        response = f"通用回复：{user_message}"
    
    # 5. 满意度评估
    satisfaction = satisfaction_evaluator(response, context.customer_level)
    
    # 6. 检查是否需要人工介入
    if satisfaction < 0.6 or priority == "urgent":
        human_intervention = interrupt({
            "reason": "低满意度或紧急问题",
            "customer_level": context.customer_level,
            "priority": priority,
            "response": response
        })
        response = f"{response}\n\n人工补充：{human_intervention}"
    
    return {
        "original_query": user_message,
        "intent": intent,
        "category": category,
        "priority": priority,
        "assigned_agent": agent_type,
        "response": response,
        "satisfaction": satisfaction,
        "customer_id": context.customer_id
    }

# 使用示例
def run_customer_service():
    """运行客服系统"""
    context = CustomerContext(
        customer_id="CUST_001",
        customer_level="premium",
        language="zh-CN"
    )
    
    config = {"configurable": {"thread_id": context.customer_id}}
    
    # 处理用户查询
    result = customer_service_workflow.invoke(
        "我想了解你们的新产品功能",
        config=config
    )
    
    print(f"处理结果：{result}")
    
    # 如果需要人工介入，可以恢复工作流
    # resume_command = Command(resume="这是人工补充的详细信息...")
    # result = customer_service_workflow.invoke(resume_command, config)
```

### 3.1.3 设计模式特点

1. **分层处理**：意图分类 → 优先级评估 → Agent路由 → 专业处理
2. **动态路由**：根据意图、优先级、客户等级动态选择Agent
3. **人工介入**：低满意度或紧急问题时自动升级到人工
4. **状态持久化**：支持会话恢复和上下文保持
5. **多级Agent**：不同专业程度的Agent处理不同复杂度的问题

## 3.2 研究助手系统

### 3.2.1 场景描述

研究助手系统需要帮助用户进行学术研究，包括：
- 文献检索和筛选
- 数据收集和分析
- 论文写作辅助
- 研究趋势分析
- 学术规范检查

### 3.2.2 解决方案架构

```python
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing import TypedDict, Annotated, List
import asyncio

# 定义研究上下文
@dataclass
class ResearchContext:
    researcher_id: str
    research_field: str
    academic_level: Literal["undergraduate", "graduate", "phd", "professor"]
    language_preference: str = "en"

# 定义研究状态
class ResearchState(TypedDict):
    research_topic: str
    research_question: str
    literature_review: Annotated[List[str], "literature"]
    data_sources: List[str]
    analysis_results: dict
    methodology: str
    findings: List[str]
    conclusions: str
    references: List[str]
    current_phase: Literal["planning", "literature", "data", "analysis", "writing", "review"]

# 研究节点定义
def research_planner(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """研究规划器"""
    topic = state.get("research_topic", "")
    field = runtime.context.research_field
    
    # 生成研究问题
    research_question = f"在{field}领域，{topic}的研究现状和发展趋势如何？"
    
    # 制定研究计划
    methodology = f"""
    研究方法：
    1. 文献综述：检索相关文献
    2. 数据收集：收集实证数据
    3. 分析研究：进行统计分析
    4. 结果总结：撰写研究报告
    """
    
    return {
        "research_question": research_question,
        "methodology": methodology,
        "current_phase": "literature"
    }

def literature_reviewer(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """文献综述Agent"""
    topic = state.get("research_topic", "")
    question = state.get("research_question", "")
    
    # 模拟文献检索
    literature = [
        f"文献1：{topic}的基础理论研究",
        f"文献2：{topic}的最新发展动态",
        f"文献3：{topic}的应用案例分析",
        f"文献4：{topic}的未来发展趋势"
    ]
    
    # 文献分析
    analysis = f"""
    文献综述结果：
    - 共检索到{len(literature)}篇相关文献
    - 涵盖理论基础、发展动态、应用案例等方面
    - 为后续研究提供了坚实的理论基础
    """
    
    return {
        "literature_review": literature,
        "current_phase": "data"
    }

def data_collector(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """数据收集Agent"""
    topic = state.get("research_topic", "")
    
    # 模拟数据收集
    data_sources = [
        f"数据库1：{topic}相关统计数据",
        f"数据库2：{topic}案例数据",
        f"调查数据：{topic}用户调研结果",
        f"实验数据：{topic}实验验证数据"
    ]
    
    return {
        "data_sources": data_sources,
        "current_phase": "analysis"
    }

def data_analyzer(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """数据分析Agent"""
    data_sources = state.get("data_sources", [])
    
    # 模拟数据分析
    analysis_results = {
        "statistical_analysis": "描述性统计和推断性统计结果",
        "trend_analysis": "时间序列分析和趋势预测",
        "correlation_analysis": "变量间相关性分析",
        "regression_analysis": "回归分析结果"
    }
    
    findings = [
        "发现1：主要趋势和模式",
        "发现2：关键影响因素",
        "发现3：异常现象解释",
        "发现4：预测结果"
    ]
    
    return {
        "analysis_results": analysis_results,
        "findings": findings,
        "current_phase": "writing"
    }

def report_writer(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """报告撰写Agent"""
    findings = state.get("findings", [])
    literature = state.get("literature_review", [])
    
    # 生成结论
    conclusions = f"""
    研究结论：
    1. 基于{len(literature)}篇文献和{len(findings)}个主要发现
    2. 验证了研究假设
    3. 提出了新的见解和建议
    4. 为后续研究指明了方向
    """
    
    # 生成参考文献
    references = [
        "作者1. (2023). 标题1. 期刊1.",
        "作者2. (2023). 标题2. 期刊2.",
        "作者3. (2023). 标题3. 期刊3."
    ]
    
    return {
        "conclusions": conclusions,
        "references": references,
        "current_phase": "review"
    }

def quality_reviewer(state: ResearchState, runtime: Runtime[ResearchContext]) -> dict:
    """质量审查Agent"""
    academic_level = runtime.context.academic_level
    
    # 根据学术等级进行质量检查
    quality_checks = {
        "undergraduate": ["格式规范", "基本逻辑"],
        "graduate": ["格式规范", "逻辑严谨", "方法合理"],
        "phd": ["格式规范", "逻辑严谨", "方法创新", "贡献明确"],
        "professor": ["格式规范", "逻辑严谨", "方法创新", "贡献重大", "影响深远"]
    }
    
    checks = quality_checks.get(academic_level, [])
    
    review_result = f"""
    质量审查结果（{academic_level}级别）：
    {chr(10).join(f"- {check}: 通过" for check in checks)}
    """
    
    return {
        "quality_review": review_result,
        "current_phase": "completed"
    }

# 构建研究助手工作流
def create_research_assistant() -> StateGraph:
    """创建研究助手工作流"""
    
    workflow = StateGraph(
        state_schema=ResearchState,
        context_schema=ResearchContext
    )
    
    # 添加节点
    workflow.add_node("planner", research_planner)
    workflow.add_node("literature_reviewer", literature_reviewer)
    workflow.add_node("data_collector", data_collector)
    workflow.add_node("data_analyzer", data_analyzer)
    workflow.add_node("report_writer", report_writer)
    workflow.add_node("quality_reviewer", quality_reviewer)
    
    # 设置流程
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "literature_reviewer")
    workflow.add_edge("literature_reviewer", "data_collector")
    workflow.add_edge("data_collector", "data_analyzer")
    workflow.add_edge("data_analyzer", "report_writer")
    workflow.add_edge("report_writer", "quality_reviewer")
    workflow.set_finish_point("quality_reviewer")
    
    return workflow

# 使用示例
def run_research_assistant():
    """运行研究助手"""
    context = ResearchContext(
        researcher_id="RES_001",
        research_field="人工智能",
        academic_level="phd"
    )
    
    workflow = create_research_assistant()
    app = workflow.compile()
    
    # 开始研究
    result = app.invoke(
        {"research_topic": "大语言模型在教育领域的应用"},
        context=context
    )
    
    print(f"研究结果：{result}")
```

### 3.2.3 设计模式特点

1. **阶段化处理**：规划 → 文献 → 数据 → 分析 → 写作 → 审查
2. **专业化Agent**：每个阶段由专门的Agent处理
3. **质量分级**：根据学术等级进行不同深度的质量检查
4. **上下文感知**：根据研究领域和学术等级调整处理策略
5. **可扩展架构**：易于添加新的研究阶段或Agent

## 3.3 内容创作系统

### 3.3.1 场景描述

内容创作系统需要协助用户创作各种类型的内容，包括：
- 文章写作
- 营销文案
- 技术文档
- 创意内容
- 多语言翻译

### 3.3.2 解决方案架构

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, Literal

# 定义创作上下文
@dataclass
class ContentContext:
    author_id: str
    content_type: Literal["article", "marketing", "technical", "creative"]
    target_audience: str
    tone: Literal["formal", "casual", "professional", "creative"]
    language: str = "zh-CN"

# 定义创作状态
class ContentCreationState(TypedDict):
    topic: str
    outline: List[str]
    content_sections: Annotated[List[str], "sections"]
    keywords: List[str]
    seo_optimization: dict
    readability_score: float
    grammar_check: dict
    final_content: str
    metadata: dict

# 任务定义
@task
def topic_analyzer(topic: str, content_type: str) -> dict:
    """主题分析器"""
    # 分析主题的复杂度和方向
    complexity = "medium"
    if len(topic.split()) > 10:
        complexity = "high"
    
    # 确定内容方向
    direction = {
        "article": "informative",
        "marketing": "persuasive", 
        "technical": "explanatory",
        "creative": "entertaining"
    }.get(content_type, "informative")
    
    return {
        "complexity": complexity,
        "direction": direction,
        "estimated_length": "1500-2000 words"
    }

@task
def outline_generator(topic: str, content_type: str, direction: str) -> List[str]:
    """大纲生成器"""
    if content_type == "article":
        return [
            "引言：背景介绍",
            "主体：核心内容",
            "分析：深入探讨", 
            "结论：总结观点"
        ]
    elif content_type == "marketing":
        return [
            "痛点：问题描述",
            "解决方案：产品介绍",
            "优势：核心卖点",
            "行动：购买引导"
        ]
    elif content_type == "technical":
        return [
            "概述：技术背景",
            "实现：技术细节",
            "示例：代码演示",
            "总结：最佳实践"
        ]
    else:  # creative
        return [
            "开头：吸引注意",
            "发展：情节推进",
            "高潮：关键转折",
            "结尾：情感共鸣"
        ]

@task
def content_writer(outline: List[str], topic: str, tone: str) -> str:
    """内容撰写器"""
    content = f"基于大纲'{outline}'，以{tone}的语气撰写关于'{topic}'的内容..."
    return content

@task
def seo_optimizer(content: str, keywords: List[str]) -> dict:
    """SEO优化器"""
    # 模拟SEO优化
    optimization = {
        "keyword_density": "2.5%",
        "readability_score": "85/100",
        "meta_description": f"关于{keywords[0]}的详细内容",
        "title_optimization": "已优化"
    }
    return optimization

@task
def grammar_checker(content: str) -> dict:
    """语法检查器"""
    # 模拟语法检查
    issues = []
    if "。" in content and "。" not in content.split("。")[-1]:
        issues.append("句号使用不当")
    
    return {
        "total_issues": len(issues),
        "issues": issues,
        "score": max(90 - len(issues) * 5, 60)
    }

@task
def content_enhancer(content: str, feedback: str = None) -> str:
    """内容增强器"""
    if feedback:
        enhanced = f"{content}\n\n根据反馈'{feedback}'进行改进..."
    else:
        enhanced = f"{content}\n\n自动优化：提升表达清晰度和逻辑性..."
    
    return enhanced

# 主创作工作流
@entrypoint(checkpointer=InMemorySaver())
def content_creation_workflow(
    topic: str,
    context: ContentContext
) -> dict:
    """内容创作工作流"""
    
    # 1. 主题分析
    analysis = topic_analyzer(topic, context.content_type)
    complexity = analysis["complexity"]
    direction = analysis["direction"]
    
    # 2. 生成大纲
    outline = outline_generator(topic, context.content_type, direction)
    
    # 3. 撰写内容
    content = content_writer(outline, topic, context.tone)
    
    # 4. SEO优化
    keywords = [topic.split()[0], topic.split()[-1]]  # 简单关键词提取
    seo_result = seo_optimizer(content, keywords)
    
    # 5. 语法检查
    grammar_result = grammar_checker(content)
    
    # 6. 质量评估和人工介入
    if grammar_result["score"] < 80:
        human_feedback = interrupt({
            "reason": "语法问题较多，需要人工审核",
            "grammar_score": grammar_result["score"],
            "issues": grammar_result["issues"]
        })
        
        # 根据反馈增强内容
        enhanced_content = content_enhancer(content, human_feedback)
    else:
        enhanced_content = content_enhancer(content)
    
    return {
        "topic": topic,
        "content_type": context.content_type,
        "outline": outline,
        "final_content": enhanced_content,
        "seo_optimization": seo_result,
        "grammar_check": grammar_result,
        "complexity": complexity,
        "direction": direction,
        "author_id": context.author_id
    }

# 使用示例
def run_content_creation():
    """运行内容创作系统"""
    context = ContentContext(
        author_id="AUTH_001",
        content_type="marketing",
        target_audience="年轻消费者",
        tone="casual"
    )
    
    config = {"configurable": {"thread_id": context.author_id}}
    
    # 创作内容
    result = content_creation_workflow.invoke(
        "智能手表产品推广文案",
        config=config
    )
    
    print(f"创作结果：{result}")
```

### 3.3.3 设计模式特点

1. **类型化处理**：根据内容类型采用不同的创作策略
2. **多阶段优化**：分析 → 大纲 → 写作 → 优化 → 检查
3. **质量保证**：SEO优化、语法检查、可读性评估
4. **人工介入**：质量不达标时自动请求人工审核
5. **个性化定制**：根据目标受众和语气调整内容风格

## 3.4 数据分析系统

### 3.4.1 场景描述

数据分析系统需要处理各种数据分析任务，包括：
- 数据清洗和预处理
- 探索性数据分析
- 统计建模
- 可视化生成
- 报告撰写

### 3.4.2 解决方案架构

```python
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing import TypedDict, Annotated, List, Dict, Any
import pandas as pd
import numpy as np

# 定义分析上下文
@dataclass
class AnalysisContext:
    analyst_id: str
    project_name: str
    data_source: str
    analysis_type: Literal["descriptive", "predictive", "prescriptive"]
    visualization_preference: Literal["charts", "tables", "both"]

# 定义分析状态
class DataAnalysisState(TypedDict):
    raw_data: Dict[str, Any]
    cleaned_data: Dict[str, Any]
    data_summary: dict
    exploratory_analysis: dict
    statistical_tests: List[dict]
    models: List[dict]
    visualizations: List[str]
    insights: List[str]
    recommendations: List[str]
    report: str
    current_step: Literal["data_loading", "cleaning", "exploration", "modeling", "visualization", "reporting"]

# 分析节点定义
def data_loader(state: DataAnalysisState, runtime: Runtime[AnalysisContext]) -> dict:
    """数据加载器"""
    data_source = runtime.context.data_source
    
    # 模拟数据加载
    if data_source == "csv":
        # 模拟CSV数据
        data = {
            "columns": ["id", "name", "age", "salary", "department"],
            "data": [
                [1, "Alice", 25, 50000, "Engineering"],
                [2, "Bob", 30, 60000, "Marketing"],
                [3, "Charlie", 35, 70000, "Sales"],
                [4, "Diana", 28, 55000, "Engineering"],
                [5, "Eve", 32, 65000, "Marketing"]
            ]
        }
    else:
        data = {"columns": [], "data": []}
    
    return {
        "raw_data": data,
        "current_step": "cleaning"
    }

def data_cleaner(state: DataAnalysisState, runtime: Runtime[AnalysisContext]) -> dict:
    """数据清洗器"""
    raw_data = state.get("raw_data", {})
    
    # 模拟数据清洗
    cleaned_data = raw_data.copy()
    
    # 处理缺失值
    if "data" in cleaned_data:
        for row in cleaned_data["data"]:
            for i, value in enumerate(row):
                if value is None or value == "":
                    row[i] = "Unknown"
    
    # 数据类型转换
    data_summary = {
        "total_rows": len(cleaned_data.get("data", [])),
        "total_columns": len(cleaned_data.get("columns", [])),
        "missing_values": 0,
        "data_types": {
            "id": "integer",
            "name": "string", 
            "age": "integer",
            "salary": "integer",
            "department": "string"
        }
    }
    
    return {
        "cleaned_data": cleaned_data,
        "data_summary": data_summary,
        "current_step": "exploration"
    }

def exploratory_analyzer(state: DataAnalysisState, runtime: Runtime[AnalysisContext]) -> dict:
    """探索性数据分析器"""
    cleaned_data = state.get("cleaned_data", {})
    
    # 模拟探索性分析
    analysis = {
        "descriptive_stats": {
            "age": {"mean": 30, "median": 30, "std": 3.5},
            "salary": {"mean": 60000, "median": 60000, "std": 7500}
        },
        "correlations": {
            "age_salary": 0.85,
            "department_distribution": {
                "Engineering": 0.4,
                "Marketing": 0.4,
                "Sales": 0.2
            }
        },
        "outliers": ["Eve"],  # 高薪员工
        "patterns": [
            "年龄与薪资呈正相关",
            "工程部门员工较多",
            "薪资分布相对均匀"
        ]
    }
    
    return {
        "exploratory_analysis": analysis,
        "current_step": "modeling"
    }

def model_builder(state: DataAnalysisState, runtime: Runtime[AnalysisContext]) -> dict:
    """模型构建器"""
    analysis_type = runtime.context.analysis_type
    exploratory_analysis = state.get("exploratory_analysis", {})
    
    models = []
    
    if analysis_type == "descriptive":
        models.append({
            "type": "descriptive",
            "name": "薪资分布模型",
            "accuracy": 0.95,
            "insights": "薪资分布符合正态分布"
        })
    elif analysis_type == "predictive":
        models.append({
            "type": "regression",
            "name": "薪资预测模型",
            "accuracy": 0.88,
            "features": ["age", "department"],
            "target": "salary"
        })
    elif analysis_type == "prescriptive":
        models.append({
            "type": "optimization",
            "name": "薪资优化模型",
            "accuracy": 0.92,
            "recommendations": [
                "提高工程部门薪资",
                "建立绩效奖金制度"
            ]
        })
    
    return {
        "models": models,
        "current_step": "visualization"
    }

def visualization_generator(state: DataAnalysisState, runtime: Runtime[AnalysisContext]) -> dict:
    """可视化生成器"""
    preference = runtime.context.visualization_preference
    exploratory_analysis = state.get("exploratory_analysis", {})
    
    visualizations = []
    
    if preference in ["charts", "both"]:
        visualizations.extend([
            "薪资分布直方图",
            "年龄与薪资散点图",
            "部门员工分布饼图"
        ])
    
    if preference in ["tables", "both"]:
        visualizations.extend([
            "描述性统计表",
            "相关性矩阵表",
            "模型性能对比表"
        ])
    
    return {
        "visualizations": visualizations,
        "current_step": "reporting"
    }

def report_generator(state: DataAnalysisState, runtime: Runtime[AnalysisContext]) -> dict:
    """报告生成器"""
    project_name = runtime.context.project_name
    analysis_type = runtime.context.analysis_type
    
    # 收集所有分析结果
    data_summary = state.get("data_summary", {})
    exploratory_analysis = state.get("exploratory_analysis", {})
    models = state.get("models", [])
    visualizations = state.get("visualizations", [])
    
    # 生成洞察
    insights = [
        "数据质量良好，缺失值较少",
        "年龄与薪资存在强相关性",
        "工程部门是主要部门",
        f"模型准确率达到{models[0]['accuracy'] if models else 0}"
    ]
    
    # 生成建议
    recommendations = [
        "继续收集更多数据以提高模型准确性",
        "考虑添加更多特征变量",
        "定期更新模型以保持准确性"
    ]
    
    # 生成报告
    report = f"""
    数据分析报告：{project_name}
    
    1. 数据概览
    - 总行数：{data_summary.get('total_rows', 0)}
    - 总列数：{data_summary.get('total_columns', 0)}
    
    2. 主要发现
    {chr(10).join(f"- {insight}" for insight in insights)}
    
    3. 模型结果
    {chr(10).join(f"- {model['name']}: 准确率{model['accuracy']}" for model in models)}
    
    4. 可视化
    {chr(10).join(f"- {viz}" for viz in visualizations)}
    
    5. 建议
    {chr(10).join(f"- {rec}" for rec in recommendations)}
    """
    
    return {
        "insights": insights,
        "recommendations": recommendations,
        "report": report,
        "current_step": "completed"
    }

# 构建数据分析工作流
def create_data_analysis_workflow() -> StateGraph:
    """创建数据分析工作流"""
    
    workflow = StateGraph(
        state_schema=DataAnalysisState,
        context_schema=AnalysisContext
    )
    
    # 添加节点
    workflow.add_node("data_loader", data_loader)
    workflow.add_node("data_cleaner", data_cleaner)
    workflow.add_node("exploratory_analyzer", exploratory_analyzer)
    workflow.add_node("model_builder", model_builder)
    workflow.add_node("visualization_generator", visualization_generator)
    workflow.add_node("report_generator", report_generator)
    
    # 设置流程
    workflow.set_entry_point("data_loader")
    workflow.add_edge("data_loader", "data_cleaner")
    workflow.add_edge("data_cleaner", "exploratory_analyzer")
    workflow.add_edge("exploratory_analyzer", "model_builder")
    workflow.add_edge("model_builder", "visualization_generator")
    workflow.add_edge("visualization_generator", "report_generator")
    workflow.set_finish_point("report_generator")
    
    return workflow

# 使用示例
def run_data_analysis():
    """运行数据分析系统"""
    context = AnalysisContext(
        analyst_id="ANALYST_001",
        project_name="员工薪资分析",
        data_source="csv",
        analysis_type="predictive",
        visualization_preference="both"
    )
    
    workflow = create_data_analysis_workflow()
    app = workflow.compile()
    
    # 开始分析
    result = app.invoke({}, context=context)
    
    print(f"分析结果：{result}")
```

### 3.4.3 设计模式特点

1. **流水线处理**：数据加载 → 清洗 → 探索 → 建模 → 可视化 → 报告
2. **类型化分析**：根据分析类型选择不同的建模策略
3. **可视化定制**：根据偏好生成不同类型的可视化
4. **质量保证**：数据清洗、异常检测、模型验证
5. **自动化报告**：自动生成洞察、建议和完整报告

## 3.5 设计模式总结

### 3.5.1 通用设计模式

1. **分层架构模式**
   - 意图识别层
   - 路由决策层
   - 专业处理层
   - 质量保证层

2. **状态机模式**
   - 明确的状态定义
   - 状态转换规则
   - 状态持久化

3. **观察者模式**
   - 事件驱动的处理
   - 松耦合的组件
   - 可扩展的架构

4. **策略模式**
   - 可插拔的处理策略
   - 动态策略选择
   - 策略参数化

### 3.5.2 业务场景适配

1. **客服系统**：分层处理 + 动态路由 + 人工介入
2. **研究助手**：阶段化处理 + 专业化Agent + 质量分级
3. **内容创作**：类型化处理 + 多阶段优化 + 个性化定制
4. **数据分析**：流水线处理 + 类型化分析 + 自动化报告

### 3.5.3 最佳实践

1. **状态设计**
   - 使用TypedDict定义明确的状态结构
   - 使用Annotated进行类型注解
   - 避免循环引用和大型对象

2. **节点设计**
   - 单一职责原则
   - 输入验证和错误处理
   - 清晰的输出格式

3. **工作流设计**
   - 模块化和可重用
   - 条件分支和循环控制
   - 状态持久化和恢复

4. **性能优化**
   - 缓存策略
   - 异步处理
   - 流式执行

通过这些设计模式和最佳实践，可以构建出高效、可靠、可扩展的LangGraph应用系统。 