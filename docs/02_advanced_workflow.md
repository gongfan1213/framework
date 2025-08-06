# LangGraph 多Agents 搭建教程 - 第2部分：高级工作流

## 概述

第2部分将深入讲解LangGraph的高级特性，包括条件分支、循环控制、子工作流、工具集成等。这些特性是构建复杂多Agent系统的核心。

## 2.1 条件分支与路由

### 条件边 (Conditional Edges)

LangGraph支持基于条件的动态路由，这是构建智能多Agent系统的关键特性。

```python
# langgraph/libs/langgraph/langgraph/graph/state.py
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | list[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    """添加条件边"""
```

#### 基本条件路由示例

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    current_task: str
    task_type: Literal["research", "analysis", "summary"]

# 路由函数
def route_by_task_type(state: WorkflowState) -> str:
    """根据任务类型路由到不同的Agent"""
    task_type = state.get("task_type", "research")
    
    if task_type == "research":
        return "researcher"
    elif task_type == "analysis":
        return "analyst"
    elif task_type == "summary":
        return "summarizer"
    else:
        return "general_agent"

# 构建工作流
workflow = StateGraph(WorkflowState)

# 添加节点
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("summarizer", summarizer_agent)
workflow.add_node("general_agent", general_agent)

# 添加条件边
workflow.add_conditional_edges(
    "start",
    route_by_task_type,
    {
        "researcher": "researcher",
        "analyst": "analyst", 
        "summarizer": "summarizer",
        "general_agent": "general_agent"
    }
)

# 所有节点都连接到结束
workflow.add_edge("researcher", END)
workflow.add_edge("analyst", END)
workflow.add_edge("summarizer", END)
workflow.add_edge("general_agent", END)
```

#### 复杂条件路由

```python
def complex_router(state: WorkflowState) -> list[str]:
    """复杂路由：可以返回多个目标"""
    task_type = state.get("task_type")
    priority = state.get("priority", "normal")
    
    routes = []
    
    # 根据任务类型确定主要Agent
    if task_type == "research":
        routes.append("researcher")
    elif task_type == "analysis":
        routes.append("analyst")
    
    # 根据优先级添加审核Agent
    if priority == "high":
        routes.append("reviewer")
    
    # 总是添加总结Agent
    routes.append("summarizer")
    
    return routes

# 使用复杂路由
workflow.add_conditional_edges(
    "start",
    complex_router,
    ["researcher", "analyst", "reviewer", "summarizer"]
)
```

### 动态路由策略

#### 基于内容的路由

```python
def content_based_router(state: WorkflowState) -> str:
    """基于消息内容的路由"""
    messages = state.get("messages", [])
    
    if not messages:
        return "greeter"
    
    last_message = messages[-1][1].lower()
    
    # 关键词路由
    if any(word in last_message for word in ["research", "study", "investigate"]):
        return "researcher"
    elif any(word in last_message for word in ["analyze", "examine", "evaluate"]):
        return "analyst"
    elif any(word in last_message for word in ["summarize", "summary", "conclude"]):
        return "summarizer"
    else:
        return "general_agent"
```

#### 基于状态的智能路由

```python
def intelligent_router(state: WorkflowState) -> str:
    """智能路由：考虑多个因素"""
    task_type = state.get("task_type")
    complexity = state.get("complexity", "medium")
    urgency = state.get("urgency", "normal")
    
    # 复杂任务需要专家Agent
    if complexity == "high":
        if task_type == "research":
            return "senior_researcher"
        elif task_type == "analysis":
            return "senior_analyst"
    
    # 紧急任务需要快速处理
    if urgency == "high":
        return "fast_processor"
    
    # 默认路由
    return task_type + "_agent"
```

## 2.2 循环控制与迭代

### 循环模式

LangGraph支持多种循环模式，从简单的重复到复杂的迭代控制。

#### 基本循环

```python
def basic_loop_agent(state: WorkflowState) -> dict:
    """基本循环Agent"""
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    
    # 检查循环条件
    if iteration >= 5:
        return {"status": "finished", "final_result": "Max iterations reached"}
    
    # 处理逻辑
    response = f"Processing iteration {iteration + 1}"
    
    return {
        "messages": [("assistant", response)],
        "iteration": iteration + 1,
        "status": "continue"
    }

# 循环路由
def loop_router(state: WorkflowState) -> str:
    """循环路由"""
    status = state.get("status", "continue")
    
    if status == "finished":
        return END
    else:
        return "basic_loop_agent"

# 构建循环工作流
workflow = StateGraph(WorkflowState)
workflow.add_node("basic_loop_agent", basic_loop_agent)
workflow.add_conditional_edges("basic_loop_agent", loop_router)
```

#### 条件循环

```python
def conditional_loop_agent(state: WorkflowState) -> dict:
    """条件循环Agent"""
    messages = state.get("messages", [])
    target_quality = state.get("target_quality", 0.8)
    current_quality = state.get("current_quality", 0.0)
    
    # 质量评估
    new_quality = assess_quality(messages)
    
    if new_quality >= target_quality:
        return {
            "status": "finished",
            "final_quality": new_quality,
            "current_quality": new_quality
        }
    else:
        # 继续改进
        improvement = generate_improvement(messages)
        return {
            "messages": [("assistant", improvement)],
            "current_quality": new_quality,
            "status": "continue"
        }

def quality_loop_router(state: WorkflowState) -> str:
    """质量循环路由"""
    status = state.get("status")
    
    if status == "finished":
        return END
    else:
        return "conditional_loop_agent"
```

#### 多阶段循环

```python
class MultiStageState(TypedDict):
    messages: Annotated[list, add_messages]
    stage: Literal["planning", "execution", "review", "refinement"]
    iteration: int
    quality_score: float

def multi_stage_agent(state: MultiStageState) -> dict:
    """多阶段Agent"""
    stage = state.get("stage", "planning")
    iteration = state.get("iteration", 0)
    
    if stage == "planning":
        plan = create_plan(state)
        return {
            "messages": [("assistant", f"Plan: {plan}")],
            "stage": "execution"
        }
    elif stage == "execution":
        result = execute_plan(state)
        return {
            "messages": [("assistant", f"Result: {result}")],
            "stage": "review"
        }
    elif stage == "review":
        quality = review_result(state)
        if quality >= 0.8 or iteration >= 3:
            return {
                "status": "finished",
                "final_quality": quality
            }
        else:
            return {
                "messages": [("assistant", "Need refinement")],
                "stage": "refinement",
                "quality_score": quality
            }
    elif stage == "refinement":
        refined = refine_result(state)
        return {
            "messages": [("assistant", f"Refined: {refined}")],
            "stage": "planning",
            "iteration": iteration + 1
        }

def multi_stage_router(state: MultiStageState) -> str:
    """多阶段路由"""
    status = state.get("status")
    
    if status == "finished":
        return END
    else:
        return "multi_stage_agent"
```

## 2.3 子工作流与嵌套

### 子工作流概念

LangGraph支持工作流的嵌套，可以将复杂的工作流分解为更小的、可重用的组件。

```python
# langgraph/libs/langgraph/langgraph/graph/state.py
def add_node(self, node, *, defer=False, metadata=None):
    """添加节点，支持子工作流"""
    if isinstance(node, StateGraph):
        # 子工作流节点
        self.nodes[name] = node.compile()
    else:
        # 普通节点
        self.nodes[name] = node
```

#### 创建子工作流

```python
# 研究子工作流
def create_research_workflow() -> StateGraph:
    """创建研究子工作流"""
    
    class ResearchState(TypedDict):
        messages: Annotated[list, add_messages]
        research_topic: str
        sources: list
        findings: list
    
    workflow = StateGraph(ResearchState)
    
    # 添加研究节点
    workflow.add_node("topic_analyzer", topic_analyzer_agent)
    workflow.add_node("source_finder", source_finder_agent)
    workflow.add_node("content_analyzer", content_analyzer_agent)
    workflow.add_node("findings_synthesizer", findings_synthesizer_agent)
    
    # 连接节点
    workflow.add_edge("topic_analyzer", "source_finder")
    workflow.add_edge("source_finder", "content_analyzer")
    workflow.add_edge("content_analyzer", "findings_synthesizer")
    workflow.add_edge("findings_synthesizer", END)
    
    workflow.set_entry_point("topic_analyzer")
    
    return workflow

# 分析子工作流
def create_analysis_workflow() -> StateGraph:
    """创建分析子工作流"""
    
    class AnalysisState(TypedDict):
        messages: Annotated[list, add_messages]
        data: dict
        analysis_type: str
        insights: list
    
    workflow = StateGraph(AnalysisState)
    
    # 添加分析节点
    workflow.add_node("data_validator", data_validator_agent)
    workflow.add_node("statistical_analyzer", statistical_analyzer_agent)
    workflow.add_node("trend_analyzer", trend_analyzer_agent)
    workflow.add_node("insight_generator", insight_generator_agent)
    
    # 连接节点
    workflow.add_edge("data_validator", "statistical_analyzer")
    workflow.add_edge("statistical_analyzer", "trend_analyzer")
    workflow.add_edge("trend_analyzer", "insight_generator")
    workflow.add_edge("insight_generator", END)
    
    workflow.set_entry_point("data_validator")
    
    return workflow
```

#### 集成子工作流

```python
# 主工作流
class MainWorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    task_type: str
    research_results: dict
    analysis_results: dict
    final_report: str

def main_workflow():
    """主工作流"""
    workflow = StateGraph(MainWorkflowState)
    
    # 添加子工作流
    research_workflow = create_research_workflow()
    analysis_workflow = create_analysis_workflow()
    
    workflow.add_node("research", research_workflow)
    workflow.add_node("analysis", analysis_workflow)
    workflow.add_node("report_generator", report_generator_agent)
    
    # 条件路由
    def main_router(state: MainWorkflowState) -> str:
        task_type = state.get("task_type")
        
        if task_type == "research":
            return "research"
        elif task_type == "analysis":
            return "analysis"
        else:
            return "report_generator"
    
    workflow.add_conditional_edges("start", main_router)
    workflow.add_edge("research", "report_generator")
    workflow.add_edge("analysis", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow
```

## 2.4 工具集成与调用

### 工具系统架构

LangGraph的工具集成基于LangChain的工具系统，支持复杂的工具调用链。

```python
# langgraph/libs/langgraph/langgraph/graph/state.py
def add_node(self, node, *, defer=False, metadata=None):
    """添加节点，支持工具集成"""
    if hasattr(node, 'tools'):
        # 节点包含工具
        self.nodes[name] = node
```

#### 基本工具集成

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    # 实现搜索逻辑
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression"

@tool
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 实现天气API调用
    return f"Weather in {city}: Sunny, 25°C"

# 创建带工具的Agent
def tool_agent(state: WorkflowState) -> dict:
    """带工具的Agent"""
    messages = state.get("messages", [])
    
    # 创建工具链
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # 检测是否需要工具
    last_message = messages[-1][1] if messages else ""
    
    if "search" in last_message.lower():
        result = search_web(last_message)
        return {"messages": [("assistant", f"Search result: {result}")]}
    elif "calculate" in last_message.lower():
        # 提取数学表达式
        import re
        expression = re.findall(r'calculate\s+(.+)', last_message, re.IGNORECASE)
        if expression:
            result = calculate(expression[0])
            return {"messages": [("assistant", result)]}
    elif "weather" in last_message.lower():
        # 提取城市名
        import re
        city = re.findall(r'weather\s+in\s+(\w+)', last_message, re.IGNORECASE)
        if city:
            result = get_weather(city[0])
            return {"messages": [("assistant", result)]}
    
    # 默认LLM响应
    response = llm.invoke(messages)
    return {"messages": [("assistant", response.content)]}
```

#### 动态工具选择

```python
def dynamic_tool_agent(state: WorkflowState) -> dict:
    """动态工具选择Agent"""
    messages = state.get("messages", [])
    
    # 工具映射
    tools = {
        "search": search_web,
        "calculate": calculate,
        "weather": get_weather
    }
    
    # 分析消息，选择合适工具
    last_message = messages[-1][1] if messages else ""
    
    selected_tool = None
    tool_args = {}
    
    # 工具选择逻辑
    if "search" in last_message.lower():
        selected_tool = "search"
        # 提取搜索查询
        import re
        query = re.findall(r'search\s+(.+)', last_message, re.IGNORECASE)
        if query:
            tool_args["query"] = query[0]
    elif "calculate" in last_message.lower():
        selected_tool = "calculate"
        import re
        expression = re.findall(r'calculate\s+(.+)', last_message, re.IGNORECASE)
        if expression:
            tool_args["expression"] = expression[0]
    elif "weather" in last_message.lower():
        selected_tool = "weather"
        import re
        city = re.findall(r'weather\s+in\s+(\w+)', last_message, re.IGNORECASE)
        if city:
            tool_args["city"] = city[0]
    
    if selected_tool and tool_args:
        # 执行工具
        result = tools[selected_tool](**tool_args)
        return {
            "messages": [("assistant", f"Tool result: {result}")],
            "tool_used": selected_tool,
            "tool_args": tool_args
        }
    else:
        # 默认LLM响应
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        response = llm.invoke(messages)
        return {"messages": [("assistant", response.content)]}
```

#### 工具链集成

```python
def tool_chain_agent(state: WorkflowState) -> dict:
    """工具链Agent"""
    messages = state.get("messages", [])
    
    # 复杂的工具链逻辑
    last_message = messages[-1][1] if messages else ""
    
    if "research" in last_message.lower():
        # 研究工具链：搜索 -> 分析 -> 总结
        search_result = search_web(last_message)
        
        # 分析搜索结果
        analysis_result = analyze_content(search_result)
        
        # 生成总结
        summary = summarize_findings(analysis_result)
        
        return {
            "messages": [("assistant", f"Research summary: {summary}")],
            "search_result": search_result,
            "analysis_result": analysis_result,
            "summary": summary
        }
    
    # 默认处理
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(messages)
    return {"messages": [("assistant", response.content)]}

# 辅助函数
def analyze_content(content: str) -> str:
    """分析内容"""
    return f"Analysis of: {content[:100]}..."

def summarize_findings(analysis: str) -> str:
    """总结发现"""
    return f"Summary: {analysis[:200]}..."
```

## 2.5 状态持久化与恢复

### Checkpoint 机制

LangGraph提供了强大的状态持久化机制，支持工作流的暂停和恢复。

```python
# langgraph/libs/langgraph/langgraph/checkpoint/base.py
class Checkpoint:
    """检查点基类"""
    def __init__(self, config: dict):
        self.config = config
        self.values = {}
```

#### 基本检查点使用

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点保存器
checkpointer = MemorySaver()

# 编译时添加检查点
workflow = StateGraph(WorkflowState)
# ... 添加节点和边 ...
app = workflow.compile(checkpointer=checkpointer)

# 运行工作流
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [("user", "Hello")]}, config=config)

# 获取检查点
checkpoint = app.get_state(config)
print(f"Current state: {checkpoint.values}")
```

#### 分布式检查点

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 使用PostgreSQL作为检查点存储
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph"
)

# 编译工作流
app = workflow.compile(checkpointer=checkpointer)

# 运行工作流
config = {"configurable": {"thread_id": "user-456"}}
result = app.invoke({"messages": [("user", "Continue")]}, config=config)
```

#### 检查点恢复

```python
def resume_workflow(app, thread_id: str):
    """恢复工作流"""
    config = {"configurable": {"thread_id": thread_id}}
    
    # 获取当前状态
    checkpoint = app.get_state(config)
    
    if checkpoint.values:
        # 有现有状态，继续执行
        print(f"Resuming workflow with state: {checkpoint.values}")
        result = app.invoke({}, config=config)
    else:
        # 新工作流
        print("Starting new workflow")
        result = app.invoke({"messages": [("user", "Start")]}, config=config)
    
    return result
```

## 2.6 并发控制与同步

### 并发执行模式

LangGraph支持多种并发执行模式，从简单的并行到复杂的同步控制。

#### 并行节点执行

```python
def parallel_agents_workflow():
    """并行Agent工作流"""
    
    class ParallelState(TypedDict):
        messages: Annotated[list, add_messages]
        research_result: str
        analysis_result: str
        summary_result: str
    
    workflow = StateGraph(ParallelState)
    
    # 添加并行节点
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("summarizer", summarizer_agent)
    workflow.add_node("coordinator", coordinator_agent)
    
    # 并行执行：多个节点同时运行
    workflow.add_edge("start", "researcher")
    workflow.add_edge("start", "analyst")
    workflow.add_edge("start", "summarizer")
    
    # 收集结果
    workflow.add_edge("researcher", "coordinator")
    workflow.add_edge("analyst", "coordinator")
    workflow.add_edge("summarizer", "coordinator")
    
    workflow.add_edge("coordinator", END)
    
    return workflow
```

#### 同步控制

```python
def synchronized_workflow():
    """同步工作流"""
    
    class SyncState(TypedDict):
        messages: Annotated[list, add_messages]
        phase: Literal["planning", "execution", "review"]
        planning_done: bool
        execution_done: bool
        review_done: bool
    
    workflow = StateGraph(SyncState)
    
    # 添加同步节点
    workflow.add_node("planner", planner_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("reviewer", reviewer_agent)
    workflow.add_node("coordinator", coordinator_agent)
    
    # 同步路由
    def sync_router(state: SyncState) -> str:
        phase = state.get("phase", "planning")
        
        if phase == "planning" and not state.get("planning_done"):
            return "planner"
        elif phase == "execution" and not state.get("execution_done"):
            return "executor"
        elif phase == "review" and not state.get("review_done"):
            return "reviewer"
        else:
            return "coordinator"
    
    workflow.add_conditional_edges("start", sync_router)
    workflow.add_edge("planner", "coordinator")
    workflow.add_edge("executor", "coordinator")
    workflow.add_edge("reviewer", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow
```

## 2.7 错误处理与恢复

### 错误处理策略

```python
# langgraph/libs/langgraph/langgraph/errors.py
class GraphRecursionError(RecursionError):
    """图递归限制错误"""
    pass

class InvalidUpdateError(Exception):
    """无效更新错误"""
    pass
```

#### 节点级错误处理

```python
def robust_agent(state: WorkflowState) -> dict:
    """健壮的Agent，包含错误处理"""
    try:
        messages = state.get("messages", [])
        
        # 主要逻辑
        result = process_messages(messages)
        
        return {
            "messages": [("assistant", result)],
            "status": "success"
        }
        
    except Exception as e:
        # 错误处理
        error_message = f"Error occurred: {str(e)}"
        
        return {
            "messages": [("assistant", error_message)],
            "status": "error",
            "error": str(e)
        }
```

#### 工作流级错误处理

```python
def error_handling_workflow():
    """错误处理工作流"""
    
    class ErrorState(TypedDict):
        messages: Annotated[list, add_messages]
        status: str
        error: str
        retry_count: int
    
    workflow = StateGraph(ErrorState)
    
    # 添加节点
    workflow.add_node("main_agent", robust_agent)
    workflow.add_node("error_handler", error_handler_agent)
    workflow.add_node("retry_agent", retry_agent)
    
    # 错误处理路由
    def error_router(state: ErrorState) -> str:
        status = state.get("status", "success")
        retry_count = state.get("retry_count", 0)
        
        if status == "error" and retry_count < 3:
            return "retry_agent"
        elif status == "error":
            return "error_handler"
        else:
            return END
    
    workflow.add_conditional_edges("main_agent", error_router)
    workflow.add_edge("retry_agent", "main_agent")
    workflow.add_edge("error_handler", END)
    
    return workflow
```

## 2.8 性能优化技巧

### 缓存策略

```python
from langgraph.cache import InMemoryCache

# 使用缓存
cache = InMemoryCache()

# 编译时添加缓存
app = workflow.compile(cache=cache)
```

### 异步执行

```python
import asyncio

async def async_agent(state: WorkflowState) -> dict:
    """异步Agent"""
    messages = state.get("messages", [])
    
    # 异步处理
    result = await async_process_messages(messages)
    
    return {"messages": [("assistant", result)]}

# 异步运行
async def run_async_workflow():
    result = await app.ainvoke({"messages": [("user", "Hello")]})
    return result
```

### 流式处理

```python
# 流式执行
for chunk in app.stream({"messages": [("user", "Hello")]}):
    print(f"Chunk: {chunk}")
```

## 总结

第2部分深入讲解了LangGraph的高级特性：

1. **条件分支与路由**：动态路由、复杂条件、智能路由
2. **循环控制与迭代**：基本循环、条件循环、多阶段循环
3. **子工作流与嵌套**：子工作流创建、集成、重用
4. **工具集成与调用**：基本工具、动态选择、工具链
5. **状态持久化与恢复**：检查点机制、分布式存储、状态恢复
6. **并发控制与同步**：并行执行、同步控制、协调机制
7. **错误处理与恢复**：错误分类、处理策略、恢复机制
8. **性能优化技巧**：缓存、异步、流式处理

在下一部分中，我们将通过实际案例，展示如何构建复杂的多Agent系统，包括智能客服、研究助手、数据分析等实际应用场景。 