# LangGraph 常用API及实际业务场景完整指南

## 概述

LangGraph是LangChain生态系统中的工作流框架，基于**Pregel算法**和**Actor模型**构建，专门用于构建复杂的多Agent系统和状态驱动的工作流。

## 核心概念

### 1. 图计算模型
```python
from langgraph.graph import StateGraph
from langgraph.types import StateType

# 状态定义
class State(StateType):
    messages: list
    current_step: str
    data: dict

# 图构建
graph = StateGraph(State)
```

### 2. BSP执行模型
LangGraph采用Bulk Synchronous Parallel (BSP) 模型：
- **Plan阶段**：确定要执行的节点
- **Execution阶段**：并行执行所有选中的节点  
- **Update阶段**：更新通道中的值

### 3. Actor模型
每个节点都是一个独立的Actor：
- **独立状态**：每个节点维护自己的状态
- **消息传递**：节点间通过通道进行异步通信
- **并发执行**：多个节点可以并行运行

## 基础API

### 1. 状态图构建
```python
from langgraph.graph import StateGraph
from langgraph.types import StateType
from typing import TypedDict, Annotated
from langgraph.channels import LastValue

# 状态定义
class State(TypedDict):
    messages: Annotated[list, LastValue]
    current_step: str
    user_data: dict
    results: list

# 创建状态图
graph = StateGraph(State)

# 添加节点
def process_node(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "处理完成"}]}

graph.add_node("process", process_node)

# 添加边
graph.add_edge("process", "end")

# 编译图
compiled_graph = graph.compile()
```

### 2. 节点函数
```python
def simple_node(state: State) -> dict:
    """简单节点函数"""
    return {"messages": [{"role": "assistant", "content": "Hello"}]}

def conditional_node(state: State) -> dict:
    """条件节点函数"""
    if len(state["messages"]) > 5:
        return {"messages": [{"role": "assistant", "content": "消息过多"}]}
    return {"messages": [{"role": "assistant", "content": "继续处理"}]}
```

### 3. 边和条件
```python
from langgraph.constants import START, END

# 条件边
def should_continue(state: State) -> str:
    if state["current_step"] == "processing":
        return "continue_processing"
    elif state["current_step"] == "complete":
        return END
    else:
        return "error_handler"

# 添加条件边
graph.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue_processing": "process",
        "error_handler": "handle_error",
        END: END
    }
)
```

## 状态管理

### 1. 状态定义
```python
from typing import TypedDict, Annotated
from langgraph.channels import LastValue, Topic

class State(TypedDict):
    # 消息历史 - 使用LastValue通道
    messages: Annotated[list, LastValue]
    
    # 当前步骤 - 简单字符串
    current_step: str
    
    # 用户数据 - 字典
    user_data: dict
    
    # 结果列表 - 使用Topic通道支持多值
    results: Annotated[list, Topic]
    
    # 错误信息 - 可选字段
    error: str | None
```

### 2. 状态更新
```python
def update_state(state: State) -> dict:
    """更新状态"""
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": "新消息"}],
        "current_step": "updated",
        "results": ["结果1", "结果2"]
    }

def partial_update(state: State) -> dict:
    """部分更新状态"""
    return {
        "current_step": "partial_updated"
        # 只更新current_step，其他字段保持不变
    }
```

## 通道系统

### 1. LastValue通道
```python
from langgraph.channels import LastValue

class State(TypedDict):
    # 只保留最后一个值
    current_message: Annotated[str, LastValue]
    count: Annotated[int, LastValue]

def node_with_last_value(state: State) -> dict:
    return {
        "current_message": "最新消息",
        "count": state.get("count", 0) + 1
    }
```

### 2. Topic通道
```python
from langgraph.channels import Topic

class State(TypedDict):
    # 累积多个值
    all_messages: Annotated[list, Topic]
    notifications: Annotated[list, Topic]

def node_with_topic(state: State) -> dict:
    return {
        "all_messages": ["消息1", "消息2", "消息3"],
        "notifications": ["通知1", "通知2"]
    }
```

## 并行执行

### 1. 基础并行
```python
def parallel_node_1(state: State) -> dict:
    return {"parallel_results": ["结果1"]}

def parallel_node_2(state: State) -> dict:
    return {"parallel_results": ["结果2"]}

def parallel_node_3(state: State) -> dict:
    return {"parallel_results": ["结果3"]}

# 构建并行图
graph = StateGraph(State)
graph.add_node("node1", parallel_node_1)
graph.add_node("node2", parallel_node_2)
graph.add_node("node3", parallel_node_3)

# 并行执行
graph.add_edge(START, "node1")
graph.add_edge(START, "node2")
graph.add_edge(START, "node3")

# 合并结果
def merge_results(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": f"合并结果: {state['parallel_results']}"}]}

graph.add_node("merge", merge_results)
graph.add_edge("node1", "merge")
graph.add_edge("node2", "merge")
graph.add_edge("node3", "merge")
```

### 2. Map-Reduce模式
```python
def map_node(state: State) -> dict:
    """Map节点 - 并行处理每个输入"""
    results = []
    for item in state["input_data"]:
        results.append(f"处理: {item}")
    return {"mapped_results": results}

def reduce_node(state: State) -> dict:
    """Reduce节点 - 合并所有结果"""
    combined = ", ".join(state["mapped_results"])
    return {"final_result": f"最终结果: {combined}"}

# 构建Map-Reduce图
graph = StateGraph(State)
graph.add_node("map", map_node)
graph.add_node("reduce", reduce_node)

graph.add_edge(START, "map")
graph.add_edge("map", "reduce")
```

## 多Agent系统

### 1. 监督者模式
```python
from langchain_openai import ChatOpenAI

def supervisor_agent(state: State) -> dict:
    """监督者Agent"""
    llm = ChatOpenAI()
    # 根据消息内容决定调用哪个Agent
    if "数据分析" in str(state["messages"]):
        return {"current_agent": "data_analyst"}
    elif "代码生成" in str(state["messages"]):
        return {"current_agent": "code_generator"}
    else:
        return {"current_agent": "general_assistant"}

def data_analyst_agent(state: State) -> dict:
    return {
        "agent_results": {"data_analyst": "数据分析结果"},
        "messages": [{"role": "assistant", "content": "数据分析完成"}]
    }

def code_generator_agent(state: State) -> dict:
    return {
        "agent_results": {"code_generator": "代码生成结果"},
        "messages": [{"role": "assistant", "content": "代码生成完成"}]
    }

# 构建多Agent系统
graph = StateGraph(State)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("data_analyst", data_analyst_agent)
graph.add_node("code_generator", code_generator_agent)

# 条件边
def route_to_agent(state: State) -> str:
    return state["current_agent"]

graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "data_analyst": "data_analyst",
        "code_generator": "code_generator"
    }
)
```

### 2. 群组模式
```python
def agent_1(state: State) -> dict:
    return {"agent_messages": {"agent1": "Agent 1的观点"}}

def agent_2(state: State) -> dict:
    return {"agent_messages": {"agent2": "Agent 2的观点"}}

def agent_3(state: State) -> dict:
    return {"agent_messages": {"agent3": "Agent 3的观点"}}

def consensus_builder(state: State) -> dict:
    """共识构建器"""
    agent_messages = state.get("agent_messages", {})
    consensus = "基于所有Agent的观点达成共识"
    return {
        "consensus_result": consensus,
        "messages": [{"role": "assistant", "content": consensus}]
    }

# 构建群组系统
graph = StateGraph(State)
graph.add_node("agent1", agent_1)
graph.add_node("agent2", agent_2)
graph.add_node("agent3", agent_3)
graph.add_node("consensus", consensus_builder)

# 并行执行所有Agent
graph.add_edge(START, "agent1")
graph.add_edge(START, "agent2")
graph.add_edge(START, "agent3")

# 等待所有Agent完成后构建共识
graph.add_edge("agent1", "consensus")
graph.add_edge("agent2", "consensus")
graph.add_edge("agent3", "consensus")
```

## 实际业务场景

### 1. 智能客服系统
```python
class CustomerServiceState(TypedDict):
    messages: list
    user_intent: str
    current_department: str
    ticket_id: str
    resolution: str

def intent_classifier(state: CustomerServiceState) -> dict:
    """意图分类器"""
    last_message = state["messages"][-1]["content"]
    
    if "退款" in last_message:
        return {"user_intent": "refund", "current_department": "billing"}
    elif "技术问题" in last_message:
        return {"user_intent": "technical", "current_department": "support"}
    else:
        return {"user_intent": "general", "current_department": "general"}

def billing_agent(state: CustomerServiceState) -> dict:
    return {
        "resolution": "退款申请已处理",
        "messages": [{"role": "assistant", "content": "您的退款申请已提交"}]
    }

def support_agent(state: CustomerServiceState) -> dict:
    return {
        "resolution": "技术问题已解决",
        "messages": [{"role": "assistant", "content": "技术问题已记录"}]
    }

# 构建客服系统
graph = StateGraph(CustomerServiceState)
graph.add_node("intent_classifier", intent_classifier)
graph.add_node("billing", billing_agent)
graph.add_node("support", support_agent)

# 路由逻辑
def route_to_department(state: CustomerServiceState) -> str:
    return state["current_department"]

graph.add_conditional_edges(
    "intent_classifier",
    route_to_department,
    {
        "billing": "billing",
        "support": "support"
    }
)
```

### 2. 文档处理流水线
```python
class DocumentProcessingState(TypedDict):
    raw_document: str
    extracted_text: str
    processed_sections: list
    summary: str
    keywords: list
    final_report: str

def text_extractor(state: DocumentProcessingState) -> dict:
    raw_doc = state["raw_document"]
    extracted = f"提取的文本: {raw_doc[:100]}..."
    return {"extracted_text": extracted}

def section_processor(state: DocumentProcessingState) -> dict:
    text = state["extracted_text"]
    sections = [f"段落{i+1}: {text[i*50:(i+1)*50]}" for i in range(3)]
    return {"processed_sections": sections}

def summarizer(state: DocumentProcessingState) -> dict:
    sections = state["processed_sections"]
    summary = f"文档摘要: 包含{len(sections)}个段落"
    return {"summary": summary}

# 构建文档处理流水线
graph = StateGraph(DocumentProcessingState)
graph.add_node("extract", text_extractor)
graph.add_node("process_sections", section_processor)
graph.add_node("summarize", summarizer)

# 流水线顺序
graph.add_edge(START, "extract")
graph.add_edge("extract", "process_sections")
graph.add_edge("process_sections", "summarize")
```

### 3. 数据分析工作流
```python
class DataAnalysisState(TypedDict):
    raw_data: list
    cleaned_data: list
    statistical_analysis: dict
    visualization_data: dict
    insights: list
    final_report: str

def data_cleaner(state: DataAnalysisState) -> dict:
    raw_data = state["raw_data"]
    cleaned = [item for item in raw_data if item is not None]
    return {"cleaned_data": cleaned}

def statistical_analyzer(state: DataAnalysisState) -> dict:
    data = state["cleaned_data"]
    stats = {
        "count": len(data),
        "mean": sum(data) / len(data) if data else 0,
        "max": max(data) if data else 0,
        "min": min(data) if data else 0
    }
    return {"statistical_analysis": stats}

def insight_generator(state: DataAnalysisState) -> dict:
    stats = state["statistical_analysis"]
    insights = [
        f"数据点数量: {stats['count']}",
        f"平均值: {stats['mean']:.2f}",
        f"数据范围: {stats['min']} - {stats['max']}"
    ]
    return {"insights": insights}

# 构建数据分析工作流
graph = StateGraph(DataAnalysisState)
graph.add_node("clean", data_cleaner)
graph.add_node("analyze", statistical_analyzer)
graph.add_node("generate_insights", insight_generator)

# 工作流顺序
graph.add_edge(START, "clean")
graph.add_edge("clean", "analyze")
graph.add_edge("analyze", "generate_insights")
```

## 常见易错点

### 1. 状态更新冲突
```python
# ❌ 错误示例 - 多个节点同时更新同一字段
def node1(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "消息1"}]}

def node2(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "消息2"}]}

# 如果node1和node2并行执行，会导致状态冲突

# ✅ 正确示例 - 使用Topic通道
class State(TypedDict):
    messages: Annotated[list, Topic]  # 使用Topic支持多值

def node1(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "消息1"}]}

def node2(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "消息2"}]}
```

### 2. 循环依赖
```python
# ❌ 错误示例 - 循环依赖
graph.add_edge("node1", "node2")
graph.add_edge("node2", "node1")  # 这会导致无限循环

# ✅ 正确示例 - 添加条件终止
def should_continue(state: State) -> str:
    if state["step_count"] > 10:
        return END
    return "node2"

graph.add_conditional_edges(
    "node1",
    should_continue,
    {"node2": "node2", END: END}
)
```

### 3. 状态类型不匹配
```python
# ❌ 错误示例 - 状态类型不匹配
class State(TypedDict):
    count: int

def node(state: State) -> dict:
    return {"count": "not_a_number"}  # 类型错误

# ✅ 正确示例 - 确保类型匹配
def node(state: State) -> dict:
    return {"count": 42}  # 正确的整数类型
```

## 复杂场景应用

### 1. 工具调用并行和结果合并
```python
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class ToolExecutionState(TypedDict):
    query: str
    search_results: dict
    wiki_results: dict
    combined_results: str
    final_answer: str

def search_tool(state: ToolExecutionState) -> dict:
    """搜索工具"""
    search = DuckDuckGoSearchRun()
    query = state["query"]
    results = search.run(query)
    return {"search_results": {"source": "search", "results": results}}

def wiki_tool(state: ToolExecutionState) -> dict:
    """维基百科工具"""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    query = state["query"]
    results = wiki.run(query)
    return {"wiki_results": {"source": "wikipedia", "results": results}}

def result_merger(state: ToolExecutionState) -> dict:
    """结果合并器"""
    search_results = state.get("search_results", {})
    wiki_results = state.get("wiki_results", {})
    
    combined = f"""
    搜索结果: {search_results.get('results', '无结果')}
    维基百科结果: {wiki_results.get('results', '无结果')}
    """
    return {"combined_results": combined}

# 构建工具并行执行系统
graph = StateGraph(ToolExecutionState)
graph.add_node("search", search_tool)
graph.add_node("wiki", wiki_tool)
graph.add_node("merge", result_merger)

# 并行执行工具
graph.add_edge(START, "search")
graph.add_edge(START, "wiki")

# 等待所有工具完成后合并结果
graph.add_edge("search", "merge")
graph.add_edge("wiki", "merge")
```

### 2. 数据流冲突处理
```python
from langgraph.channels import NamedBarrierValue

class DataFlowState(TypedDict):
    input_data: list
    processed_data: Annotated[list, NamedBarrierValue]
    validated_data: list
    final_output: str

def data_processor_1(state: DataFlowState) -> dict:
    """数据处理器1"""
    input_data = state["input_data"]
    processed = [f"处理1_{item}" for item in input_data]
    return {"processed_data": processed}

def data_processor_2(state: DataFlowState) -> dict:
    """数据处理器2"""
    input_data = state["input_data"]
    processed = [f"处理2_{item}" for item in input_data]
    return {"processed_data": processed}

def data_validator(state: DataFlowState) -> dict:
    """数据验证器"""
    processed_data = state["processed_data"]
    validated = [item for item in processed_data if "处理" in item]
    return {"validated_data": validated}

# 构建数据流系统
graph = StateGraph(DataFlowState)
graph.add_node("processor1", data_processor_1)
graph.add_node("processor2", data_processor_2)
graph.add_node("validator", data_validator)

# 并行处理
graph.add_edge(START, "processor1")
graph.add_edge(START, "processor2")

# 等待所有处理器完成后验证
graph.add_edge("processor1", "validator")
graph.add_edge("processor2", "validator")
```

### 3. 动态工作流
```python
class DynamicWorkflowState(TypedDict):
    task_list: list
    current_task: str
    completed_tasks: list
    task_results: dict
    workflow_status: str

def task_scheduler(state: DynamicWorkflowState) -> dict:
    """任务调度器"""
    tasks = state["task_list"]
    completed = state.get("completed_tasks", [])
    
    if not tasks:
        return {"workflow_status": "completed"}
    
    next_task = tasks[0]
    return {"current_task": next_task}

def dynamic_worker(state: DynamicWorkflowState) -> dict:
    """动态工作节点"""
    task = state["current_task"]
    result = f"完成任务: {task}"
    
    completed = state.get("completed_tasks", [])
    completed.append(task)
    
    results = state.get("task_results", {})
    results[task] = result
    
    return {
        "completed_tasks": completed,
        "task_results": results,
        "task_list": state["task_list"][1:]  # 移除已完成的任务
    }

# 构建动态工作流
graph = StateGraph(DynamicWorkflowState)
graph.add_node("scheduler", task_scheduler)
graph.add_node("worker", dynamic_worker)

# 工作流循环
def should_continue(state: DynamicWorkflowState) -> str:
    if state["workflow_status"] == "completed":
        return END
    else:
        return "scheduler"

graph.add_conditional_edges(
    "scheduler",
    should_continue,
    {"worker": "worker", END: END}
)

graph.add_edge("worker", "scheduler")
```

## 最佳实践

### 1. 状态设计原则
```python
# ✅ 好的状态设计
class WellDesignedState(TypedDict):
    # 使用适当的通道类型
    messages: Annotated[list, LastValue]  # 消息历史
    parallel_results: Annotated[list, Topic]  # 并行结果
    current_step: str  # 当前步骤
    metadata: dict  # 元数据
```

### 2. 错误处理
```python
def robust_node(state: State) -> dict:
    """健壮的节点函数"""
    try:
        # 业务逻辑
        result = process_data(state["input"])
        return {"output": result}
    except Exception as e:
        # 错误处理
        return {
            "error": str(e),
            "status": "failed"
        }
```

### 3. 性能优化
```python
# 使用缓存
from langgraph.cache import InMemoryCache
import langgraph

langgraph.cache = InMemoryCache()

# 批量处理
def batch_processor(state: State) -> dict:
    """批量处理器"""
    items = state["items"]
    batch_size = 10
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_result = process_batch(batch)
        results.extend(batch_result)
    
    return {"results": results}
```

### 4. 监控和日志
```python
import logging
from langgraph.callbacks import BaseCallbackHandler

class MonitoringCallback(BaseCallbackHandler):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.logger.info(f"节点开始执行: {serialized.get('name', 'unknown')}")
    
    def on_chain_end(self, outputs, **kwargs):
        self.logger.info(f"节点执行完成: {len(outputs)} 个输出")
    
    def on_chain_error(self, error, **kwargs):
        self.logger.error(f"节点执行出错: {error}")

# 使用监控
monitor = MonitoringCallback()
compiled_graph = graph.compile(checkpointer=checkpointer, callbacks=[monitor])
```

## 总结

LangGraph提供了强大的工作流框架，支持复杂的多Agent系统和状态驱动应用。通过合理使用通道系统、并行执行和状态管理，可以构建高效、可扩展的AI应用。

### 关键要点
1. **状态设计**：合理使用通道类型，避免状态冲突
2. **并行执行**：利用BSP模型实现高效并行
3. **错误处理**：完善的错误处理和恢复机制
4. **性能优化**：使用缓存和批量处理
5. **监控调试**：全面的监控和日志系统

### 学习路径
1. 掌握基础概念：StateGraph、Channel、Pregel
2. 学习状态管理：LastValue、Topic、自定义通道
3. 理解并行执行：BSP模型、Map-Reduce、动态并行
4. 实践复杂场景：多Agent系统、工具调用、数据流
5. 应用最佳实践：错误处理、性能优化、监控调试

通过本文档的学习，您应该能够熟练使用LangGraph构建各种复杂的AI工作流，并能够处理并行执行、数据合并、冲突解决等复杂场景。
