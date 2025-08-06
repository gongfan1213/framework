# LangGraph 多Agents 搭建教程 - 第1部分：基础概念

## 概述

LangGraph是一个基于图计算的工作流框架，特别适合构建复杂的多Agent系统。本教程将从零开始，结合源码详细讲解如何使用LangGraph搭建多agents系统。

## 1.1 LangGraph 核心概念

### 什么是 LangGraph？

LangGraph 是 LangChain 生态系统中的工作流框架，它基于 **Pregel 算法** 和 **Actor 模型**，将复杂的AI应用分解为可管理的组件。

```python
# 核心思想：将应用分解为节点和边
from langgraph.graph import StateGraph

# 每个节点是一个Agent或工具
# 节点之间通过边连接
# 数据通过状态在节点间流动
```

### 核心组件

#### 1. StateGraph - 状态图
```python
# langgraph/libs/langgraph/langgraph/graph/state.py
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state."""
    
    def __init__(self, state_schema: type[StateT]):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = set()
```

**作用**：定义工作流的结构，管理节点间的连接关系。

#### 2. Channel - 通信通道
```python
# langgraph/libs/langgraph/langgraph/channels/__init__.py
from langgraph.channels.base import BaseChannel
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
```

**作用**：节点间数据传递的机制，支持不同类型的通信模式。

#### 3. Pregel - 执行引擎
```python
# langgraph/libs/langgraph/langgraph/pregel/main.py
class Pregel(Generic[StateT, ContextT, InputT, OutputT]):
    """Pregel manages the runtime behavior for LangGraph applications."""
    
    def __init__(self, *, nodes, channels, input_channels, output_channels):
        self.nodes = nodes
        self.channels = channels
```

**作用**：负责工作流的执行，管理节点调度和状态更新。

## 1.2 基本架构模式

### BSP (Bulk Synchronous Parallel) 模型

LangGraph 采用 BSP 模型，每个执行步骤包含三个阶段：

1. **Plan 阶段**：确定要执行的节点
2. **Execution 阶段**：并行执行所有选中的节点
3. **Update 阶段**：更新通道中的值

```python
# 执行流程示意
def execute_step(self):
    # 1. Plan: 确定要执行的节点
    nodes_to_execute = self.plan_nodes()
    
    # 2. Execution: 并行执行节点
    results = self.execute_nodes_parallel(nodes_to_execute)
    
    # 3. Update: 更新通道
    self.update_channels(results)
```

### Actor 模型

每个节点都是一个独立的 Actor，具有以下特性：

- **独立状态**：每个节点维护自己的状态
- **消息传递**：节点间通过通道进行异步通信
- **并发执行**：多个节点可以并行运行

```python
# Actor模型示例
class MyActor:
    def __init__(self, name: str):
        self.name = name
        self.state = {}
    
    def receive(self, message: dict):
        # 处理接收到的消息
        return self.process(message)
    
    def process(self, message: dict):
        # 业务逻辑处理
        return {"result": f"Processed by {self.name}"}
```

## 1.3 新的API架构

### 1.3.1 函数式API (Functional API)

LangGraph v0.6+ 引入了新的函数式API，提供了更简洁的编程方式：

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def process_data(data: str) -> str:
    """处理数据的任务"""
    return f"Processed: {data}"

@task
def validate_result(result: str) -> bool:
    """验证结果的任务"""
    return len(result) > 10

@entrypoint(checkpointer=InMemorySaver())
def workflow(input_data: str) -> dict:
    """主工作流"""
    # 并行执行任务
    processed_future = process_data(input_data)
    validation_future = validate_result(processed_future.result())
    
    return {
        "processed": processed_future.result(),
        "valid": validation_future.result()
    }

# 使用工作流
result = workflow.invoke("Hello World")
```

### 1.3.2 状态图API (StateGraph API)

状态图API是LangGraph的核心API，用于构建复杂的工作流：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

# 定义状态模式
class State(TypedDict):
    messages: Annotated[list, "messages"]
    current_step: str
    final_answer: str

# 定义上下文模式
class Context(TypedDict):
    user_id: str
    session_id: str

# 创建状态图
workflow = StateGraph(
    state_schema=State,
    context_schema=Context
)

# 添加节点
def agent_node(state: State, runtime: Runtime[Context]) -> dict:
    """Agent节点处理逻辑"""
    user_id = runtime.context.get("user_id", "unknown")
    messages = state["messages"]
    
    # 处理消息并返回更新
    return {
        "current_step": "agent_processed",
        "final_answer": f"Processed by {user_id}"
    }

workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

# 编译工作流
compiled_workflow = workflow.compile()
```

### 1.3.3 通道系统 (Channel System)

LangGraph提供了多种通道类型来支持不同的通信模式：

#### 基础通道

```python
from langgraph.channels import LastValue, EphemeralValue, Topic

# LastValue - 存储最后一个值
last_value_channel = LastValue(str)

# EphemeralValue - 临时值，不持久化
ephemeral_channel = EphemeralValue(str)

# Topic - 发布订阅模式
topic_channel = Topic(str, accumulate=True)
```

#### 高级通道

```python
from langgraph.channels import BinaryOperatorAggregate, NamedBarrierValue

# BinaryOperatorAggregate - 聚合操作
def reducer(current: str, update: str) -> str:
    return current + " | " + update

aggregate_channel = BinaryOperatorAggregate(str, operator=reducer)

# NamedBarrierValue - 命名屏障
barrier_channel = NamedBarrierValue(str)
```

### 1.3.4 检查点系统 (Checkpoint System)

LangGraph提供了强大的状态持久化机制：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 内存检查点
memory_checkpointer = InMemorySaver()

# SQLite检查点
sqlite_checkpointer = SqliteSaver("workflow.db")

# 使用检查点
@entrypoint(checkpointer=memory_checkpointer)
def persistent_workflow(input_data: str, previous: str = None) -> str:
    """支持状态持久化的工作流"""
    if previous:
        return f"Previous: {previous}, Current: {input_data}"
    return f"First run: {input_data}"
```

## 1.4 执行模型

### 1.4.1 BSP执行模型

LangGraph采用Bulk Synchronous Parallel (BSP)模型，每个执行步骤包含三个阶段：

```python
# BSP执行流程
class BSPExecutor:
    def execute_step(self, nodes: list, channels: dict):
        # 1. Plan阶段：确定要执行的节点
        nodes_to_execute = self.plan_nodes(nodes, channels)
        
        # 2. Execution阶段：并行执行节点
        results = self.execute_nodes_parallel(nodes_to_execute)
        
        # 3. Update阶段：更新通道
        self.update_channels(results, channels)
        
        return results
```

### 1.4.2 流式执行

LangGraph支持流式执行，可以实时获取执行结果：

```python
# 流式执行示例
for step_result in workflow.stream({"input": "test"}):
    print(f"Step: {step_result}")

# 异步流式执行
async for step_result in workflow.astream({"input": "test"}):
    print(f"Async Step: {step_result}")
```

### 1.4.3 中断和恢复

LangGraph支持工作流的中断和恢复：

```python
from langgraph.types import interrupt, Command

@entrypoint(checkpointer=InMemorySaver())
def interruptible_workflow(input_data: str) -> dict:
    """可中断的工作流"""
    # 处理数据
    processed_data = process_data(input_data)
    
    # 中断等待人工审核
    human_review = interrupt({
        "question": "请审核结果",
        "data": processed_data
    })
    
    return {
        "original": input_data,
        "processed": processed_data,
        "review": human_review
    }

# 恢复工作流
resume_command = Command(resume="审核通过")
result = workflow.invoke(resume_command)
```

## 1.5 错误处理和重试

### 1.5.1 重试策略

LangGraph提供了灵活的重试机制：

```python
from langgraph.types import RetryPolicy
import time

# 指数退避重试
exponential_retry = RetryPolicy(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0
)

# 固定间隔重试
fixed_retry = RetryPolicy(
    max_retries=5,
    delay=2.0
)

@task(retry_policy=exponential_retry)
def unreliable_task(data: str) -> str:
    """可能失败的任务"""
    if time.time() % 3 == 0:  # 模拟随机失败
        raise Exception("Random failure")
    return f"Success: {data}"
```

### 1.5.2 缓存策略

LangGraph支持结果缓存以提高性能：

```python
from langgraph.types import CachePolicy
from langgraph.cache import InMemoryCache

# 内存缓存
memory_cache = InMemoryCache()

# 缓存策略
cache_policy = CachePolicy(
    ttl=3600,  # 1小时过期
    max_size=1000
)

@task(cache_policy=cache_policy)
def expensive_computation(data: str) -> str:
    """昂贵的计算任务"""
    time.sleep(2)  # 模拟耗时操作
    return f"Computed: {data}"
```

## 1.6 最佳实践

### 1.6.1 状态设计

```python
# 好的状态设计
class GoodState(TypedDict):
    # 使用明确的类型注解
    messages: Annotated[list, "messages"]
    current_step: str
    metadata: dict
    
    # 避免循环引用
    # 避免存储大型对象
    # 使用不可变数据结构
```

### 1.6.2 节点设计

```python
# 好的节点设计
def well_designed_node(state: State, runtime: Runtime[Context]) -> dict:
    """设计良好的节点"""
    try:
        # 1. 输入验证
        if not state.get("messages"):
            return {"error": "No messages found"}
        
        # 2. 业务逻辑
        result = process_messages(state["messages"])
        
        # 3. 输出格式化
        return {
            "current_step": "completed",
            "result": result
        }
    except Exception as e:
        # 4. 错误处理
        return {"error": str(e)}
```

### 1.6.3 性能优化

```python
# 性能优化技巧
@task(cache_policy=CachePolicy(ttl=3600))
def optimized_node(state: State) -> dict:
    """优化的节点"""
    # 1. 使用缓存
    # 2. 避免不必要的计算
    # 3. 使用异步操作
    # 4. 合理设置超时
    return {"optimized_result": "done"}
```

## 1.7 总结

LangGraph是一个强大的工作流框架，它基于Pregel算法和Actor模型，提供了：

- **灵活的状态管理**：通过状态图管理复杂的工作流状态
- **高效的并发执行**：支持节点间的并行处理
- **强大的通道系统**：多种通信模式满足不同需求
- **完善的状态持久化**：支持工作流的中断和恢复
- **丰富的错误处理**：重试和缓存机制提高可靠性

通过掌握这些基础概念，你可以构建出高效、可靠的AI应用工作流。

## 1.8 运行时系统 (Runtime System)

### 1.8.1 Runtime对象

LangGraph v0.6+ 引入了Runtime对象，它提供了运行时的上下文和工具：

```python
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    session_id: str

def node_with_runtime(state: State, runtime: Runtime[Context]) -> dict:
    """使用Runtime的节点"""
    # 访问上下文
    user_id = runtime.context.user_id
    
    # 访问存储
    if runtime.store:
        user_data = runtime.store.get(("users",), user_id)
        if user_data:
            return {"user_info": user_data.value}
    
    return {"user_id": user_id}
```

### 1.8.2 存储系统 (Store System)

LangGraph提供了灵活的存储系统：

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

# 内存存储
memory_store = InMemoryStore()

# PostgreSQL存储
postgres_store = PostgresStore(
    connection_string="postgresql://user:pass@localhost/db"
)

# 使用存储
memory_store.put(("users",), "user_123", {"name": "Alice"})
user_data = memory_store.get(("users",), "user_123")
```

## 1.9 托管值 (Managed Values)

### 1.9.1 IsLastStep

`IsLastStep`是一个托管值，用于判断当前是否为最后一步：

```python
from langgraph.managed import IsLastStep
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: list
    is_final: Annotated[bool, IsLastStep]

def conditional_node(state: State) -> dict:
    """条件节点"""
    if state.get("is_final"):
        return {"status": "finished"}
    else:
        return {"status": "continue"}
```

### 1.9.2 RemainingSteps

`RemainingSteps`提供剩余步数信息：

```python
from langgraph.managed import RemainingSteps

class State(TypedDict):
    messages: list
    remaining: Annotated[int, RemainingSteps]

def adaptive_node(state: State) -> dict:
    """自适应节点"""
    remaining = state.get("remaining", 0)
    
    if remaining <= 2:
        return {"strategy": "final_summary"}
    else:
        return {"strategy": "detailed_analysis"}
```

## 1.10 开发环境准备

### 1.10.1 安装依赖

```bash
# 安装 LangGraph
pip install langgraph

# 安装相关依赖
pip install langchain
pip install langchain-openai
pip install pydantic
pip install typing-extensions
```

### 1.10.2 基本导入

```python
# 基本导入
from langgraph.graph import StateGraph
from langgraph.func import entrypoint, task
from langgraph.channels import LastValue, EphemeralValue
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
import asyncio
```

## 1.11 第一个完整示例

让我们创建一个完整的AI助手工作流：

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated

# 定义状态
class AssistantState(TypedDict):
    user_input: str
    processed_response: str
    human_feedback: str
    final_answer: str

# 任务定义
@task
def process_user_input(input_text: str) -> str:
    """处理用户输入"""
    return f"Processed: {input_text}"

@task
def generate_response(processed_input: str) -> str:
    """生成AI响应"""
    return f"AI Response to: {processed_input}"

@task
def validate_response(response: str) -> bool:
    """验证响应质量"""
    return len(response) > 10

# 主工作流
@entrypoint(checkpointer=InMemorySaver())
def ai_assistant_workflow(user_input: str) -> dict:
    """AI助手工作流"""
    # 1. 处理用户输入
    processed = process_user_input(user_input)
    
    # 2. 生成AI响应
    ai_response = generate_response(processed)
    
    # 3. 验证响应
    is_valid = validate_response(ai_response)
    
    if not is_valid:
        # 中断等待人工干预
        feedback = interrupt({
            "question": "响应质量不佳，请提供改进建议",
            "response": ai_response
        })
        
        # 基于反馈重新生成
        improved_response = generate_response(f"{processed} [改进: {feedback}]")
        return {
            "original_input": user_input,
            "final_response": improved_response,
            "human_feedback": feedback
        }
    
    return {
        "original_input": user_input,
        "final_response": ai_response,
        "human_feedback": None
    }

# 使用工作流
config = {"configurable": {"thread_id": "user_123"}}

# 正常执行
result = ai_assistant_workflow.invoke("Hello, how are you?", config)

# 如果需要人工干预，可以恢复工作流
# resume_command = Command(resume="请使用更友好的语气")
# result = ai_assistant_workflow.invoke(resume_command, config)
```

## 1.12 常见问题与解决方案

### 问题1：状态更新错误

```python
# 错误示例
def wrong_node(state: State) -> dict:
    return {"messages": "Hello"}  # 错误：应该是列表

# 正确示例
def correct_node(state: State) -> dict:
    return {"messages": [("user", "Hello")]}
```

### 问题2：循环依赖

```python
# 避免无限循环
def safe_agent(state: State) -> dict:
    messages = state["messages"]
    
    # 检查是否应该停止
    if len(messages) > 10:
        return {"status": "finished"}
    
    return {"messages": [("assistant", "Continue...")]}
```

### 问题3：类型错误

```python
# 确保状态定义正确
class State(TypedDict):
    messages: Annotated[list, "messages"]  # 使用正确的注解
    # 不要使用: list  # 这样不会自动合并
```

### 问题4：Runtime使用错误

```python
# 错误示例
def wrong_runtime_usage(state: State, runtime: Runtime) -> dict:
    user_id = runtime.context.user_id  # 可能为None

# 正确示例
def correct_runtime_usage(state: State, runtime: Runtime[Context]) -> dict:
    user_id = runtime.context.user_id if runtime.context else "unknown"
    return {"user_id": user_id}
```

## 总结

第1部分介绍了LangGraph的基础概念和最新特性：

1. **核心组件**：StateGraph、Channel、Pregel、Runtime
2. **API架构**：函数式API、状态图API
3. **通道系统**：多种通道类型和通信模式
4. **检查点系统**：状态持久化和恢复
5. **执行模型**：BSP模型、流式执行、中断恢复
6. **错误处理**：重试策略、缓存策略
7. **运行时系统**：Runtime对象、存储系统
8. **托管值**：IsLastStep、RemainingSteps
9. **最佳实践**：状态设计、节点设计、性能优化
10. **完整示例**：AI助手工作流
11. **常见问题**：状态更新、循环依赖、类型错误

在下一部分中，我们将深入学习如何构建更复杂的多Agent系统，包括条件分支、循环控制、工具集成等高级特性。 