# LangGraph 设计模式深度分析

## 概述

LangGraph采用了多种经典设计模式，构建了一个高度模块化、可扩展的工作流框架。本文档基于源码分析，详细解析LangGraph中的核心设计模式。

## 1. Pregel算法模式

### 1.1 核心思想
LangGraph基于Google的Pregel算法，实现了Bulk Synchronous Parallel (BSP) 模型。

```python
# langgraph/libs/langgraph/langgraph/pregel/_algo.py
class PregelTaskWrites(NamedTuple):
    """Pregel任务写入的抽象"""
    path: tuple[str | int | tuple, ...]
    name: str
    triggers: Sequence[str]
    writes: Sequence[tuple[str, Any]]
    metadata: dict[str, Any]
```

### 1.2 BSP执行模型
```python
# 伪代码展示BSP的三个阶段
def execute_bsp_step():
    # 1. Plan阶段：确定要执行的节点
    nodes_to_execute = plan_nodes()
    
    # 2. Execution阶段：并行执行所有选中的节点
    results = execute_nodes_parallel(nodes_to_execute)
    
    # 3. Update阶段：更新通道中的值
    update_channels(results)
```

### 1.3 实现细节
```python
# langgraph/libs/langgraph/langgraph/pregel/_algo.py
def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
    """应用写入操作到检查点和通道"""
    
    # 按路径排序任务，确保确定性顺序
    tasks = sorted(tasks, key=lambda t: task_path_str(t.path[:3]))
    
    # 更新版本信息
    for task in tasks:
        checkpoint["versions_seen"].setdefault(task.name, {}).update(
            {
                chan: checkpoint["channel_versions"][chan]
                for chan in task.triggers
                if chan in checkpoint["channel_versions"]
            }
        )
    
    # 应用写入到通道
    updated_channels: set[str] = set()
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            if channels[chan].update(vals) and next_version is not None:
                checkpoint["channel_versions"][chan] = next_version
                updated_channels.add(chan)
    
    return updated_channels
```

## 2. Actor模型模式

### 2.1 Actor抽象
每个节点都是一个独立的Actor，具有以下特性：

```python
# langgraph/libs/langgraph/langgraph/pregel/_read.py
class PregelNode:
    """Pregel节点，实现Actor模式"""
    
    def __init__(
        self,
        channels: str | list[str],
        triggers: list[str],
        tags: list[str],
        metadata: dict[str, Any],
        bound: Runnable,
        retry_policy: list[RetryPolicy],
        cache_policy: CachePolicy | None,
    ):
        self.channels = channels
        self.triggers = triggers
        self.tags = tags
        self.metadata = metadata
        self.bound = bound
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
```

### 2.2 消息传递机制
```python
# Actor间的消息传递通过Channel实现
class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    """通道基类，实现Actor间的消息传递"""
    
    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool:
        """更新通道值"""
        pass
    
    @abstractmethod
    def get(self) -> Value:
        """获取通道值"""
        pass
    
    def consume(self) -> bool:
        """消费通道值，防止重复使用"""
        return False
```

### 2.3 Actor生命周期
```python
# Actor的生命周期管理
def prepare_single_task(
    task: PregelTask,
    channels: Mapping[str, BaseChannel],
    managed: Mapping[str, ManagedValueSpec],
    scratchpad: PregelScratchpad,
    select: Sequence[str] | None,
) -> PregelExecutableTask:
    """准备单个任务（Actor）的执行"""
    
    # 1. 读取输入通道
    values = read_channels(local_channels, select)
    
    # 2. 添加管理值
    if managed_keys:
        values.update({k: managed[k].get(scratchpad) for k in managed_keys})
    
    # 3. 创建可执行任务
    return PregelExecutableTask(
        path=task.path,
        name=task.name,
        bound=task.bound,
        values=values,
        writes=task.writes,
        triggers=task.triggers,
        metadata=task.metadata,
    )
```

## 3. 观察者模式

### 3.1 通道订阅机制
```python
# langgraph/libs/langgraph/langgraph/pregel/_read.py
def subscribe_to(
    self,
    *channels: str,
    read: bool = True,
) -> Self:
    """订阅通道，实现观察者模式"""
    
    if read:
        if not self._channels:
            self._channels = list(channels)
        else:
            self._channels.extend(channels)
    
    if isinstance(channels, str):
        self._triggers.append(channels)
    else:
        self._triggers.extend(channels)
    
    return self
```

### 3.2 触发机制
```python
# 当通道更新时，自动触发订阅的节点
def prepare_next_tasks(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    trigger_to_nodes: Mapping[str, Sequence[str]],
    get_next_version: GetNextVersion | None,
) -> list[PregelTask]:
    """准备下一轮要执行的任务"""
    
    # 检查哪些通道有更新
    updated_channels = {
        chan for chan, channel in channels.items()
        if channel.is_available()
    }
    
    # 找到被触发的节点
    triggered_nodes = set()
    for chan in updated_channels:
        if chan in trigger_to_nodes:
            triggered_nodes.update(trigger_to_nodes[chan])
    
    # 创建任务
    tasks = []
    for node_name in triggered_nodes:
        task = create_task_for_node(node_name, checkpoint)
        tasks.append(task)
    
    return tasks
```

## 4. 策略模式

### 4.1 通道策略
```python
# 不同的通道类型实现不同的策略
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """LastValue策略：只保留最后一个值"""
    
    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False
        if len(values) != 1:
            raise InvalidUpdateError("Can receive only one value per step")
        self.value = values[-1]
        return True

class Topic(Generic[Value], BaseChannel[Sequence[Value], Union[Value, list[Value]], list[Value]]):
    """Topic策略：累积多个值"""
    
    def update(self, values: Sequence[Value | list[Value]]) -> bool:
        updated = False
        if not self.accumulate:
            updated = bool(self.values)
            self.values = list[Value]()
        if flat_values := tuple(_flatten(values)):
            updated = True
            self.values.extend(flat_values)
        return updated
```

### 4.2 重试策略
```python
# langgraph/libs/langgraph/langgraph/pregel/_retry.py
class RetryPolicy:
    """重试策略基类"""
    
    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """判断是否应该重试"""
        pass
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """获取重试延迟"""
        pass

class ExponentialBackoff(RetryPolicy):
    """指数退避策略"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2 ** (attempt - 1))
        return min(delay, self.max_delay)
```

## 5. 工厂模式

### 5.1 节点工厂
```python
# langgraph/libs/langgraph/langgraph/pregel/_read.py
def coerce_to_runnable(
    obj: RunnableLike,
    name: str | None = None,
    trace: bool = True,
) -> Runnable:
    """将对象转换为可运行对象"""
    
    if isinstance(obj, Runnable):
        return obj
    elif callable(obj):
        return RunnableLambda(obj, name=name, trace=trace)
    else:
        raise TypeError(f"Cannot coerce {type(obj)} to Runnable")
```

### 5.2 通道工厂
```python
# langgraph/libs/langgraph/langgraph/graph/state.py
def _create_channels_from_schema(schema: type[Any]) -> dict[str, BaseChannel]:
    """从模式创建通道"""
    
    channels = {}
    for field_name, field_info in schema.__annotations__.items():
        if hasattr(field_info, "__metadata__"):
            # 有注解的字段
            metadata = field_info.__metadata__
            if metadata and hasattr(metadata[0], "__call__"):
                # 创建自定义通道
                channel = metadata[0](field_info.__args__[0])
                channels[field_name] = channel
        else:
            # 简单字段，使用LastValue
            channels[field_name] = LastValue(field_info)
    
    return channels
```

## 6. 状态模式

### 6.1 状态定义
```python
# 状态通过TypedDict定义
class State(TypedDict):
    messages: Annotated[list, LastValue]
    current_step: str
    data: dict

# 状态转换通过节点函数实现
def state_transition(state: State) -> dict:
    """状态转换函数"""
    if state["current_step"] == "start":
        return {"current_step": "processing"}
    elif state["current_step"] == "processing":
        return {"current_step": "complete"}
    else:
        return {"current_step": "error"}
```

### 6.2 状态持久化
```python
# langgraph/libs/langgraph/langgraph/checkpoint/base.py
class Checkpoint:
    """检查点，实现状态持久化"""
    
    def __init__(self):
        self.channel_versions: ChannelVersions = {}
        self.versions_seen: dict[str, dict[str, int]] = {}
        self.metadata: CheckpointMetadata = {}
        self.channels: dict[str, Any] = {}
```

## 7. 命令模式

### 7.1 命令抽象
```python
# langgraph/libs/langgraph/langgraph/types.py
class Command:
    """命令基类"""
    
    @property
    @abstractmethod
    def type(self) -> str:
        """命令类型"""
        pass

class Send(Command):
    """发送命令"""
    
    def __init__(self, node: str, value: Any):
        self.node = node
        self.value = value
    
    @property
    def type(self) -> str:
        return "send"
```

### 7.2 命令处理
```python
# langgraph/libs/langgraph/langgraph/pregel/_io.py
def map_command(
    command: Command,
    channels: Mapping[str, BaseChannel],
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> PregelTaskWrites:
    """映射命令到任务写入"""
    
    if isinstance(command, Send):
        return PregelTaskWrites(
            path=(command.node,),
            name=command.node,
            triggers=[],
            writes=[(command.node, command.value)],
            metadata={},
        )
    else:
        raise ValueError(f"Unknown command type: {command.type}")
```

## 8. 模板方法模式

### 8.1 执行模板
```python
# langgraph/libs/langgraph/langgraph/pregel/_loop.py
class PregelLoop:
    """Pregel执行循环，实现模板方法模式"""
    
    def run(self, input_data: Any) -> Any:
        """执行模板方法"""
        # 1. 初始化
        self.initialize()
        
        # 2. 执行循环
        while not self.should_stop():
            self.execute_step()
        
        # 3. 清理
        self.cleanup()
        
        return self.get_result()
    
    def initialize(self):
        """初始化步骤"""
        pass
    
    def should_stop(self) -> bool:
        """判断是否应该停止"""
        pass
    
    def execute_step(self):
        """执行单个步骤"""
        pass
    
    def cleanup(self):
        """清理步骤"""
        pass
    
    def get_result(self) -> Any:
        """获取结果"""
        pass
```

## 9. 责任链模式

### 9.1 节点链
```python
# 节点通过边连接形成责任链
def build_chain():
    """构建责任链"""
    graph = StateGraph(State)
    
    # 添加节点
    graph.add_node("validator", validate_input)
    graph.add_node("processor", process_data)
    graph.add_node("output", generate_output)
    
    # 连接链
    graph.add_edge("validator", "processor")
    graph.add_edge("processor", "output")
    
    return graph.compile()

def validate_input(state: State) -> dict:
    """验证输入"""
    if not state["input"]:
        return {"error": "Invalid input"}
    return {"validated": True}

def process_data(state: State) -> dict:
    """处理数据"""
    if state.get("error"):
        return {"error": state["error"]}
    return {"processed": state["input"] * 2}
```

## 10. 适配器模式

### 10.1 外部系统适配
```python
# 适配外部工具和API
class ToolAdapter:
    """工具适配器"""
    
    def __init__(self, tool: BaseTool):
        self.tool = tool
    
    def adapt_to_node(self) -> Callable:
        """适配为节点函数"""
        def node_function(state: State) -> dict:
            try:
                result = self.tool.run(state["input"])
                return {"output": result}
            except Exception as e:
                return {"error": str(e)}
        return node_function

# 使用适配器
search_tool = DuckDuckGoSearchRun()
search_node = ToolAdapter(search_tool).adapt_to_node()
graph.add_node("search", search_node)
```

## 总结

LangGraph通过巧妙组合多种设计模式，构建了一个高度模块化、可扩展的工作流框架：

1. **Pregel算法模式**：提供并行执行的基础
2. **Actor模型模式**：实现节点间的松耦合通信
3. **观察者模式**：实现事件驱动的节点触发
4. **策略模式**：支持不同的通道和重试策略
5. **工厂模式**：简化对象创建
6. **状态模式**：管理复杂的状态转换
7. **命令模式**：支持动态控制流
8. **模板方法模式**：定义执行流程模板
9. **责任链模式**：构建节点执行链
10. **适配器模式**：集成外部系统

这些设计模式的组合使得LangGraph能够：
- 支持复杂的并行计算
- 提供灵活的状态管理
- 实现高度可扩展的架构
- 保持代码的清晰和可维护性
