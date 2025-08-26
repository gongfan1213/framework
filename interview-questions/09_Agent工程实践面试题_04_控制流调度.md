# Agent工程实践面试题 - 控制流与调度能力

## 1. 控制流与调度能力分析

### 1.1 详细分析项目的控制流与调度能力

**面试题：请详细分析你的项目中控制流与调度能力，是否具备DAG工作流设计能力？任务能否幂等执行？是否支持失败回滚、优先级调度、异步协同等机制？**

**答案要点：**

**1. DAG工作流设计：**
```python
class DAGWorkflowEngine:
    def __init__(self):
        self.workflows = {}
        self.execution_engine = WorkflowExecutionEngine()
        self.scheduler = TaskScheduler()
        self.monitor = WorkflowMonitor()
    
    def create_workflow(self, workflow_id, nodes, edges):
        # 创建DAG工作流
        workflow = {
            "id": workflow_id,
            "nodes": nodes,
            "edges": edges,
            "status": "created",
            "created_at": time.time()
        }
        
        # 验证DAG结构
        if self._validate_dag_structure(nodes, edges):
            self.workflows[workflow_id] = workflow
            return workflow
        else:
            raise ValueError("Invalid DAG structure: contains cycles")
    
    def execute_workflow(self, workflow_id, input_data):
        # 执行工作流
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # 创建执行实例
        execution_id = self._generate_execution_id()
        execution = {
            "id": execution_id,
            "workflow_id": workflow_id,
            "input_data": input_data,
            "status": "running",
            "start_time": time.time(),
            "completed_nodes": set(),
            "failed_nodes": set(),
            "node_results": {}
        }
        
        # 开始执行
        return self.execution_engine.execute(execution, workflow)
    
    def _validate_dag_structure(self, nodes, edges):
        # 验证DAG结构（无环）
        graph = {node: [] for node in nodes}
        for edge in edges:
            graph[edge["from"]].append(edge["to"])
        
        # 使用拓扑排序检测环
        return self._topological_sort(graph) is not None
    
    def _topological_sort(self, graph):
        # 拓扑排序
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(graph) else None
```

**2. 任务调度器：**
```python
class TaskScheduler:
    def __init__(self):
        self.task_queue = PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.worker_pool = WorkerPool()
        self.scheduling_policies = {
            "fifo": FIFOScheduler(),
            "priority": PriorityScheduler(),
            "fair": FairScheduler(),
            "deadline": DeadlineScheduler()
        }
    
    def schedule_task(self, task, policy="priority"):
        # 调度任务
        scheduler = self.scheduling_policies[policy]
        scheduled_task = scheduler.schedule(task)
        
        # 添加到队列
        self.task_queue.put(scheduled_task)
        
        return scheduled_task["id"]
    
    def execute_tasks(self):
        # 执行任务
        while not self.task_queue.empty():
            task = self.task_queue.get()
            
            # 检查任务依赖
            if self._check_dependencies(task):
                # 分配worker
                worker = self.worker_pool.get_available_worker()
                if worker:
                    # 执行任务
                    self._execute_task_on_worker(task, worker)
                else:
                    # 重新入队
                    self.task_queue.put(task)
            else:
                # 依赖未满足，重新入队
                self.task_queue.put(task)
    
    def _check_dependencies(self, task):
        # 检查任务依赖
        dependencies = task.get("dependencies", [])
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _execute_task_on_worker(self, task, worker):
        # 在worker上执行任务
        task_id = task["id"]
        self.running_tasks[task_id] = {
            "task": task,
            "worker": worker,
            "start_time": time.time()
        }
        
        # 异步执行
        worker.execute_task(task, self._task_completion_callback)
    
    def _task_completion_callback(self, task_id, result, status):
        # 任务完成回调
        if status == "success":
            self.completed_tasks[task_id] = result
        else:
            self.failed_tasks[task_id] = result
        
        # 从运行中移除
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
```

**3. 幂等执行机制：**
```python
class IdempotentExecutor:
    def __init__(self):
        self.execution_cache = {}
        self.idempotency_keys = {}
        self.cache_ttl = 3600  # 1小时
    
    def execute_idempotent(self, task, idempotency_key=None):
        # 幂等执行
        if idempotency_key:
            # 检查是否已执行
            cached_result = self._get_cached_result(idempotency_key)
            if cached_result:
                return cached_result
        
        # 生成幂等键
        if not idempotency_key:
            idempotency_key = self._generate_idempotency_key(task)
        
        # 执行任务
        try:
            result = self._execute_task(task)
            
            # 缓存结果
            self._cache_result(idempotency_key, result)
            
            return result
        except Exception as e:
            # 记录失败
            self._record_failure(idempotency_key, e)
            raise
    
    def _generate_idempotency_key(self, task):
        # 生成幂等键
        task_content = {
            "function": task.get("function"),
            "parameters": task.get("parameters"),
            "context": task.get("context", {})
        }
        
        # 使用哈希生成唯一键
        content_str = json.dumps(task_content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _get_cached_result(self, idempotency_key):
        # 获取缓存结果
        if idempotency_key in self.execution_cache:
            cached_item = self.execution_cache[idempotency_key]
            
            # 检查是否过期
            if time.time() - cached_item["timestamp"] < self.cache_ttl:
                return cached_item["result"]
            else:
                # 清理过期缓存
                del self.execution_cache[idempotency_key]
        
        return None
    
    def _cache_result(self, idempotency_key, result):
        # 缓存结果
        self.execution_cache[idempotency_key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def _record_failure(self, idempotency_key, error):
        # 记录失败
        self.idempotency_keys[idempotency_key] = {
            "status": "failed",
            "error": str(error),
            "timestamp": time.time()
        }
```

### 1.2 失败回滚机制

**面试题：请详细描述你的项目中失败回滚机制的设计和实现？**

**答案要点：**

**1. 回滚策略设计：**
```python
class RollbackManager:
    def __init__(self):
        self.rollback_strategies = {
            "full_rollback": FullRollbackStrategy(),
            "partial_rollback": PartialRollbackStrategy(),
            "compensating_actions": CompensatingActionsStrategy(),
            "checkpoint_rollback": CheckpointRollbackStrategy()
        }
        self.checkpoint_manager = CheckpointManager()
        self.compensation_manager = CompensationManager()
    
    def handle_failure(self, execution_context, error):
        # 处理失败并执行回滚
        # 1. 分析失败影响
        impact_analysis = self._analyze_failure_impact(execution_context, error)
        
        # 2. 选择回滚策略
        rollback_strategy = self._select_rollback_strategy(impact_analysis)
        
        # 3. 执行回滚
        rollback_result = rollback_strategy.execute(execution_context, impact_analysis)
        
        # 4. 记录回滚信息
        self._record_rollback(execution_context, error, rollback_result)
        
        return rollback_result
    
    def _analyze_failure_impact(self, execution_context, error):
        # 分析失败影响
        impact = {
            "affected_nodes": [],
            "data_consistency": "unknown",
            "rollback_complexity": "low",
            "compensation_required": False
        }
        
        # 分析受影响的节点
        for node_id, node_status in execution_context["node_status"].items():
            if node_status == "running" or node_status == "completed":
                impact["affected_nodes"].append(node_id)
        
        # 评估数据一致性
        impact["data_consistency"] = self._assess_data_consistency(execution_context)
        
        # 评估回滚复杂度
        impact["rollback_complexity"] = self._assess_rollback_complexity(impact["affected_nodes"])
        
        # 判断是否需要补偿操作
        impact["compensation_required"] = self._assess_compensation_need(execution_context)
        
        return impact
    
    def _select_rollback_strategy(self, impact_analysis):
        # 选择回滚策略
        if impact_analysis["rollback_complexity"] == "high":
            return self.rollback_strategies["checkpoint_rollback"]
        elif impact_analysis["compensation_required"]:
            return self.rollback_strategies["compensating_actions"]
        elif len(impact_analysis["affected_nodes"]) > 5:
            return self.rollback_strategies["partial_rollback"]
        else:
            return self.rollback_strategies["full_rollback"]
```

**2. 检查点回滚策略：**
```python
class CheckpointRollbackStrategy:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def execute(self, execution_context, impact_analysis):
        # 执行检查点回滚
        try:
            # 1. 找到最近的检查点
            checkpoint = self.checkpoint_manager.get_latest_checkpoint(execution_context["workflow_id"])
            
            if not checkpoint:
                raise RollbackError("No checkpoint available for rollback")
            
            # 2. 恢复到检查点状态
            restored_state = self._restore_from_checkpoint(checkpoint)
            
            # 3. 重新执行未完成的任务
            remaining_tasks = self._identify_remaining_tasks(execution_context, checkpoint)
            
            # 4. 重新调度任务
            self._reschedule_tasks(remaining_tasks, restored_state)
            
            return {
                "status": "success",
                "strategy": "checkpoint_rollback",
                "checkpoint_id": checkpoint["id"],
                "restored_state": restored_state,
                "remaining_tasks": len(remaining_tasks)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "strategy": "checkpoint_rollback",
                "error": str(e)
            }
    
    def _restore_from_checkpoint(self, checkpoint):
        # 从检查点恢复状态
        restored_state = {
            "workflow_state": checkpoint["workflow_state"],
            "data_state": checkpoint["data_state"],
            "node_states": checkpoint["node_states"]
        }
        
        # 恢复数据状态
        self._restore_data_state(checkpoint["data_state"])
        
        # 恢复节点状态
        self._restore_node_states(checkpoint["node_states"])
        
        return restored_state
    
    def _identify_remaining_tasks(self, execution_context, checkpoint):
        # 识别剩余任务
        completed_nodes = set(checkpoint["completed_nodes"])
        all_nodes = set(execution_context["workflow"]["nodes"])
        
        remaining_nodes = all_nodes - completed_nodes
        remaining_tasks = []
        
        for node_id in remaining_nodes:
            node = execution_context["workflow"]["nodes"][node_id]
            remaining_tasks.append({
                "node_id": node_id,
                "task": node["task"],
                "dependencies": node.get("dependencies", [])
            })
        
        return remaining_tasks
```

**3. 补偿操作策略：**
```python
class CompensatingActionsStrategy:
    def __init__(self):
        self.compensation_manager = CompensationManager()
    
    def execute(self, execution_context, impact_analysis):
        # 执行补偿操作
        try:
            # 1. 识别需要补偿的操作
            compensation_actions = self._identify_compensation_actions(execution_context)
            
            # 2. 按逆序执行补偿操作
            compensation_results = []
            for action in reversed(compensation_actions):
                result = self.compensation_manager.execute_compensation(action)
                compensation_results.append(result)
            
            # 3. 验证补偿结果
            compensation_status = self._verify_compensation(compensation_results)
            
            return {
                "status": "success",
                "strategy": "compensating_actions",
                "compensation_actions": len(compensation_actions),
                "compensation_results": compensation_results,
                "verification_status": compensation_status
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "strategy": "compensating_actions",
                "error": str(e)
            }
    
    def _identify_compensation_actions(self, execution_context):
        # 识别补偿操作
        compensation_actions = []
        
        for node_id, node_status in execution_context["node_status"].items():
            if node_status == "completed":
                node = execution_context["workflow"]["nodes"][node_id]
                
                # 检查是否有补偿操作
                if "compensation" in node:
                    compensation_actions.append({
                        "node_id": node_id,
                        "action": node["compensation"],
                        "context": execution_context["node_results"].get(node_id, {})
                    })
        
        return compensation_actions

class CompensationManager:
    def __init__(self):
        self.compensation_handlers = {
            "database_rollback": DatabaseRollbackHandler(),
            "api_compensation": APICompensationHandler(),
            "file_cleanup": FileCleanupHandler(),
            "notification_compensation": NotificationCompensationHandler()
        }
    
    def execute_compensation(self, compensation_action):
        # 执行补偿操作
        action_type = compensation_action["action"]["type"]
        handler = self.compensation_handlers.get(action_type)
        
        if not handler:
            raise CompensationError(f"Unknown compensation type: {action_type}")
        
        try:
            result = handler.execute(compensation_action)
            return {
                "action_id": compensation_action["node_id"],
                "type": action_type,
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "action_id": compensation_action["node_id"],
                "type": action_type,
                "status": "failed",
                "error": str(e)
            }

class DatabaseRollbackHandler:
    def execute(self, compensation_action):
        # 数据库回滚处理
        action_config = compensation_action["action"]["config"]
        context = compensation_action["context"]
        
        # 执行数据库回滚
        if "transaction_id" in context:
            return self._rollback_transaction(context["transaction_id"])
        elif "backup_data" in context:
            return self._restore_from_backup(context["backup_data"])
        else:
            raise CompensationError("No rollback data available")
    
    def _rollback_transaction(self, transaction_id):
        # 回滚事务
        # 这里实现具体的事务回滚逻辑
        pass
    
    def _restore_from_backup(self, backup_data):
        # 从备份恢复
        # 这里实现具体的备份恢复逻辑
        pass
```

### 1.3 优先级调度机制

**面试题：请详细描述你的项目中优先级调度机制的设计和实现？**

**答案要点：**

**1. 优先级调度器：**
```python
class PriorityScheduler:
    def __init__(self):
        self.priority_queues = {
            "critical": PriorityQueue(),
            "high": PriorityQueue(),
            "normal": PriorityQueue(),
            "low": PriorityQueue()
        }
        self.priority_weights = {
            "critical": 4,
            "high": 3,
            "normal": 2,
            "low": 1
        }
        self.resource_manager = ResourceManager()
    
    def schedule(self, task):
        # 优先级调度
        priority = self._calculate_priority(task)
        task["priority"] = priority
        
        # 添加到对应优先级队列
        queue = self.priority_queues[priority]
        queue.put((self._get_priority_score(task), task))
        
        return task
    
    def get_next_task(self):
        # 获取下一个任务
        # 按优先级顺序检查队列
        for priority in ["critical", "high", "normal", "low"]:
            queue = self.priority_queues[priority]
            if not queue.empty():
                score, task = queue.get()
                return task
        
        return None
    
    def _calculate_priority(self, task):
        # 计算任务优先级
        priority_factors = {
            "deadline": self._calculate_deadline_priority(task),
            "business_value": self._calculate_business_value_priority(task),
            "resource_requirement": self._calculate_resource_priority(task),
            "dependency": self._calculate_dependency_priority(task),
            "user_priority": self._calculate_user_priority(task)
        }
        
        # 加权计算总优先级
        total_score = sum(
            factor_score * self._get_factor_weight(factor_name)
            for factor_name, factor_score in priority_factors.items()
        )
        
        # 映射到优先级级别
        if total_score >= 0.8:
            return "critical"
        elif total_score >= 0.6:
            return "high"
        elif total_score >= 0.4:
            return "normal"
        else:
            return "low"
    
    def _calculate_deadline_priority(self, task):
        # 计算截止时间优先级
        deadline = task.get("deadline")
        if not deadline:
            return 0.5
        
        current_time = time.time()
        time_remaining = deadline - current_time
        
        if time_remaining <= 0:
            return 1.0  # 已超时
        elif time_remaining <= 300:  # 5分钟内
            return 0.9
        elif time_remaining <= 3600:  # 1小时内
            return 0.7
        elif time_remaining <= 86400:  # 1天内
            return 0.5
        else:
            return 0.3
    
    def _calculate_business_value_priority(self, task):
        # 计算业务价值优先级
        business_value = task.get("business_value", 0)
        max_value = task.get("max_business_value", 100)
        
        return min(business_value / max_value, 1.0)
    
    def _calculate_resource_priority(self, task):
        # 计算资源需求优先级
        required_resources = task.get("required_resources", {})
        available_resources = self.resource_manager.get_available_resources()
        
        # 资源稀缺性越高，优先级越高
        scarcity_score = 0
        for resource_type, required_amount in required_resources.items():
            available_amount = available_resources.get(resource_type, 0)
            if available_amount > 0:
                scarcity = required_amount / available_amount
                scarcity_score = max(scarcity_score, scarcity)
        
        return min(scarcity_score, 1.0)
    
    def _get_priority_score(self, task):
        # 获取优先级分数
        base_score = self.priority_weights[task["priority"]]
        
        # 考虑子因素
        sub_factors = {
            "creation_time": task.get("creation_time", time.time()),
            "retry_count": task.get("retry_count", 0),
            "user_level": task.get("user_level", "normal")
        }
        
        # 计算综合分数
        score = base_score
        
        # 时间因子（越早创建分数越高）
        time_factor = 1.0 / (time.time() - sub_factors["creation_time"] + 1)
        score += time_factor * 0.1
        
        # 重试因子（重试次数越多分数越高）
        retry_factor = sub_factors["retry_count"] * 0.2
        score += retry_factor
        
        # 用户级别因子
        user_level_weights = {"vip": 0.5, "premium": 0.3, "normal": 0.1}
        user_factor = user_level_weights.get(sub_factors["user_level"], 0.1)
        score += user_factor
        
        return score
```

**2. 资源管理器：**
```python
class ResourceManager:
    def __init__(self):
        self.resources = {
            "cpu": {"total": 100, "used": 0, "reserved": 0},
            "memory": {"total": 1024, "used": 0, "reserved": 0},
            "gpu": {"total": 4, "used": 0, "reserved": 0},
            "network": {"total": 1000, "used": 0, "reserved": 0}
        }
        self.resource_locks = {}
        self.reservation_queue = PriorityQueue()
    
    def allocate_resources(self, task):
        # 分配资源
        required_resources = task.get("required_resources", {})
        
        # 检查资源可用性
        if not self._check_resource_availability(required_resources):
            # 资源不足，加入等待队列
            self.reservation_queue.put((task["priority_score"], task))
            return False
        
        # 分配资源
        allocation_id = self._generate_allocation_id()
        allocation = {
            "id": allocation_id,
            "task_id": task["id"],
            "resources": required_resources,
            "allocation_time": time.time()
        }
        
        # 更新资源使用情况
        for resource_type, amount in required_resources.items():
            self.resources[resource_type]["used"] += amount
        
        self.resource_locks[allocation_id] = allocation
        task["resource_allocation_id"] = allocation_id
        
        return True
    
    def release_resources(self, allocation_id):
        # 释放资源
        if allocation_id in self.resource_locks:
            allocation = self.resource_locks[allocation_id]
            
            # 释放资源
            for resource_type, amount in allocation["resources"].items():
                self.resources[resource_type]["used"] -= amount
            
            # 移除分配记录
            del self.resource_locks[allocation_id]
            
            # 处理等待队列
            self._process_reservation_queue()
    
    def _check_resource_availability(self, required_resources):
        # 检查资源可用性
        for resource_type, required_amount in required_resources.items():
            if resource_type not in self.resources:
                return False
            
            available = (self.resources[resource_type]["total"] - 
                        self.resources[resource_type]["used"] - 
                        self.resources[resource_type]["reserved"])
            
            if available < required_amount:
                return False
        
        return True
    
    def _process_reservation_queue(self):
        # 处理资源预留队列
        while not self.reservation_queue.empty():
            priority_score, task = self.reservation_queue.get()
            
            if self.allocate_resources(task):
                # 资源分配成功，重新调度任务
                self.scheduler.reschedule_task(task)
                break
            else:
                # 资源仍然不足，重新入队
                self.reservation_queue.put((priority_score, task))
                break
```

### 1.4 异步协同机制

**面试题：请详细描述你的项目中异步协同机制的设计和实现？**

**答案要点：**

**1. 异步任务执行器：**
```python
class AsyncTaskExecutor:
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.max_workers = 10
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
    
    async def start(self):
        # 启动异步执行器
        # 创建worker协程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def submit_task(self, task):
        # 提交异步任务
        task_id = self._generate_task_id()
        task["id"] = task_id
        task["status"] = "pending"
        task["submission_time"] = time.time()
        
        await self.task_queue.put(task)
        return task_id
    
    async def _worker(self, worker_id):
        # 工作协程
        while True:
            try:
                # 获取任务
                task = await self.task_queue.get()
                
                # 执行任务
                await self._execute_task(task, worker_id)
                
                # 标记任务完成
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _execute_task(self, task, worker_id):
        # 执行异步任务
        task_id = task["id"]
        self.running_tasks[task_id] = {
            "task": task,
            "worker_id": worker_id,
            "start_time": time.time()
        }
        
        try:
            # 执行任务
            if task["type"] == "async_function":
                result = await self._execute_async_function(task)
            elif task["type"] == "http_request":
                result = await self._execute_http_request(task)
            elif task["type"] == "database_operation":
                result = await self._execute_database_operation(task)
            else:
                raise ValueError(f"Unknown task type: {task['type']}")
            
            # 记录成功结果
            self.completed_tasks[task_id] = {
                "result": result,
                "completion_time": time.time(),
                "worker_id": worker_id
            }
            
        except Exception as e:
            # 记录失败结果
            self.failed_tasks[task_id] = {
                "error": str(e),
                "failure_time": time.time(),
                "worker_id": worker_id
            }
        
        finally:
            # 清理运行状态
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _execute_async_function(self, task):
        # 执行异步函数
        function_name = task["function"]
        parameters = task.get("parameters", {})
        
        # 动态调用函数
        if hasattr(self, function_name):
            function = getattr(self, function_name)
            if asyncio.iscoroutinefunction(function):
                return await function(**parameters)
            else:
                return function(**parameters)
        else:
            raise ValueError(f"Function not found: {function_name}")
    
    async def _execute_http_request(self, task):
        # 执行HTTP请求
        url = task["url"]
        method = task.get("method", "GET")
        headers = task.get("headers", {})
        data = task.get("data")
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=data) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "body": await response.text()
                }
    
    async def _execute_database_operation(self, task):
        # 执行数据库操作
        operation = task["operation"]
        table = task["table"]
        data = task.get("data", {})
        
        # 这里实现具体的数据库操作
        # 例如使用asyncpg或aiomysql
        pass
```

**2. 异步协同协调器：**
```python
class AsyncCoordinationManager:
    def __init__(self):
        self.coordination_patterns = {
            "fan_out": FanOutPattern(),
            "fan_in": FanInPattern(),
            "pipeline": PipelinePattern(),
            "scatter_gather": ScatterGatherPattern()
        }
        self.async_executor = AsyncTaskExecutor()
        self.result_collector = ResultCollector()
    
    async def coordinate_tasks(self, pattern, tasks, coordination_config):
        # 协调异步任务
        pattern_handler = self.coordination_patterns[pattern]
        return await pattern_handler.execute(tasks, coordination_config)
    
    async def fan_out_execution(self, tasks, max_concurrency=None):
        # Fan-out模式：并行执行多个任务
        if max_concurrency:
            semaphore = asyncio.Semaphore(max_concurrency)
        else:
            semaphore = None
        
        async def execute_with_semaphore(task):
            if semaphore:
                async with semaphore:
                    return await self.async_executor.submit_task(task)
            else:
                return await self.async_executor.submit_task(task)
        
        # 并行提交所有任务
        task_futures = [execute_with_semaphore(task) for task in tasks]
        
        # 等待所有任务完成
        results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        return results
    
    async def fan_in_execution(self, tasks, aggregation_function):
        # Fan-in模式：收集多个任务结果并聚合
        # 执行所有任务
        results = await self.fan_out_execution(tasks)
        
        # 聚合结果
        aggregated_result = aggregation_function(results)
        
        return aggregated_result
    
    async def pipeline_execution(self, stages):
        # Pipeline模式：流水线执行
        stage_results = []
        current_input = None
        
        for stage in stages:
            # 执行当前阶段
            stage_result = await self.async_executor.submit_task({
                **stage,
                "input": current_input
            })
            
            stage_results.append(stage_result)
            current_input = stage_result
        
        return stage_results
    
    async def scatter_gather_execution(self, scatter_task, gather_task, worker_tasks):
        # Scatter-Gather模式
        # 1. Scatter阶段
        scatter_result = await self.async_executor.submit_task(scatter_task)
        
        # 2. 并行执行worker任务
        worker_results = await self.fan_out_execution(worker_tasks)
        
        # 3. Gather阶段
        gather_result = await self.async_executor.submit_task({
            **gather_task,
            "worker_results": worker_results
        })
        
        return {
            "scatter_result": scatter_result,
            "worker_results": worker_results,
            "gather_result": gather_result
        }

class FanOutPattern:
    async def execute(self, tasks, config):
        # Fan-out模式实现
        max_concurrency = config.get("max_concurrency")
        timeout = config.get("timeout")
        
        if timeout:
            return await asyncio.wait_for(
                self._execute_fan_out(tasks, max_concurrency),
                timeout=timeout
            )
        else:
            return await self._execute_fan_out(tasks, max_concurrency)
    
    async def _execute_fan_out(self, tasks, max_concurrency):
        # 实现Fan-out逻辑
        pass

class FanInPattern:
    async def execute(self, tasks, config):
        # Fan-in模式实现
        aggregation_function = config.get("aggregation_function", self._default_aggregation)
        return await self._execute_fan_in(tasks, aggregation_function)
    
    async def _execute_fan_in(self, tasks, aggregation_function):
        # 实现Fan-in逻辑
        pass
    
    def _default_aggregation(self, results):
        # 默认聚合函数
        return results
```

**3. 异步结果收集器：**
```python
class ResultCollector:
    def __init__(self):
        self.collection_strategies = {
            "all": self.collect_all_results,
            "first": self.collect_first_result,
            "majority": self.collect_majority_results,
            "best": self.collect_best_result
        }
    
    async def collect_results(self, task_ids, strategy="all", timeout=None):
        # 收集异步任务结果
        collection_strategy = self.collection_strategies[strategy]
        
        if timeout:
            return await asyncio.wait_for(
                collection_strategy(task_ids),
                timeout=timeout
            )
        else:
            return await collection_strategy(task_ids)
    
    async def collect_all_results(self, task_ids):
        # 收集所有结果
        results = {}
        
        for task_id in task_ids:
            result = await self._wait_for_task_completion(task_id)
            results[task_id] = result
        
        return results
    
    async def collect_first_result(self, task_ids):
        # 收集第一个完成的结果
        tasks = [self._wait_for_task_completion(task_id) for task_id in task_ids]
        
        # 等待第一个完成
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # 取消其他任务
        for task in pending:
            task.cancel()
        
        # 返回第一个结果
        first_result = done.pop().result()
        return first_result
    
    async def collect_majority_results(self, task_ids):
        # 收集多数结果
        majority_threshold = len(task_ids) // 2 + 1
        results = []
        
        tasks = [self._wait_for_task_completion(task_id) for task_id in task_ids]
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            
            if len(results) >= majority_threshold:
                break
        
        return results
    
    async def collect_best_result(self, task_ids, evaluation_function=None):
        # 收集最佳结果
        if not evaluation_function:
            evaluation_function = self._default_evaluation
        
        best_result = None
        best_score = float('-inf')
        
        for task_id in task_ids:
            result = await self._wait_for_task_completion(task_id)
            score = evaluation_function(result)
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    async def _wait_for_task_completion(self, task_id):
        # 等待任务完成
        while True:
            if task_id in self.async_executor.completed_tasks:
                return self.async_executor.completed_tasks[task_id]["result"]
            elif task_id in self.async_executor.failed_tasks:
                raise Exception(self.async_executor.failed_tasks[task_id]["error"])
            
            await asyncio.sleep(0.1)
    
    def _default_evaluation(self, result):
        # 默认评估函数
        if isinstance(result, dict) and "score" in result:
            return result["score"]
        else:
            return 0.0
```
