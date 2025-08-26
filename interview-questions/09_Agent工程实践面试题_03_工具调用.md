# Agent工程实践面试题 - 工具调用与API注入机制

## 1. 工具调用机制分析

### 1.1 详细分析项目的工具调用与API注入机制

**面试题：请详细分析你的项目中工具调用与API注入机制，是否是工具的动态注册？如何解析schema？是否支持工具选择、参数对齐、调用fallback？能否抽象出统一工具执行接口？**

**答案要点：**

**1. 工具调用架构设计：**
```python
class ToolCallSystem:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.schema_parser = SchemaParser()
        self.tool_selector = ToolSelector()
        self.parameter_aligner = ParameterAligner()
        self.execution_engine = ExecutionEngine()
        self.fallback_handler = FallbackHandler()
    
    def execute_tool_call(self, tool_request):
        # 1. 解析工具请求
        parsed_request = self.schema_parser.parse(tool_request)
        
        # 2. 选择合适工具
        selected_tool = self.tool_selector.select(parsed_request)
        
        # 3. 参数对齐
        aligned_params = self.parameter_aligner.align(selected_tool, parsed_request)
        
        # 4. 执行工具调用
        try:
            result = self.execution_engine.execute(selected_tool, aligned_params)
            return result
        except Exception as e:
            # 5. 处理失败情况
            return self.fallback_handler.handle(selected_tool, aligned_params, e)
```

**2. 动态工具注册机制：**
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.tool_categories = {}
        self.tool_metadata = {}
        self.registration_hooks = []
    
    def register_tool(self, tool_name, tool_func, schema, metadata=None):
        # 动态注册工具
        tool_id = self._generate_tool_id(tool_name)
        
        # 验证schema
        validated_schema = self._validate_schema(schema)
        
        # 注册工具
        self.tools[tool_id] = {
            "name": tool_name,
            "function": tool_func,
            "schema": validated_schema,
            "metadata": metadata or {},
            "registration_time": time.time(),
            "usage_count": 0
        }
        
        # 分类管理
        category = metadata.get("category", "general") if metadata else "general"
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(tool_id)
        
        # 触发注册钩子
        self._trigger_registration_hooks(tool_id, self.tools[tool_id])
        
        return tool_id
    
    def unregister_tool(self, tool_id):
        # 注销工具
        if tool_id in self.tools:
            tool_info = self.tools[tool_id]
            
            # 从分类中移除
            category = tool_info["metadata"].get("category", "general")
            if category in self.tool_categories and tool_id in self.tool_categories[category]:
                self.tool_categories[category].remove(tool_id)
            
            # 删除工具
            del self.tools[tool_id]
            
            return True
        return False
    
    def get_tool(self, tool_id):
        return self.tools.get(tool_id)
    
    def list_tools(self, category=None):
        if category:
            return [self.tools[tool_id] for tool_id in self.tool_categories.get(category, [])]
        return list(self.tools.values())
    
    def _validate_schema(self, schema):
        # 验证工具schema
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in schema:
                raise ValueError(f"Missing required field: {field}")
        
        # 验证参数schema
        if "parameters" in schema:
            self._validate_parameters_schema(schema["parameters"])
        
        return schema
    
    def _validate_parameters_schema(self, parameters):
        # 验证参数schema
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        if "properties" not in parameters:
            raise ValueError("Parameters must have 'properties' field")
        
        for param_name, param_schema in parameters["properties"].items():
            if "type" not in param_schema:
                raise ValueError(f"Parameter {param_name} must have 'type' field")
```

**3. Schema解析机制：**
```python
class SchemaParser:
    def __init__(self):
        self.parsers = {
            "json": JSONSchemaParser(),
            "openapi": OpenAPISchemaParser(),
            "custom": CustomSchemaParser()
        }
        self.schema_cache = {}
    
    def parse(self, tool_request):
        # 解析工具请求
        schema_type = self._detect_schema_type(tool_request)
        parser = self.parsers.get(schema_type)
        
        if not parser:
            raise ValueError(f"Unsupported schema type: {schema_type}")
        
        # 检查缓存
        cache_key = self._generate_cache_key(tool_request)
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        # 解析schema
        parsed_schema = parser.parse(tool_request)
        
        # 缓存结果
        self.schema_cache[cache_key] = parsed_schema
        
        return parsed_schema
    
    def _detect_schema_type(self, tool_request):
        # 检测schema类型
        if isinstance(tool_request, dict):
            if "openapi" in tool_request:
                return "openapi"
            elif "type" in tool_request and tool_request["type"] == "object":
                return "json"
            else:
                return "custom"
        elif isinstance(tool_request, str):
            try:
                parsed = json.loads(tool_request)
                return self._detect_schema_type(parsed)
            except:
                return "custom"
        else:
            return "custom"

class JSONSchemaParser:
    def parse(self, schema):
        # 解析JSON Schema
        parsed = {
            "name": schema.get("name", ""),
            "description": schema.get("description", ""),
            "parameters": self._parse_parameters(schema.get("parameters", {})),
            "required": schema.get("required", []),
            "returns": schema.get("returns", {}),
            "examples": schema.get("examples", [])
        }
        
        return parsed
    
    def _parse_parameters(self, parameters):
        # 解析参数定义
        parsed_params = {
            "type": parameters.get("type", "object"),
            "properties": {},
            "required": parameters.get("required", [])
        }
        
        for param_name, param_schema in parameters.get("properties", {}).items():
            parsed_params["properties"][param_name] = {
                "type": param_schema.get("type", "string"),
                "description": param_schema.get("description", ""),
                "default": param_schema.get("default"),
                "enum": param_schema.get("enum"),
                "format": param_schema.get("format"),
                "minimum": param_schema.get("minimum"),
                "maximum": param_schema.get("maximum")
            }
        
        return parsed_params
```

### 1.2 工具选择机制

**面试题：请详细描述你的项目中工具选择的机制，如何根据任务需求选择最合适的工具？**

**答案要点：**

**1. 工具选择策略：**
```python
class ToolSelector:
    def __init__(self):
        self.selection_strategies = {
            "exact_match": self.exact_match_selection,
            "semantic_match": self.semantic_match_selection,
            "capability_match": self.capability_match_selection,
            "hybrid_match": self.hybrid_match_selection
        }
        self.tool_embeddings = {}
        self.selection_cache = {}
    
    def select(self, parsed_request, strategy="hybrid_match"):
        # 选择工具
        selection_strategy = self.selection_strategies[strategy]
        return selection_strategy(parsed_request)
    
    def exact_match_selection(self, parsed_request):
        # 精确匹配选择
        request_name = parsed_request.get("name", "").lower()
        request_description = parsed_request.get("description", "").lower()
        
        candidates = []
        for tool_id, tool_info in self.tool_registry.tools.items():
            tool_name = tool_info["name"].lower()
            tool_description = tool_info["description"].lower()
            
            # 名称匹配
            if request_name in tool_name or tool_name in request_name:
                candidates.append((tool_id, 1.0))
            # 描述匹配
            elif request_description and tool_description:
                similarity = self._calculate_text_similarity(request_description, tool_description)
                if similarity > 0.8:
                    candidates.append((tool_id, similarity))
        
        # 按匹配度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    def semantic_match_selection(self, parsed_request):
        # 语义匹配选择
        request_embedding = self._generate_embedding(parsed_request["description"])
        
        best_tool = None
        best_similarity = 0
        
        for tool_id, tool_info in self.tool_registry.tools.items():
            if tool_id not in self.tool_embeddings:
                tool_embedding = self._generate_embedding(tool_info["description"])
                self.tool_embeddings[tool_id] = tool_embedding
            
            similarity = self._calculate_cosine_similarity(
                request_embedding, 
                self.tool_embeddings[tool_id]
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_tool = tool_id
        
        return best_tool if best_similarity > 0.6 else None
    
    def capability_match_selection(self, parsed_request):
        # 能力匹配选择
        required_capabilities = self._extract_capabilities(parsed_request)
        
        candidates = []
        for tool_id, tool_info in self.tool_registry.tools.items():
            tool_capabilities = tool_info["metadata"].get("capabilities", [])
            
            # 计算能力匹配度
            matched_capabilities = set(required_capabilities) & set(tool_capabilities)
            match_ratio = len(matched_capabilities) / len(required_capabilities) if required_capabilities else 0
            
            if match_ratio > 0.5:
                candidates.append((tool_id, match_ratio))
        
        # 按匹配度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    def hybrid_match_selection(self, parsed_request):
        # 混合匹配选择
        exact_result = self.exact_match_selection(parsed_request)
        if exact_result:
            return exact_result
        
        semantic_result = self.semantic_match_selection(parsed_request)
        if semantic_result:
            return semantic_result
        
        capability_result = self.capability_match_selection(parsed_request)
        return capability_result
```

**2. 工具评分机制：**
```python
class ToolScorer:
    def __init__(self):
        self.scoring_factors = {
            "relevance": 0.4,
            "performance": 0.2,
            "reliability": 0.2,
            "cost": 0.1,
            "availability": 0.1
        }
    
    def score_tool(self, tool_id, parsed_request):
        tool_info = self.tool_registry.get_tool(tool_id)
        
        scores = {
            "relevance": self._calculate_relevance_score(tool_info, parsed_request),
            "performance": self._calculate_performance_score(tool_info),
            "reliability": self._calculate_reliability_score(tool_info),
            "cost": self._calculate_cost_score(tool_info),
            "availability": self._calculate_availability_score(tool_info)
        }
        
        # 计算加权总分
        total_score = sum(
            scores[factor] * self.scoring_factors[factor] 
            for factor in self.scoring_factors
        )
        
        return {
            "tool_id": tool_id,
            "total_score": total_score,
            "factor_scores": scores
        }
    
    def _calculate_relevance_score(self, tool_info, parsed_request):
        # 计算相关性分数
        request_description = parsed_request.get("description", "")
        tool_description = tool_info.get("description", "")
        
        # 文本相似度
        text_similarity = self._calculate_text_similarity(request_description, tool_description)
        
        # 参数匹配度
        param_match_ratio = self._calculate_parameter_match_ratio(
            parsed_request.get("parameters", {}),
            tool_info.get("schema", {}).get("parameters", {})
        )
        
        return (text_similarity + param_match_ratio) / 2
    
    def _calculate_performance_score(self, tool_info):
        # 计算性能分数
        avg_response_time = tool_info["metadata"].get("avg_response_time", 1000)
        max_response_time = tool_info["metadata"].get("max_response_time", 5000)
        
        # 响应时间越短，分数越高
        time_score = max(0, 1 - (avg_response_time / max_response_time))
        
        return time_score
    
    def _calculate_reliability_score(self, tool_info):
        # 计算可靠性分数
        success_rate = tool_info["metadata"].get("success_rate", 0.95)
        error_rate = tool_info["metadata"].get("error_rate", 0.05)
        
        return success_rate * (1 - error_rate)
```

### 1.3 参数对齐机制

**面试题：请详细描述你的项目中参数对齐的机制，如何处理参数类型转换和验证？**

**答案要点：**

**1. 参数对齐策略：**
```python
class ParameterAligner:
    def __init__(self):
        self.type_converters = {
            "string": StringConverter(),
            "integer": IntegerConverter(),
            "number": NumberConverter(),
            "boolean": BooleanConverter(),
            "array": ArrayConverter(),
            "object": ObjectConverter()
        }
        self.validators = {
            "required": RequiredValidator(),
            "format": FormatValidator(),
            "range": RangeValidator(),
            "enum": EnumValidator(),
            "pattern": PatternValidator()
        }
    
    def align(self, selected_tool, parsed_request):
        # 参数对齐
        tool_schema = selected_tool["schema"]
        request_params = parsed_request.get("parameters", {})
        
        # 1. 参数映射
        mapped_params = self._map_parameters(tool_schema, request_params)
        
        # 2. 类型转换
        converted_params = self._convert_types(mapped_params, tool_schema)
        
        # 3. 参数验证
        validated_params = self._validate_parameters(converted_params, tool_schema)
        
        # 4. 默认值填充
        final_params = self._fill_defaults(validated_params, tool_schema)
        
        return final_params
    
    def _map_parameters(self, tool_schema, request_params):
        # 参数映射
        mapped_params = {}
        schema_properties = tool_schema.get("parameters", {}).get("properties", {})
        
        for param_name, param_value in request_params.items():
            # 直接映射
            if param_name in schema_properties:
                mapped_params[param_name] = param_value
            else:
                # 模糊匹配
                matched_name = self._fuzzy_match_parameter(param_name, schema_properties)
                if matched_name:
                    mapped_params[matched_name] = param_value
        
        return mapped_params
    
    def _convert_types(self, mapped_params, tool_schema):
        # 类型转换
        converted_params = {}
        schema_properties = tool_schema.get("parameters", {}).get("properties", {})
        
        for param_name, param_value in mapped_params.items():
            if param_name in schema_properties:
                param_schema = schema_properties[param_name]
                param_type = param_schema.get("type", "string")
                
                converter = self.type_converters.get(param_type)
                if converter:
                    try:
                        converted_value = converter.convert(param_value, param_schema)
                        converted_params[param_name] = converted_value
                    except Exception as e:
                        raise ValueError(f"Type conversion failed for {param_name}: {e}")
                else:
                    converted_params[param_name] = param_value
        
        return converted_params
    
    def _validate_parameters(self, converted_params, tool_schema):
        # 参数验证
        validated_params = {}
        schema_properties = tool_schema.get("parameters", {}).get("properties", {})
        required_params = tool_schema.get("parameters", {}).get("required", [])
        
        # 检查必需参数
        for param_name in required_params:
            if param_name not in converted_params:
                raise ValueError(f"Required parameter missing: {param_name}")
        
        # 验证每个参数
        for param_name, param_value in converted_params.items():
            if param_name in schema_properties:
                param_schema = schema_properties[param_name]
                
                # 应用所有验证器
                for validator_name, validator in self.validators.items():
                    if validator_name in param_schema:
                        validator.validate(param_value, param_schema[validator_name])
                
                validated_params[param_name] = param_value
        
        return validated_params
    
    def _fill_defaults(self, validated_params, tool_schema):
        # 填充默认值
        final_params = validated_params.copy()
        schema_properties = tool_schema.get("parameters", {}).get("properties", {})
        
        for param_name, param_schema in schema_properties.items():
            if param_name not in final_params and "default" in param_schema:
                final_params[param_name] = param_schema["default"]
        
        return final_params
```

**2. 类型转换器实现：**
```python
class StringConverter:
    def convert(self, value, schema):
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float, bool)):
            return str(value)
        else:
            raise ValueError(f"Cannot convert {type(value)} to string")

class IntegerConverter:
    def convert(self, value, schema):
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to integer")
        elif isinstance(value, float):
            if value.is_integer():
                return int(value)
            else:
                raise ValueError(f"Cannot convert {value} to integer")
        else:
            raise ValueError(f"Cannot convert {type(value)} to integer")

class NumberConverter:
    def convert(self, value, schema):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to number")
        else:
            raise ValueError(f"Cannot convert {type(value)} to number")

class BooleanConverter:
    def convert(self, value, schema):
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            if value.lower() in ["true", "1", "yes", "on"]:
                return True
            elif value.lower() in ["false", "0", "no", "off"]:
                return False
            else:
                raise ValueError(f"Cannot convert '{value}' to boolean")
        elif isinstance(value, int):
            return bool(value)
        else:
            raise ValueError(f"Cannot convert {type(value)} to boolean")

class ArrayConverter:
    def convert(self, value, schema):
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # 尝试按分隔符分割
                return value.split(",")
        else:
            raise ValueError(f"Cannot convert {type(value)} to array")

class ObjectConverter:
    def convert(self, value, schema):
        if isinstance(value, dict):
            return value
        elif isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot convert '{value}' to object")
        else:
            raise ValueError(f"Cannot convert {type(value)} to object")
```

### 1.4 调用Fallback机制

**面试题：请详细描述你的项目中工具调用的Fallback机制，如何处理工具调用失败的情况？**

**答案要点：**

**1. Fallback处理策略：**
```python
class FallbackHandler:
    def __init__(self):
        self.fallback_strategies = {
            "retry": self.retry_strategy,
            "alternative_tool": self.alternative_tool_strategy,
            "degraded_service": self.degraded_service_strategy,
            "manual_fallback": self.manual_fallback_strategy
        }
        self.error_classifier = ErrorClassifier()
        self.fallback_config = self._load_fallback_config()
    
    def handle(self, selected_tool, aligned_params, error):
        # 分析错误类型
        error_type = self.error_classifier.classify(error)
        
        # 选择fallback策略
        strategy = self._select_fallback_strategy(error_type, selected_tool)
        
        # 执行fallback
        return self.fallback_strategies[strategy](selected_tool, aligned_params, error)
    
    def retry_strategy(self, selected_tool, aligned_params, error):
        # 重试策略
        max_retries = self.fallback_config.get("max_retries", 3)
        retry_delay = self.fallback_config.get("retry_delay", 1)
        
        for attempt in range(max_retries):
            try:
                # 等待一段时间后重试
                time.sleep(retry_delay * (attempt + 1))
                
                # 重新执行工具调用
                result = self.execution_engine.execute(selected_tool, aligned_params)
                return result
                
            except Exception as retry_error:
                if attempt == max_retries - 1:
                    # 最后一次重试失败，尝试其他策略
                    return self.alternative_tool_strategy(selected_tool, aligned_params, retry_error)
        
        return None
    
    def alternative_tool_strategy(self, selected_tool, aligned_params, error):
        # 替代工具策略
        # 1. 查找功能相似的替代工具
        alternative_tools = self._find_alternative_tools(selected_tool)
        
        # 2. 尝试每个替代工具
        for alternative_tool in alternative_tools:
            try:
                # 参数适配
                adapted_params = self._adapt_parameters(aligned_params, alternative_tool)
                
                # 执行替代工具
                result = self.execution_engine.execute(alternative_tool, adapted_params)
                return result
                
            except Exception as alt_error:
                continue
        
        # 所有替代工具都失败，尝试降级服务
        return self.degraded_service_strategy(selected_tool, aligned_params, error)
    
    def degraded_service_strategy(self, selected_tool, aligned_params, error):
        # 降级服务策略
        # 1. 简化参数
        simplified_params = self._simplify_parameters(aligned_params)
        
        # 2. 使用简化版本的工具
        simplified_tool = self._get_simplified_tool(selected_tool)
        
        try:
            result = self.execution_engine.execute(simplified_tool, simplified_params)
            return result
        except Exception as simplified_error:
            # 降级服务也失败，使用手动fallback
            return self.manual_fallback_strategy(selected_tool, aligned_params, simplified_error)
    
    def manual_fallback_strategy(self, selected_tool, aligned_params, error):
        # 手动fallback策略
        # 1. 记录错误
        self._log_error(selected_tool, aligned_params, error)
        
        # 2. 通知管理员
        self._notify_admin(selected_tool, aligned_params, error)
        
        # 3. 返回错误信息
        return {
            "status": "error",
            "error_type": "manual_fallback_required",
            "message": "Tool execution failed, manual intervention required",
            "tool_id": selected_tool["id"],
            "error_details": str(error),
            "timestamp": time.time()
        }
    
    def _find_alternative_tools(self, selected_tool):
        # 查找替代工具
        alternatives = []
        selected_capabilities = selected_tool["metadata"].get("capabilities", [])
        selected_category = selected_tool["metadata"].get("category", "general")
        
        for tool_id, tool_info in self.tool_registry.tools.items():
            if tool_id == selected_tool["id"]:
                continue
            
            # 检查能力匹配
            tool_capabilities = tool_info["metadata"].get("capabilities", [])
            capability_overlap = len(set(selected_capabilities) & set(tool_capabilities))
            
            # 检查类别匹配
            tool_category = tool_info["metadata"].get("category", "general")
            category_match = (tool_category == selected_category)
            
            # 计算相似度分数
            similarity_score = (capability_overlap / len(selected_capabilities)) * 0.7 + category_match * 0.3
            
            if similarity_score > 0.5:
                alternatives.append((tool_info, similarity_score))
        
        # 按相似度排序
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in alternatives[:3]]
```

**2. 错误分类器：**
```python
class ErrorClassifier:
    def __init__(self):
        self.error_patterns = {
            "network_error": [
                "ConnectionError", "TimeoutError", "NetworkError",
                "connection refused", "timeout", "network unreachable"
            ],
            "authentication_error": [
                "AuthenticationError", "UnauthorizedError", "ForbiddenError",
                "invalid token", "unauthorized", "forbidden"
            ],
            "parameter_error": [
                "ValueError", "TypeError", "ParameterError",
                "invalid parameter", "missing parameter", "type error"
            ],
            "rate_limit_error": [
                "RateLimitError", "TooManyRequestsError",
                "rate limit", "too many requests", "quota exceeded"
            ],
            "service_error": [
                "ServiceError", "InternalServerError", "BadGatewayError",
                "internal error", "service unavailable", "bad gateway"
            ]
        }
    
    def classify(self, error):
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        for category, patterns in self.error_patterns.items():
            # 检查错误类型
            if error_type in patterns:
                return category
            
            # 检查错误消息
            for pattern in patterns:
                if pattern.lower() in error_message:
                    return category
        
        return "unknown_error"
```

### 1.5 统一工具执行接口

**面试题：请详细描述你如何抽象出统一的工具执行接口，支持不同类型的工具调用？**

**答案要点：**

**1. 统一执行接口设计：**
```python
class UnifiedExecutionInterface:
    def __init__(self):
        self.execution_adapters = {
            "http": HTTPExecutionAdapter(),
            "grpc": GRPCExecutionAdapter(),
            "local": LocalExecutionAdapter(),
            "async": AsyncExecutionAdapter(),
            "batch": BatchExecutionAdapter()
        }
        self.execution_monitor = ExecutionMonitor()
        self.result_formatter = ResultFormatter()
    
    def execute(self, tool, parameters, execution_config=None):
        # 统一执行接口
        execution_id = self._generate_execution_id()
        
        try:
            # 1. 准备执行
            prepared_tool = self._prepare_tool(tool, parameters)
            
            # 2. 选择执行适配器
            adapter = self._select_adapter(prepared_tool, execution_config)
            
            # 3. 执行工具
            raw_result = adapter.execute(prepared_tool, parameters)
            
            # 4. 格式化结果
            formatted_result = self.result_formatter.format(raw_result, tool)
            
            # 5. 记录执行信息
            self.execution_monitor.record_execution(
                execution_id, tool, parameters, formatted_result, "success"
            )
            
            return formatted_result
            
        except Exception as error:
            # 记录错误
            self.execution_monitor.record_execution(
                execution_id, tool, parameters, None, "error", error
            )
            raise
    
    def _prepare_tool(self, tool, parameters):
        # 准备工具执行
        prepared_tool = {
            "id": tool["id"],
            "name": tool["name"],
            "type": tool["metadata"].get("type", "http"),
            "endpoint": tool["metadata"].get("endpoint"),
            "timeout": tool["metadata"].get("timeout", 30),
            "retry_config": tool["metadata"].get("retry_config", {}),
            "auth_config": tool["metadata"].get("auth_config", {}),
            "schema": tool["schema"]
        }
        
        return prepared_tool
    
    def _select_adapter(self, prepared_tool, execution_config):
        # 选择执行适配器
        tool_type = prepared_tool["type"]
        
        if execution_config and "adapter" in execution_config:
            adapter_name = execution_config["adapter"]
        else:
            adapter_name = tool_type
        
        adapter = self.execution_adapters.get(adapter_name)
        if not adapter:
            raise ValueError(f"Unsupported execution adapter: {adapter_name}")
        
        return adapter
```

**2. 执行适配器实现：**
```python
class HTTPExecutionAdapter:
    def __init__(self):
        self.session = requests.Session()
        self.auth_handlers = {
            "api_key": APIKeyAuthHandler(),
            "bearer": BearerAuthHandler(),
            "basic": BasicAuthHandler(),
            "oauth": OAuthAuthHandler()
        }
    
    def execute(self, tool, parameters):
        # HTTP工具执行
        url = tool["endpoint"]
        method = tool["metadata"].get("method", "POST")
        timeout = tool["timeout"]
        
        # 处理认证
        auth_handler = self._get_auth_handler(tool["auth_config"])
        headers = auth_handler.get_headers(tool["auth_config"])
        
        # 准备请求数据
        request_data = self._prepare_request_data(parameters, tool["schema"])
        
        # 发送请求
        response = self.session.request(
            method=method,
            url=url,
            json=request_data,
            headers=headers,
            timeout=timeout
        )
        
        # 处理响应
        return self._handle_response(response, tool["schema"])
    
    def _get_auth_handler(self, auth_config):
        auth_type = auth_config.get("type", "api_key")
        return self.auth_handlers.get(auth_type, self.auth_handlers["api_key"])
    
    def _prepare_request_data(self, parameters, schema):
        # 准备请求数据
        if schema.get("parameters", {}).get("type") == "object":
            return parameters
        else:
            return {"parameters": parameters}
    
    def _handle_response(self, response, schema):
        # 处理响应
        if response.status_code == 200:
            return {
                "status": "success",
                "data": response.json(),
                "status_code": response.status_code
            }
        else:
            raise HTTPError(f"HTTP {response.status_code}: {response.text}")

class LocalExecutionAdapter:
    def __init__(self):
        self.function_registry = {}
    
    def execute(self, tool, parameters):
        # 本地函数执行
        function_name = tool["metadata"].get("function_name")
        if not function_name:
            raise ValueError("Local tool must specify function_name")
        
        function = self.function_registry.get(function_name)
        if not function:
            raise ValueError(f"Function not found: {function_name}")
        
        try:
            result = function(**parameters)
            return {
                "status": "success",
                "data": result
            }
        except Exception as e:
            raise ExecutionError(f"Function execution failed: {e}")
    
    def register_function(self, function_name, function):
        self.function_registry[function_name] = function

class AsyncExecutionAdapter:
    def __init__(self):
        self.async_session = aiohttp.ClientSession()
    
    async def execute(self, tool, parameters):
        # 异步工具执行
        url = tool["endpoint"]
        method = tool["metadata"].get("method", "POST")
        timeout = aiohttp.ClientTimeout(total=tool["timeout"])
        
        # 准备请求数据
        request_data = self._prepare_request_data(parameters, tool["schema"])
        
        async with self.async_session.request(
            method=method,
            url=url,
            json=request_data,
            timeout=timeout
        ) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "status": "success",
                    "data": data,
                    "status_code": response.status
                }
            else:
                text = await response.text()
                raise HTTPError(f"HTTP {response.status}: {text}")
```

**3. 结果格式化器：**
```python
class ResultFormatter:
    def __init__(self):
        self.formatters = {
            "json": JSONFormatter(),
            "xml": XMLFormatter(),
            "text": TextFormatter(),
            "binary": BinaryFormatter()
        }
    
    def format(self, raw_result, tool):
        # 格式化结果
        result_format = tool["metadata"].get("result_format", "json")
        formatter = self.formatters.get(result_format, self.formatters["json"])
        
        formatted_result = formatter.format(raw_result)
        
        # 添加元数据
        formatted_result["metadata"] = {
            "tool_id": tool["id"],
            "tool_name": tool["name"],
            "execution_time": time.time(),
            "format": result_format
        }
        
        return formatted_result

class JSONFormatter:
    def format(self, raw_result):
        if isinstance(raw_result, dict):
            return raw_result
        elif isinstance(raw_result, str):
            try:
                return json.loads(raw_result)
            except json.JSONDecodeError:
                return {"raw_data": raw_result}
        else:
            return {"data": raw_result}

class TextFormatter:
    def format(self, raw_result):
        if isinstance(raw_result, str):
            return {"text": raw_result}
        elif isinstance(raw_result, dict):
            return {"text": json.dumps(raw_result, ensure_ascii=False)}
        else:
            return {"text": str(raw_result)}
```

**4. 执行监控：**
```python
class ExecutionMonitor:
    def __init__(self):
        self.execution_log = []
        self.performance_metrics = {}
        self.error_tracker = ErrorTracker()
    
    def record_execution(self, execution_id, tool, parameters, result, status, error=None):
        # 记录执行信息
        execution_record = {
            "execution_id": execution_id,
            "tool_id": tool["id"],
            "tool_name": tool["name"],
            "parameters": parameters,
            "result": result,
            "status": status,
            "error": str(error) if error else None,
            "timestamp": time.time(),
            "duration": self._calculate_duration(execution_id)
        }
        
        self.execution_log.append(execution_record)
        
        # 更新性能指标
        self._update_performance_metrics(tool["id"], execution_record)
        
        # 跟踪错误
        if error:
            self.error_tracker.track_error(tool["id"], error)
    
    def _update_performance_metrics(self, tool_id, execution_record):
        if tool_id not in self.performance_metrics:
            self.performance_metrics[tool_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "avg_duration": 0,
                "total_duration": 0
            }
        
        metrics = self.performance_metrics[tool_id]
        metrics["total_executions"] += 1
        
        if execution_record["status"] == "success":
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
        
        if execution_record["duration"]:
            metrics["total_duration"] += execution_record["duration"]
            metrics["avg_duration"] = metrics["total_duration"] / metrics["total_executions"]
    
    def get_tool_performance(self, tool_id):
        return self.performance_metrics.get(tool_id, {})
    
    def get_execution_history(self, tool_id=None, limit=100):
        if tool_id:
            return [record for record in self.execution_log if record["tool_id"] == tool_id][-limit:]
        else:
            return self.execution_log[-limit:]
```
