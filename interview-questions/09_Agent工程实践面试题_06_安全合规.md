# Agent工程实践面试题 - 安全性与合规意识

## 1. 安全性与合规意识分析

### 1.1 详细分析项目的安全性与合规意识

**面试题：请详细分析你的项目中安全性与合规意识，是否是权限边界？是否考虑prompt注入、数据越权、敏感信息保护？Agent日志是否支持审计与合规回溯？**

**答案要点：**

**1. 安全架构设计：**
```python
class SecurityArchitecture:
    def __init__(self):
        self.authentication_manager = AuthenticationManager()
        self.authorization_manager = AuthorizationManager()
        self.encryption_manager = EncryptionManager()
        self.audit_manager = AuditManager()
        self.compliance_manager = ComplianceManager()
        
        # 安全组件
        self.security_components = {
            "input_validation": InputValidationEngine(),
            "prompt_injection_protection": PromptInjectionProtection(),
            "data_leakage_prevention": DataLeakagePrevention(),
            "access_control": AccessControlSystem(),
            "audit_logging": AuditLoggingSystem()
        }
    
    def initialize_security(self, security_config):
        # 初始化安全系统
        for component_name, component in self.security_components.items():
            if component_name in security_config:
                component.initialize(security_config[component_name])
    
    def validate_request(self, request, user_context):
        # 验证请求安全性
        security_checks = [
            self._validate_authentication(request, user_context),
            self._validate_authorization(request, user_context),
            self._validate_input_safety(request),
            self._validate_data_access(request, user_context)
        ]
        
        return all(security_checks)
    
    def audit_operation(self, operation, user_context, result):
        # 审计操作
        audit_record = {
            "operation": operation,
            "user_context": user_context,
            "result": result,
            "timestamp": time.time(),
            "security_level": self._assess_security_level(operation)
        }
        
        self.audit_manager.log_audit_record(audit_record)
```

**2. 权限边界管理：**
```python
class PermissionBoundaryManager:
    def __init__(self):
        self.permission_matrix = {}
        self.role_permissions = {}
        self.resource_permissions = {}
        self.dynamic_permissions = {}
    
    def define_permission_boundary(self, agent_id, permissions):
        # 定义权限边界
        self.permission_matrix[agent_id] = {
            "read_permissions": permissions.get("read", []),
            "write_permissions": permissions.get("write", []),
            "execute_permissions": permissions.get("execute", []),
            "admin_permissions": permissions.get("admin", []),
            "boundary_rules": permissions.get("boundary_rules", [])
        }
    
    def check_permission(self, agent_id, operation, resource):
        # 检查权限
        if agent_id not in self.permission_matrix:
            return False
        
        permissions = self.permission_matrix[agent_id]
        
        # 检查操作权限
        if operation == "read" and resource not in permissions["read_permissions"]:
            return False
        elif operation == "write" and resource not in permissions["write_permissions"]:
            return False
        elif operation == "execute" and resource not in permissions["execute_permissions"]:
            return False
        
        # 检查边界规则
        for rule in permissions["boundary_rules"]:
            if not self._evaluate_boundary_rule(rule, operation, resource):
                return False
        
        return True
    
    def enforce_permission_boundary(self, agent_id, operation, resource, data=None):
        # 强制执行权限边界
        if not self.check_permission(agent_id, operation, resource):
            raise PermissionError(f"Agent {agent_id} not authorized for {operation} on {resource}")
        
        # 应用数据过滤
        if data and operation == "read":
            return self._filter_data_by_permission(agent_id, resource, data)
        
        return True
    
    def _evaluate_boundary_rule(self, rule, operation, resource):
        # 评估边界规则
        rule_type = rule.get("type")
        
        if rule_type == "time_based":
            return self._evaluate_time_rule(rule)
        elif rule_type == "location_based":
            return self._evaluate_location_rule(rule)
        elif rule_type == "context_based":
            return self._evaluate_context_rule(rule)
        elif rule_type == "data_sensitivity":
            return self._evaluate_sensitivity_rule(rule, resource)
        
        return True
    
    def _filter_data_by_permission(self, agent_id, resource, data):
        # 根据权限过滤数据
        permissions = self.permission_matrix.get(agent_id, {})
        read_permissions = permissions.get("read_permissions", [])
        
        if resource not in read_permissions:
            return None
        
        # 应用数据过滤规则
        filtered_data = data.copy()
        
        # 移除敏感字段
        sensitive_fields = self._get_sensitive_fields(resource)
        for field in sensitive_fields:
            if field in filtered_data:
                del filtered_data[field]
        
        return filtered_data
```

**3. Prompt注入防护：**
```python
class PromptInjectionProtection:
    def __init__(self):
        self.injection_patterns = [
            r"system:|user:|assistant:",
            r"ignore previous instructions",
            r"forget everything",
            r"new instructions:",
            r"override:",
            r"bypass:",
            r"ignore:",
            r"disregard:"
        ]
        self.sanitization_rules = {
            "remove_system_commands": True,
            "escape_special_chars": True,
            "validate_prompt_structure": True,
            "limit_prompt_length": 10000
        }
    
    def validate_prompt(self, prompt, context=None):
        # 验证prompt安全性
        validation_result = {
            "is_safe": True,
            "warnings": [],
            "sanitized_prompt": prompt
        }
        
        # 检查注入模式
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                validation_result["is_safe"] = False
                validation_result["warnings"].append(f"Potential injection pattern detected: {pattern}")
        
        # 检查prompt长度
        if len(prompt) > self.sanitization_rules["limit_prompt_length"]:
            validation_result["warnings"].append("Prompt exceeds length limit")
        
        # 检查prompt结构
        if self.sanitization_rules["validate_prompt_structure"]:
            structure_validation = self._validate_prompt_structure(prompt)
            if not structure_validation["valid"]:
                validation_result["warnings"].append(structure_validation["reason"])
        
        # 清理prompt
        if not validation_result["is_safe"]:
            validation_result["sanitized_prompt"] = self._sanitize_prompt(prompt)
        
        return validation_result
    
    def _validate_prompt_structure(self, prompt):
        # 验证prompt结构
        # 检查是否有不合理的角色切换
        role_patterns = ["system:", "user:", "assistant:"]
        role_count = {}
        
        for role in role_patterns:
            role_count[role] = prompt.lower().count(role)
        
        # 检查是否有过多的角色切换
        total_roles = sum(role_count.values())
        if total_roles > 10:  # 假设正常prompt不应该有超过10个角色标记
            return {"valid": False, "reason": "Too many role switches detected"}
        
        return {"valid": True}
    
    def _sanitize_prompt(self, prompt):
        # 清理prompt
        sanitized = prompt
        
        # 移除注入模式
        for pattern in self.injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # 转义特殊字符
        if self.sanitization_rules["escape_special_chars"]:
            sanitized = html.escape(sanitized)
        
        # 限制长度
        if len(sanitized) > self.sanitization_rules["limit_prompt_length"]:
            sanitized = sanitized[:self.sanitization_rules["limit_prompt_length"]]
        
        return sanitized
    
    def detect_injection_attempt(self, prompt, response):
        # 检测注入尝试
        injection_indicators = [
            "I apologize, but I cannot",
            "I am not able to",
            "I cannot provide",
            "I'm sorry, but I can't",
            "I am not authorized to"
        ]
        
        # 检查响应是否包含拒绝模式
        for indicator in injection_indicators:
            if indicator.lower() in response.lower():
                return {
                    "injection_detected": True,
                    "confidence": 0.8,
                    "indicator": indicator
                }
        
        return {"injection_detected": False, "confidence": 0.0}
```

**4. 数据越权防护：**
```python
class DataAccessControl:
    def __init__(self):
        self.access_policies = {}
        self.data_classification = {}
        self.access_logs = []
        self.violation_detector = ViolationDetector()
    
    def define_access_policy(self, resource_id, policy):
        # 定义访问策略
        self.access_policies[resource_id] = {
            "allowed_users": policy.get("allowed_users", []),
            "allowed_roles": policy.get("allowed_roles", []),
            "allowed_operations": policy.get("allowed_operations", []),
            "data_sensitivity": policy.get("data_sensitivity", "public"),
            "access_conditions": policy.get("access_conditions", []),
            "audit_required": policy.get("audit_required", True)
        }
    
    def check_data_access(self, user_id, resource_id, operation, context=None):
        # 检查数据访问权限
        if resource_id not in self.access_policies:
            return {"allowed": False, "reason": "Resource not found"}
        
        policy = self.access_policies[resource_id]
        
        # 检查用户权限
        if user_id not in policy["allowed_users"]:
            return {"allowed": False, "reason": "User not authorized"}
        
        # 检查操作权限
        if operation not in policy["allowed_operations"]:
            return {"allowed": False, "reason": "Operation not allowed"}
        
        # 检查访问条件
        for condition in policy["access_conditions"]:
            if not self._evaluate_access_condition(condition, context):
                return {"allowed": False, "reason": f"Condition not met: {condition}"}
        
        # 记录访问日志
        if policy["audit_required"]:
            self._log_access(user_id, resource_id, operation, context)
        
        return {"allowed": True}
    
    def _evaluate_access_condition(self, condition, context):
        # 评估访问条件
        condition_type = condition.get("type")
        
        if condition_type == "time_based":
            return self._evaluate_time_condition(condition)
        elif condition_type == "location_based":
            return self._evaluate_location_condition(condition, context)
        elif condition_type == "role_based":
            return self._evaluate_role_condition(condition, context)
        elif condition_type == "data_sensitivity":
            return self._evaluate_sensitivity_condition(condition, context)
        
        return True
    
    def _log_access(self, user_id, resource_id, operation, context):
        # 记录访问日志
        access_record = {
            "user_id": user_id,
            "resource_id": resource_id,
            "operation": operation,
            "timestamp": time.time(),
            "context": context,
            "ip_address": context.get("ip_address") if context else None,
            "user_agent": context.get("user_agent") if context else None
        }
        
        self.access_logs.append(access_record)
        
        # 检查是否有异常访问
        self.violation_detector.check_for_violations(access_record)
    
    def get_access_history(self, user_id=None, resource_id=None, time_range=None):
        # 获取访问历史
        filtered_logs = self.access_logs
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log["user_id"] == user_id]
        
        if resource_id:
            filtered_logs = [log for log in filtered_logs if log["resource_id"] == resource_id]
        
        if time_range:
            start_time = time.time() - time_range
            filtered_logs = [log for log in filtered_logs if log["timestamp"] >= start_time]
        
        return filtered_logs

class ViolationDetector:
    def __init__(self):
        self.violation_patterns = {
            "unusual_access_time": self._detect_unusual_access_time,
            "unusual_access_frequency": self._detect_unusual_access_frequency,
            "unusual_access_location": self._detect_unusual_access_location,
            "privilege_escalation": self._detect_privilege_escalation
        }
        self.violation_thresholds = {
            "max_access_per_hour": 100,
            "max_access_per_day": 1000,
            "unusual_hours": [0, 1, 2, 3, 4, 5, 6, 22, 23]  # 非工作时间
        }
    
    def check_for_violations(self, access_record):
        # 检查违规行为
        violations = []
        
        for pattern_name, detector in self.violation_patterns.items():
            if detector(access_record):
                violations.append({
                    "pattern": pattern_name,
                    "access_record": access_record,
                    "timestamp": time.time()
                })
        
        if violations:
            self._handle_violations(violations)
        
        return violations
    
    def _detect_unusual_access_time(self, access_record):
        # 检测异常访问时间
        access_time = datetime.fromtimestamp(access_record["timestamp"])
        hour = access_time.hour
        
        return hour in self.violation_thresholds["unusual_hours"]
    
    def _detect_unusual_access_frequency(self, access_record):
        # 检测异常访问频率
        user_id = access_record["user_id"]
        current_time = access_record["timestamp"]
        
        # 检查最近一小时的访问次数
        one_hour_ago = current_time - 3600
        recent_accesses = [
            log for log in self.access_logs
            if log["user_id"] == user_id and log["timestamp"] >= one_hour_ago
        ]
        
        return len(recent_accesses) > self.violation_thresholds["max_access_per_hour"]
    
    def _detect_unusual_access_location(self, access_record):
        # 检测异常访问位置
        # 这里可以实现基于IP地址的地理位置检测
        return False
    
    def _detect_privilege_escalation(self, access_record):
        # 检测权限提升
        # 这里可以实现权限提升检测逻辑
        return False
    
    def _handle_violations(self, violations):
        # 处理违规行为
        for violation in violations:
            # 记录违规
            self._log_violation(violation)
            
            # 发送告警
            self._send_violation_alert(violation)
            
            # 采取行动（如临时阻止访问）
            self._take_action(violation)
```

**5. 敏感信息保护：**
```python
class SensitiveDataProtection:
    def __init__(self):
        self.data_classifiers = {
            "pii": PIIClassifier(),
            "financial": FinancialDataClassifier(),
            "health": HealthDataClassifier(),
            "confidential": ConfidentialDataClassifier()
        }
        self.encryption_manager = EncryptionManager()
        self.masking_strategies = {
            "redaction": RedactionStrategy(),
            "anonymization": AnonymizationStrategy(),
            "pseudonymization": PseudonymizationStrategy(),
            "tokenization": TokenizationStrategy()
        }
    
    def classify_data(self, data):
        # 分类数据敏感度
        classification_result = {
            "sensitivity_level": "public",
            "data_types": [],
            "protection_required": False
        }
        
        for classifier_name, classifier in self.data_classifiers.items():
            if classifier.is_sensitive(data):
                classification_result["data_types"].append(classifier_name)
                classification_result["protection_required"] = True
        
        # 确定敏感度级别
        if "pii" in classification_result["data_types"]:
            classification_result["sensitivity_level"] = "high"
        elif "financial" in classification_result["data_types"]:
            classification_result["sensitivity_level"] = "high"
        elif "health" in classification_result["data_types"]:
            classification_result["sensitivity_level"] = "critical"
        elif "confidential" in classification_result["data_types"]:
            classification_result["sensitivity_level"] = "medium"
        
        return classification_result
    
    def protect_sensitive_data(self, data, protection_level="standard"):
        # 保护敏感数据
        classification = self.classify_data(data)
        
        if not classification["protection_required"]:
            return data
        
        protected_data = data.copy()
        
        # 根据敏感度级别选择保护策略
        if classification["sensitivity_level"] == "critical":
            protected_data = self._apply_critical_protection(protected_data)
        elif classification["sensitivity_level"] == "high":
            protected_data = self._apply_high_protection(protected_data)
        elif classification["sensitivity_level"] == "medium":
            protected_data = self._apply_medium_protection(protected_data)
        
        return protected_data
    
    def _apply_critical_protection(self, data):
        # 应用关键级别保护
        # 使用最强加密和匿名化
        encrypted_data = self.encryption_manager.encrypt(data)
        anonymized_data = self.masking_strategies["anonymization"].apply(data)
        
        return {
            "encrypted": encrypted_data,
            "anonymized": anonymized_data,
            "protection_level": "critical"
        }
    
    def _apply_high_protection(self, data):
        # 应用高级别保护
        # 使用加密和假名化
        encrypted_data = self.encryption_manager.encrypt(data)
        pseudonymized_data = self.masking_strategies["pseudonymization"].apply(data)
        
        return {
            "encrypted": encrypted_data,
            "pseudonymized": pseudonymized_data,
            "protection_level": "high"
        }
    
    def _apply_medium_protection(self, data):
        # 应用中级别保护
        # 使用脱敏和标记化
        masked_data = self.masking_strategies["redaction"].apply(data)
        tokenized_data = self.masking_strategies["tokenization"].apply(data)
        
        return {
            "masked": masked_data,
            "tokenized": tokenized_data,
            "protection_level": "medium"
        }

class PIIClassifier:
    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "address": r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'
        }
    
    def is_sensitive(self, data):
        # 检查是否包含PII
        data_str = str(data)
        
        for pattern_name, pattern in self.pii_patterns.items():
            if re.search(pattern, data_str):
                return True
        
        return False

class EncryptionManager:
    def __init__(self):
        self.encryption_key = self._generate_encryption_key()
        self.algorithm = "AES-256-GCM"
    
    def encrypt(self, data):
        # 加密数据
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        # 使用Fernet进行加密
        f = Fernet(self.encryption_key)
        encrypted_data = f.encrypt(data_bytes)
        
        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
            "algorithm": self.algorithm,
            "timestamp": time.time()
        }
    
    def decrypt(self, encrypted_data):
        # 解密数据
        f = Fernet(self.encryption_key)
        encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])
        decrypted_bytes = f.decrypt(encrypted_bytes)
        
        return decrypted_bytes.decode('utf-8')
    
    def _generate_encryption_key(self):
        # 生成加密密钥
        return Fernet.generate_key()
```

**6. 审计与合规回溯：**
```python
class AuditComplianceManager:
    def __init__(self):
        self.audit_logs = []
        self.compliance_rules = {}
        self.retention_policies = {}
        self.report_generator = ComplianceReportGenerator()
    
    def log_audit_event(self, event_type, user_id, resource_id, action, details):
        # 记录审计事件
        audit_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "user_id": user_id,
            "resource_id": resource_id,
            "action": action,
            "details": details,
            "timestamp": time.time(),
            "ip_address": details.get("ip_address"),
            "user_agent": details.get("user_agent"),
            "session_id": details.get("session_id")
        }
        
        self.audit_logs.append(audit_event)
        
        # 检查合规性
        self._check_compliance(audit_event)
    
    def search_audit_logs(self, filters=None, time_range=None):
        # 搜索审计日志
        filtered_logs = self.audit_logs
        
        if filters:
            for key, value in filters.items():
                filtered_logs = [log for log in filtered_logs if log.get(key) == value]
        
        if time_range:
            start_time = time.time() - time_range
            filtered_logs = [log for log in filtered_logs if log["timestamp"] >= start_time]
        
        return filtered_logs
    
    def generate_compliance_report(self, report_type, time_range=None):
        # 生成合规报告
        if report_type == "access_report":
            return self.report_generator.generate_access_report(time_range)
        elif report_type == "security_report":
            return self.report_generator.generate_security_report(time_range)
        elif report_type == "data_usage_report":
            return self.report_generator.generate_data_usage_report(time_range)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _check_compliance(self, audit_event):
        # 检查合规性
        for rule_name, rule in self.compliance_rules.items():
            if self._evaluate_compliance_rule(rule, audit_event):
                self._handle_compliance_violation(rule_name, audit_event)
    
    def _evaluate_compliance_rule(self, rule, audit_event):
        # 评估合规规则
        rule_type = rule.get("type")
        
        if rule_type == "access_frequency":
            return self._evaluate_access_frequency_rule(rule, audit_event)
        elif rule_type == "data_access":
            return self._evaluate_data_access_rule(rule, audit_event)
        elif rule_type == "time_based":
            return self._evaluate_time_based_rule(rule, audit_event)
        
        return True
    
    def _handle_compliance_violation(self, rule_name, audit_event):
        # 处理合规违规
        violation = {
            "rule_name": rule_name,
            "audit_event": audit_event,
            "timestamp": time.time(),
            "severity": "medium"
        }
        
        # 记录违规
        self._log_compliance_violation(violation)
        
        # 发送告警
        self._send_compliance_alert(violation)

class ComplianceReportGenerator:
    def __init__(self):
        self.audit_manager = None  # 将在初始化时设置
    
    def generate_access_report(self, time_range=None):
        # 生成访问报告
        access_logs = self.audit_manager.search_audit_logs(
            filters={"event_type": "access"},
            time_range=time_range
        )
        
        report = {
            "report_type": "access_report",
            "generated_at": time.time(),
            "time_range": time_range,
            "summary": self._generate_access_summary(access_logs),
            "details": self._generate_access_details(access_logs)
        }
        
        return report
    
    def generate_security_report(self, time_range=None):
        # 生成安全报告
        security_logs = self.audit_manager.search_audit_logs(
            filters={"event_type": "security"},
            time_range=time_range
        )
        
        report = {
            "report_type": "security_report",
            "generated_at": time.time(),
            "time_range": time_range,
            "summary": self._generate_security_summary(security_logs),
            "details": self._generate_security_details(security_logs)
        }
        
        return report
    
    def _generate_access_summary(self, access_logs):
        # 生成访问摘要
        total_accesses = len(access_logs)
        unique_users = len(set(log["user_id"] for log in access_logs))
        unique_resources = len(set(log["resource_id"] for log in access_logs))
        
        # 按操作类型统计
        action_counts = {}
        for log in access_logs:
            action = log["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_accesses": total_accesses,
            "unique_users": unique_users,
            "unique_resources": unique_resources,
            "action_distribution": action_counts
        }
    
    def _generate_access_details(self, access_logs):
        # 生成访问详情
        details = {
            "user_access_patterns": self._analyze_user_access_patterns(access_logs),
            "resource_access_patterns": self._analyze_resource_access_patterns(access_logs),
            "time_based_analysis": self._analyze_time_based_patterns(access_logs)
        }
        
        return details
    
    def _analyze_user_access_patterns(self, access_logs):
        # 分析用户访问模式
        user_patterns = {}
        
        for log in access_logs:
            user_id = log["user_id"]
            if user_id not in user_patterns:
                user_patterns[user_id] = {
                    "total_accesses": 0,
                    "resources_accessed": set(),
                    "actions_performed": set(),
                    "access_times": []
                }
            
            user_patterns[user_id]["total_accesses"] += 1
            user_patterns[user_id]["resources_accessed"].add(log["resource_id"])
            user_patterns[user_id]["actions_performed"].add(log["action"])
            user_patterns[user_id]["access_times"].append(log["timestamp"])
        
        # 转换为可序列化的格式
        for user_id, pattern in user_patterns.items():
            pattern["resources_accessed"] = list(pattern["resources_accessed"])
            pattern["actions_performed"] = list(pattern["actions_performed"])
        
        return user_patterns
```

**7. 合规性检查与报告：**
```python
class ComplianceChecker:
    def __init__(self):
        self.compliance_frameworks = {
            "gdpr": GDPRCompliance(),
            "ccpa": CCPACompliance(),
            "sox": SOXCompliance(),
            "hipaa": HIPAACompliance()
        }
        self.compliance_status = {}
    
    def check_compliance(self, framework_name, data_context):
        # 检查合规性
        if framework_name not in self.compliance_frameworks:
            raise ValueError(f"Unknown compliance framework: {framework_name}")
        
        framework = self.compliance_frameworks[framework_name]
        compliance_result = framework.check_compliance(data_context)
        
        self.compliance_status[framework_name] = {
            "status": compliance_result["status"],
            "last_check": time.time(),
            "violations": compliance_result.get("violations", []),
            "recommendations": compliance_result.get("recommendations", [])
        }
        
        return self.compliance_status[framework_name]
    
    def generate_compliance_report(self, framework_name=None):
        # 生成合规报告
        if framework_name:
            frameworks = [framework_name]
        else:
            frameworks = list(self.compliance_frameworks.keys())
        
        report = {
            "generated_at": time.time(),
            "overall_status": "compliant",
            "framework_reports": {}
        }
        
        for framework in frameworks:
            if framework in self.compliance_status:
                status = self.compliance_status[framework]
                report["framework_reports"][framework] = status
                
                if status["status"] != "compliant":
                    report["overall_status"] = "non_compliant"
        
        return report

class GDPRCompliance:
    def check_compliance(self, data_context):
        # 检查GDPR合规性
        violations = []
        recommendations = []
        
        # 检查数据最小化原则
        if not self._check_data_minimization(data_context):
            violations.append("Data minimization principle not followed")
            recommendations.append("Review and reduce data collection to minimum necessary")
        
        # 检查用户同意
        if not self._check_user_consent(data_context):
            violations.append("User consent not properly obtained")
            recommendations.append("Implement proper consent management system")
        
        # 检查数据主体权利
        if not self._check_data_subject_rights(data_context):
            violations.append("Data subject rights not properly implemented")
            recommendations.append("Implement data subject rights management")
        
        # 检查数据保护
        if not self._check_data_protection(data_context):
            violations.append("Data protection measures insufficient")
            recommendations.append("Implement appropriate data protection measures")
        
        status = "compliant" if not violations else "non_compliant"
        
        return {
            "status": status,
            "violations": violations,
            "recommendations": recommendations
        }
    
    def _check_data_minimization(self, data_context):
        # 检查数据最小化
        collected_data = data_context.get("collected_data", [])
        purpose = data_context.get("purpose", "")
        
        # 检查收集的数据是否与目的相关
        relevant_data = self._get_relevant_data_for_purpose(purpose)
        
        for data_item in collected_data:
            if data_item not in relevant_data:
                return False
        
        return True
    
    def _check_user_consent(self, data_context):
        # 检查用户同意
        consent_mechanism = data_context.get("consent_mechanism", {})
        
        required_elements = [
            "explicit_consent",
            "purpose_disclosure",
            "withdrawal_mechanism",
            "consent_records"
        ]
        
        for element in required_elements:
            if element not in consent_mechanism:
                return False
        
        return True
    
    def _check_data_subject_rights(self, data_context):
        # 检查数据主体权利
        rights_implementation = data_context.get("data_subject_rights", {})
        
        required_rights = [
            "right_to_access",
            "right_to_rectification",
            "right_to_erasure",
            "right_to_portability",
            "right_to_object"
        ]
        
        for right in required_rights:
            if right not in rights_implementation:
                return False
        
        return True
    
    def _check_data_protection(self, data_context):
        # 检查数据保护
        protection_measures = data_context.get("protection_measures", {})
        
        required_measures = [
            "encryption",
            "access_control",
            "audit_logging",
            "data_backup",
            "incident_response"
        ]
        
        for measure in required_measures:
            if measure not in protection_measures:
                return False
        
        return True
```
