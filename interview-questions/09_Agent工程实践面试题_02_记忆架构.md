# Agent工程实践面试题 - 上下文与记忆架构设计

## 1. 上下文与记忆架构设计分析

### 1.1 详细分析项目的上下文与记忆架构设计

**面试题：请详细分析你的项目中上下文与记忆架构的设计，包括memory是临时补全还是长期语义记忆？如何控制写入/清理？是否了解agentic memory / KV memory等新型组织方式？**

**答案要点：**

**1. 记忆架构设计：**
```python
class MemoryArchitecture:
    def __init__(self):
        self.short_term_memory = ShortTermMemory()    # 临时记忆
        self.long_term_memory = LongTermMemory()      # 长期记忆
        self.semantic_memory = SemanticMemory()       # 语义记忆
        self.episodic_memory = EpisodicMemory()       # 情节记忆
        self.working_memory = WorkingMemory()         # 工作记忆
    
    def store_memory(self, content, memory_type="short_term"):
        # 根据内容类型和重要性选择存储位置
        if memory_type == "short_term":
            return self.short_term_memory.store(content)
        elif memory_type == "long_term":
            return self.long_term_memory.store(content)
        elif memory_type == "semantic":
            return self.semantic_memory.store(content)
        elif memory_type == "episodic":
            return self.episodic_memory.store(content)
    
    def retrieve_memory(self, query, memory_type="all"):
        # 从不同类型的记忆中检索信息
        results = {}
        if memory_type in ["all", "short_term"]:
            results["short_term"] = self.short_term_memory.retrieve(query)
        if memory_type in ["all", "long_term"]:
            results["long_term"] = self.long_term_memory.retrieve(query)
        if memory_type in ["all", "semantic"]:
            results["semantic"] = self.semantic_memory.retrieve(query)
        if memory_type in ["all", "episodic"]:
            results["episodic"] = self.episodic_memory.retrieve(query)
        return results
```

**2. 临时记忆 vs 长期记忆：**
```python
class ShortTermMemory:
    def __init__(self, max_size=1000, ttl=3600):
        self.memory_store = {}
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）
    
    def store(self, content):
        # 临时记忆：快速存储，自动过期
        memory_id = self._generate_id()
        self.memory_store[memory_id] = {
            "content": content,
            "timestamp": time.time(),
            "access_count": 0,
            "importance": self._calculate_importance(content)
        }
        
        # 检查容量限制
        self._enforce_capacity_limit()
        return memory_id
    
    def retrieve(self, query):
        # 基于相似度检索
        results = []
        current_time = time.time()
        
        for memory_id, memory in self.memory_store.items():
            # 检查是否过期
            if current_time - memory["timestamp"] > self.ttl:
                del self.memory_store[memory_id]
                continue
            
            # 计算相似度
            similarity = self._calculate_similarity(query, memory["content"])
            if similarity > 0.5:  # 相似度阈值
                memory["access_count"] += 1
                results.append({
                    "id": memory_id,
                    "content": memory["content"],
                    "similarity": similarity,
                    "access_count": memory["access_count"]
                })
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

class LongTermMemory:
    def __init__(self):
        self.memory_store = {}
        self.index = VectorIndex()  # 向量索引
    
    def store(self, content):
        # 长期记忆：持久化存储，需要重要性评估
        importance = self._evaluate_importance(content)
        if importance > 0.7:  # 重要性阈值
            memory_id = self._generate_id()
            self.memory_store[memory_id] = {
                "content": content,
                "timestamp": time.time(),
                "importance": importance,
                "embedding": self._generate_embedding(content)
            }
            # 更新向量索引
            self.index.add(memory_id, self.memory_store[memory_id]["embedding"])
            return memory_id
        return None
    
    def retrieve(self, query):
        # 基于向量相似度检索
        query_embedding = self._generate_embedding(query)
        similar_memories = self.index.search(query_embedding, top_k=10)
        
        results = []
        for memory_id, similarity in similar_memories:
            if memory_id in self.memory_store:
                results.append({
                    "id": memory_id,
                    "content": self.memory_store[memory_id]["content"],
                    "similarity": similarity,
                    "importance": self.memory_store[memory_id]["importance"]
                })
        
        return results
```

**3. 写入/清理控制机制：**
```python
class MemoryController:
    def __init__(self):
        self.write_policies = {
            "immediate": self.immediate_write,
            "batch": self.batch_write,
            "conditional": self.conditional_write
        }
        self.cleanup_policies = {
            "time_based": self.time_based_cleanup,
            "size_based": self.size_based_cleanup,
            "importance_based": self.importance_based_cleanup
        }
    
    def write_memory(self, content, policy="conditional"):
        return self.write_policies[policy](content)
    
    def cleanup_memory(self, memory_type="short_term", policy="time_based"):
        return self.cleanup_policies[policy](memory_type)
    
    def conditional_write(self, content):
        # 条件写入：根据内容重要性决定是否写入长期记忆
        importance = self._evaluate_importance(content)
        if importance > 0.8:
            return self.long_term_memory.store(content)
        else:
            return self.short_term_memory.store(content)
    
    def importance_based_cleanup(self, memory_type):
        # 基于重要性的清理策略
        if memory_type == "short_term":
            memories = self.short_term_memory.get_all()
            for memory_id, memory in memories.items():
                if memory["importance"] < 0.3:
                    self.short_term_memory.delete(memory_id)
        elif memory_type == "long_term":
            memories = self.long_term_memory.get_all()
            for memory_id, memory in memories.items():
                if memory["importance"] < 0.5:
                    self.long_term_memory.delete(memory_id)
```

### 1.2 Agentic Memory架构

**面试题：请详细描述你了解的Agentic Memory架构，以及如何在你的项目中实现？**

**答案要点：**

**1. Agentic Memory核心概念：**
```python
class AgenticMemory:
    def __init__(self):
        self.memory_agents = {
            "encoder": MemoryEncoder(),
            "retriever": MemoryRetriever(),
            "updater": MemoryUpdater(),
            "compressor": MemoryCompressor()
        }
        self.memory_store = HierarchicalMemoryStore()
    
    def process_memory(self, input_data):
        # 1. 编码记忆
        encoded_memory = self.memory_agents["encoder"].encode(input_data)
        
        # 2. 存储记忆
        memory_id = self.memory_store.store(encoded_memory)
        
        # 3. 更新索引
        self.memory_agents["updater"].update_index(memory_id, encoded_memory)
        
        return memory_id
    
    def retrieve_memory(self, query):
        # 1. 查询解析
        parsed_query = self.memory_agents["encoder"].encode(query)
        
        # 2. 检索相关记忆
        relevant_memories = self.memory_agents["retriever"].retrieve(parsed_query)
        
        # 3. 压缩和整合
        compressed_memories = self.memory_agents["compressor"].compress(relevant_memories)
        
        return compressed_memories
```

**2. 记忆Agent实现：**
```python
class MemoryEncoder:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.metadata_extractor = MetadataExtractor()
    
    def encode(self, content):
        # 生成文本嵌入
        embedding = self.embedding_model.encode(content)
        
        # 提取元数据
        metadata = self.metadata_extractor.extract(content)
        
        return {
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "timestamp": time.time()
        }

class MemoryRetriever:
    def __init__(self):
        self.vector_index = FAISSIndex()
        self.semantic_index = SemanticIndex()
    
    def retrieve(self, query_embedding, top_k=10):
        # 向量检索
        vector_results = self.vector_index.search(query_embedding, top_k)
        
        # 语义检索
        semantic_results = self.semantic_index.search(query_embedding, top_k)
        
        # 合并结果
        combined_results = self._merge_results(vector_results, semantic_results)
        
        return combined_results[:top_k]

class MemoryCompressor:
    def __init__(self):
        self.compression_strategies = {
            "summarization": self.summarize_memories,
            "clustering": self.cluster_memories,
            "abstraction": self.abstract_memories
        }
    
    def compress(self, memories, strategy="summarization"):
        return self.compression_strategies[strategy](memories)
    
    def summarize_memories(self, memories):
        # 使用LLM总结记忆
        combined_content = "\n".join([m["content"] for m in memories])
        summary = self.llm.summarize(combined_content)
        return {
            "type": "summary",
            "content": summary,
            "source_memories": [m["id"] for m in memories]
        }
```

### 1.3 KV Memory架构

**面试题：请详细描述KV Memory架构的设计原理和实现方式？**

**答案要点：**

**1. KV Memory核心设计：**
```python
class KVMemory:
    def __init__(self):
        self.key_store = KeyStore()
        self.value_store = ValueStore()
        self.index = KVIndex()
        self.cache = LRUCache(max_size=1000)
    
    def store(self, key, value, metadata=None):
        # 1. 生成键的嵌入
        key_embedding = self._generate_key_embedding(key)
        
        # 2. 存储键值对
        key_id = self.key_store.store(key, key_embedding)
        value_id = self.value_store.store(value)
        
        # 3. 建立索引关系
        self.index.link(key_id, value_id, metadata)
        
        # 4. 更新缓存
        self.cache.put(key, value)
        
        return {"key_id": key_id, "value_id": value_id}
    
    def retrieve(self, query, top_k=10):
        # 1. 生成查询嵌入
        query_embedding = self._generate_query_embedding(query)
        
        # 2. 相似键检索
        similar_keys = self.key_store.search(query_embedding, top_k)
        
        # 3. 获取对应的值
        results = []
        for key_id, similarity in similar_keys:
            value_id = self.index.get_value_id(key_id)
            value = self.value_store.get(value_id)
            key = self.key_store.get_key(key_id)
            
            results.append({
                "key": key,
                "value": value,
                "similarity": similarity
            })
        
        return results
```

**2. 键值存储实现：**
```python
class KeyStore:
    def __init__(self):
        self.keys = {}
        self.embeddings = {}
        self.vector_index = FAISSIndex()
    
    def store(self, key, embedding):
        key_id = self._generate_id()
        self.keys[key_id] = key
        self.embeddings[key_id] = embedding
        self.vector_index.add(key_id, embedding)
        return key_id
    
    def search(self, query_embedding, top_k=10):
        # 向量相似度搜索
        return self.vector_index.search(query_embedding, top_k)

class ValueStore:
    def __init__(self):
        self.values = {}
        self.compression = ValueCompression()
    
    def store(self, value):
        value_id = self._generate_id()
        
        # 压缩存储
        compressed_value = self.compression.compress(value)
        self.values[value_id] = compressed_value
        
        return value_id
    
    def get(self, value_id):
        compressed_value = self.values.get(value_id)
        if compressed_value:
            return self.compression.decompress(compressed_value)
        return None
```

**3. 高级KV Memory特性：**
```python
class AdvancedKVMemory(KVMemory):
    def __init__(self):
        super().__init__()
        self.hierarchical_index = HierarchicalIndex()
        self.temporal_index = TemporalIndex()
        self.semantic_index = SemanticIndex()
    
    def store_with_hierarchy(self, key, value, hierarchy_path):
        # 支持层次化存储
        memory_id = self.store(key, value)
        self.hierarchical_index.add(memory_id, hierarchy_path)
        return memory_id
    
    def store_with_temporal(self, key, value, timestamp):
        # 支持时间索引
        memory_id = self.store(key, value)
        self.temporal_index.add(memory_id, timestamp)
        return memory_id
    
    def retrieve_by_hierarchy(self, hierarchy_path):
        # 基于层次结构检索
        memory_ids = self.hierarchical_index.search(hierarchy_path)
        return [self.get_by_id(mid) for mid in memory_ids]
    
    def retrieve_by_temporal(self, start_time, end_time):
        # 基于时间范围检索
        memory_ids = self.temporal_index.search(start_time, end_time)
        return [self.get_by_id(mid) for mid in memory_ids]
```

### 1.4 记忆组织策略

**面试题：请分析你的项目中记忆的组织策略，包括分层存储、索引优化和检索效率？**

**答案要点：**

**1. 分层存储架构：**
```python
class HierarchicalMemoryStore:
    def __init__(self):
        self.layers = {
            "L0": HotMemoryLayer(),      # 热数据层
            "L1": WarmMemoryLayer(),     # 温数据层
            "L2": ColdMemoryLayer(),     # 冷数据层
            "L3": ArchiveMemoryLayer()   # 归档层
        }
        self.migration_policy = MemoryMigrationPolicy()
    
    def store(self, content, layer="L0"):
        return self.layers[layer].store(content)
    
    def retrieve(self, query):
        # 从所有层检索
        results = {}
        for layer_name, layer in self.layers.items():
            layer_results = layer.retrieve(query)
            if layer_results:
                results[layer_name] = layer_results
        
        # 合并结果
        return self._merge_layer_results(results)
    
    def migrate_memory(self):
        # 定期迁移记忆
        for layer_name, layer in self.layers.items():
            candidates = layer.get_migration_candidates()
            for candidate in candidates:
                target_layer = self.migration_policy.get_target_layer(candidate)
                if target_layer != layer_name:
                    self._migrate_memory(candidate, layer_name, target_layer)

class HotMemoryLayer:
    def __init__(self, max_size=1000):
        self.memory_store = {}
        self.max_size = max_size
        self.access_counter = {}
    
    def store(self, content):
        memory_id = self._generate_id()
        self.memory_store[memory_id] = content
        self.access_counter[memory_id] = 0
        return memory_id
    
    def retrieve(self, query):
        # 快速检索
        results = []
        for memory_id, content in self.memory_store.items():
            similarity = self._calculate_similarity(query, content)
            if similarity > 0.5:
                self.access_counter[memory_id] += 1
                results.append({
                    "id": memory_id,
                    "content": content,
                    "similarity": similarity
                })
        return results
    
    def get_migration_candidates(self):
        # 获取迁移候选
        candidates = []
        for memory_id, access_count in self.access_counter.items():
            if access_count < 5:  # 访问次数少的记忆
                candidates.append(memory_id)
        return candidates
```

**2. 索引优化策略：**
```python
class MemoryIndexOptimizer:
    def __init__(self):
        self.indexes = {
            "vector": VectorIndex(),
            "semantic": SemanticIndex(),
            "temporal": TemporalIndex(),
            "hierarchical": HierarchicalIndex()
        }
        self.optimization_strategies = {
            "reindex": self.reindex_strategy,
            "partition": self.partition_strategy,
            "cache": self.cache_strategy
        }
    
    def optimize_indexes(self, strategy="reindex"):
        return self.optimization_strategies[strategy]()
    
    def reindex_strategy(self):
        # 重新构建索引
        for index_name, index in self.indexes.items():
            index.rebuild()
    
    def partition_strategy(self):
        # 分区索引
        for index_name, index in self.indexes.items():
            if hasattr(index, 'partition'):
                index.partition()
    
    def cache_strategy(self):
        # 缓存优化
        for index_name, index in self.indexes.items():
            if hasattr(index, 'optimize_cache'):
                index.optimize_cache()
```

**3. 检索效率优化：**
```python
class MemoryRetrievalOptimizer:
    def __init__(self):
        self.cache = LRUCache(max_size=10000)
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_strategies = {
            "exact_match": self.exact_match_retrieval,
            "fuzzy_match": self.fuzzy_match_retrieval,
            "semantic_match": self.semantic_match_retrieval,
            "hybrid_match": self.hybrid_match_retrieval
        }
    
    def retrieve(self, query, strategy="hybrid_match"):
        # 检查缓存
        cache_key = self._generate_cache_key(query, strategy)
        if cache_key in self.cache:
            return self.cache.get(cache_key)
        
        # 分析查询
        query_analysis = self.query_analyzer.analyze(query)
        
        # 选择检索策略
        retrieval_strategy = self.retrieval_strategies[strategy]
        results = retrieval_strategy(query, query_analysis)
        
        # 缓存结果
        self.cache.put(cache_key, results)
        
        return results
    
    def hybrid_match_retrieval(self, query, query_analysis):
        # 混合检索策略
        results = []
        
        # 1. 精确匹配
        exact_results = self.exact_match_retrieval(query, query_analysis)
        results.extend(exact_results)
        
        # 2. 语义匹配
        semantic_results = self.semantic_match_retrieval(query, query_analysis)
        results.extend(semantic_results)
        
        # 3. 去重和排序
        unique_results = self._deduplicate_results(results)
        sorted_results = self._sort_results(unique_results, query_analysis)
        
        return sorted_results
```

### 1.5 记忆压缩与优化

**面试题：请详细描述你的项目中记忆压缩和优化的策略，如何平衡存储空间和检索质量？**

**答案要点：**

**1. 记忆压缩策略：**
```python
class MemoryCompressor:
    def __init__(self):
        self.compression_algorithms = {
            "lossless": self.lossless_compression,
            "lossy": self.lossy_compression,
            "semantic": self.semantic_compression,
            "hierarchical": self.hierarchical_compression
        }
        self.compression_ratio = 0.7  # 压缩比
    
    def compress(self, memories, algorithm="semantic"):
        return self.compression_algorithms[algorithm](memories)
    
    def semantic_compression(self, memories):
        # 语义压缩：保留核心语义信息
        compressed_memories = []
        
        for memory in memories:
            # 提取关键信息
            key_info = self._extract_key_information(memory)
            
            # 生成摘要
            summary = self._generate_summary(memory)
            
            # 保留元数据
            metadata = self._extract_metadata(memory)
            
            compressed_memory = {
                "key_info": key_info,
                "summary": summary,
                "metadata": metadata,
                "original_id": memory["id"]
            }
            
            compressed_memories.append(compressed_memory)
        
        return compressed_memories
    
    def hierarchical_compression(self, memories):
        # 层次化压缩：按重要性分层压缩
        important_memories = []
        normal_memories = []
        less_important_memories = []
        
        for memory in memories:
            importance = self._calculate_importance(memory)
            if importance > 0.8:
                important_memories.append(memory)
            elif importance > 0.5:
                normal_memories.append(memory)
            else:
                less_important_memories.append(memory)
        
        # 不同层次使用不同压缩策略
        compressed_important = self._compress_important(important_memories)
        compressed_normal = self._compress_normal(normal_memories)
        compressed_less = self._compress_less_important(less_important_memories)
        
        return {
            "important": compressed_important,
            "normal": compressed_normal,
            "less_important": compressed_less
        }
```

**2. 存储空间优化：**
```python
class StorageOptimizer:
    def __init__(self):
        self.storage_policies = {
            "tiered_storage": self.tiered_storage_policy,
            "compression_storage": self.compression_storage_policy,
            "distributed_storage": self.distributed_storage_policy
        }
    
    def optimize_storage(self, memories, policy="tiered_storage"):
        return self.storage_policies[policy](memories)
    
    def tiered_storage_policy(self, memories):
        # 分层存储策略
        storage_tiers = {
            "hot": [],    # 内存存储
            "warm": [],   # SSD存储
            "cold": []    # 硬盘存储
        }
        
        for memory in memories:
            access_frequency = self._calculate_access_frequency(memory)
            importance = self._calculate_importance(memory)
            
            if access_frequency > 0.8 or importance > 0.9:
                storage_tiers["hot"].append(memory)
            elif access_frequency > 0.5 or importance > 0.7:
                storage_tiers["warm"].append(memory)
            else:
                storage_tiers["cold"].append(memory)
        
        return storage_tiers
    
    def compression_storage_policy(self, memories):
        # 压缩存储策略
        compressed_memories = []
        
        for memory in memories:
            # 根据内容类型选择压缩算法
            content_type = self._detect_content_type(memory)
            
            if content_type == "text":
                compressed = self._compress_text(memory)
            elif content_type == "image":
                compressed = self._compress_image(memory)
            elif content_type == "structured":
                compressed = self._compress_structured(memory)
            else:
                compressed = self._compress_generic(memory)
            
            compressed_memories.append(compressed)
        
        return compressed_memories
```

**3. 检索质量优化：**
```python
class RetrievalQualityOptimizer:
    def __init__(self):
        self.quality_metrics = {
            "relevance": self.calculate_relevance,
            "completeness": self.calculate_completeness,
            "freshness": self.calculate_freshness,
            "diversity": self.calculate_diversity
        }
        self.optimization_strategies = {
            "reranking": self.rerank_results,
            "diversification": self.diversify_results,
            "expansion": self.expand_query,
            "filtering": self.filter_results
        }
    
    def optimize_retrieval_quality(self, results, query, strategy="reranking"):
        return self.optimization_strategies[strategy](results, query)
    
    def rerank_results(self, results, query):
        # 重新排序结果
        scored_results = []
        
        for result in results:
            score = self._calculate_comprehensive_score(result, query)
            scored_results.append((result, score))
        
        # 按分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, score in scored_results]
    
    def diversify_results(self, results, query):
        # 结果多样化
        diversified_results = []
        used_concepts = set()
        
        for result in results:
            concepts = self._extract_concepts(result)
            
            # 检查概念多样性
            new_concepts = concepts - used_concepts
            if new_concepts or len(diversified_results) < 3:
                diversified_results.append(result)
                used_concepts.update(concepts)
        
        return diversified_results
    
    def _calculate_comprehensive_score(self, result, query):
        # 计算综合评分
        relevance_score = self.quality_metrics["relevance"](result, query)
        completeness_score = self.quality_metrics["completeness"](result)
        freshness_score = self.quality_metrics["freshness"](result)
        diversity_score = self.quality_metrics["diversity"](result)
        
        # 加权平均
        weights = {"relevance": 0.4, "completeness": 0.3, "freshness": 0.2, "diversity": 0.1}
        total_score = (
            relevance_score * weights["relevance"] +
            completeness_score * weights["completeness"] +
            freshness_score * weights["freshness"] +
            diversity_score * weights["diversity"]
        )
        
        return total_score
```
