# LangChain 常用API及使用场景总结

## 概述

LangChain是一个用于构建LLM驱动应用的框架，提供了丰富的组件和工具来简化AI应用开发。

## 核心组件

### 1. 语言模型 (LLMs)
```python
from langchain_openai import OpenAI, ChatOpenAI

# 基础LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
response = llm.invoke("解释什么是机器学习")

# 聊天模型
chat_model = ChatOpenAI(model="gpt-4", temperature=0.1)
messages = [HumanMessage(content="你好")]
response = chat_model.invoke(messages)
```

### 2. 嵌入模型 (Embeddings)
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embedding = embeddings.embed_query("示例文本")
```

### 3. 向量存储 (Vector Stores)
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents, embeddings)
docs = vectorstore.similarity_search("查询内容", k=3)
```

### 4. 链 (Chains)
```python
from langchain.chains import LLMChain, RetrievalQA

# 基础链
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("输入内容")

# 检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

### 5. 代理 (Agents)
```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
result = agent.run("执行任务")
```

### 6. 工具 (Tools)
```python
from langchain.tools import DuckDuckGoSearchRun, PythonREPLTool

search = DuckDuckGoSearchRun()
python_tool = PythonREPLTool()

# 自定义工具
@tool
def custom_function(input: str) -> str:
    return f"处理结果: {input}"
```

### 7. 记忆 (Memory)
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
```

### 8. 提示模板 (Prompts)
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["name"],
    template="你好，{name}！"
)
```

## 实际应用场景

### 1. 智能客服系统
```python
class CustomerServiceBot:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory
        )
    
    def handle_query(self, user_input):
        return self.conversation.predict(input=user_input)
```

### 2. 文档问答系统
```python
class DocumentQASystem:
    def __init__(self, documents):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(documents, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
    
    def ask_question(self, question):
        return self.qa_chain.run(question)
```

### 3. 内容生成系统
```python
class ContentGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.chains = self._create_chains()
    
    def generate_marketing_copy(self, product, audience):
        prompt = PromptTemplate(
            input_variables=["product", "audience"],
            template="为{product}写一个针对{audience}的营销文案"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(product=product, audience=audience)
```

### 4. 数据分析助手
```python
class DataAnalysisAssistant:
    def __init__(self):
        self.tools = [PythonREPLTool()]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=ChatOpenAI(),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
    
    def analyze_data(self, data_description, analysis_request):
        prompt = f"分析数据: {data_description}\n请求: {analysis_request}"
        return self.agent.run(prompt)
```

## 最佳实践

### 1. 错误处理
```python
from langchain.callbacks import BaseCallbackHandler

class ErrorHandler(BaseCallbackHandler):
    def on_llm_error(self, error, **kwargs):
        logging.error(f"LLM错误: {error}")

def safe_llm_call(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)
```

### 2. 性能优化
```python
# 启用缓存
from langchain.cache import InMemoryCache
import langchain
langchain.cache = InMemoryCache()

# 批量处理
def batch_process(texts, batch_size=10):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = llm.batch(batch)
        results.extend(batch_results)
    return results
```

### 3. 监控和日志
```python
class MonitoringCallback(BaseCallbackHandler):
    def __init__(self):
        self.metrics = {'calls': 0, 'errors': 0}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.metrics['calls'] += 1
    
    def on_llm_error(self, error, **kwargs):
        self.metrics['errors'] += 1
```

### 4. 配置管理
```python
@dataclass
class LangChainConfig:
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    
    def create_llm(self):
        return ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.model_name,
            temperature=self.temperature
        )
```

## 常用工具和集成

### 搜索工具
- `DuckDuckGoSearchRun`: 网络搜索
- `WikipediaQueryRun`: 维基百科搜索

### 计算工具
- `PythonREPLTool`: Python代码执行
- `RequestsGetTool`: HTTP请求

### 文件工具
- `FileReadTool`: 文件读取
- `FileWriteTool`: 文件写入

### 文档加载器
- `PyPDFLoader`: PDF文档
- `Docx2txtLoader`: Word文档
- `WebBaseLoader`: 网页内容

### 向量存储
- `Chroma`: 本地向量数据库
- `Pinecone`: 云向量数据库
- `FAISS`: 高性能向量索引

## 学习路径

1. **基础组件**: LLM → Prompt → Chain
2. **高级功能**: Agent → Memory → Tools
3. **实际应用**: 客服系统 → 问答系统 → 内容生成
4. **生产优化**: 错误处理 → 性能优化 → 监控部署

## 总结

LangChain提供了完整的AI应用开发框架，通过模块化设计支持快速构建各种LLM驱动应用。掌握核心组件和最佳实践，可以高效开发生产级的AI应用。
