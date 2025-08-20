# LangChain 常用API及使用场景完整指南

## 目录
1. [概述](#概述)
2. [核心组件](#核心组件)
3. [语言模型 (LLMs)](#语言模型-llms)
4. [聊天模型 (Chat Models)](#聊天模型-chat-models)
5. [嵌入模型 (Embeddings)](#嵌入模型-embeddings)
6. [向量存储 (Vector Stores)](#向量存储-vector-stores)
7. [文档加载器 (Document Loaders)](#文档加载器-document-loaders)
8. [文本分割器 (Text Splitters)](#文本分割器-text-splitters)
9. [链 (Chains)](#链-chains)
10. [代理 (Agents)](#代理-agents)
11. [工具 (Tools)](#工具-tools)
12. [记忆 (Memory)](#记忆-memory)
13. [提示模板 (Prompts)](#提示模板-prompts)
14. [输出解析器 (Output Parsers)](#输出解析器-output-parsers)
15. [回调 (Callbacks)](#回调-callbacks)
16. [实际应用场景](#实际应用场景)
17. [最佳实践](#最佳实践)

---

## 概述

LangChain是一个用于构建LLM驱动应用的框架，提供了丰富的组件和工具来简化AI应用开发。本文档将详细介绍LangChain的核心API及其使用场景。

### 主要特性
- **模块化设计**：可组合的组件架构
- **多模型支持**：支持各种LLM提供商
- **丰富的集成**：与外部工具和服务的深度集成
- **生产就绪**：支持部署和监控

### 安装和配置
```bash
# 安装LangChain
pip install langchain

# 安装常用集成
pip install langchain-openai langchain-anthropic langchain-community

# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

---

## 核心组件

### 基础架构
```python
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
```

### 组件关系图
```
用户输入 → 提示模板 → LLM → 输出解析器 → 结果
    ↓
文档加载器 → 文本分割器 → 嵌入模型 → 向量存储
    ↓
记忆系统 ← 代理 ← 工具链
```

---

## 语言模型 (LLMs)

### 支持的提供商

#### OpenAI
```python
from langchain_openai import OpenAI

# 基础配置
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key"
)

# 使用
response = llm.invoke("解释什么是机器学习")
print(response)

# 流式输出
for chunk in llm.stream("写一个Python函数"):
    print(chunk, end="", flush=True)
```

#### Anthropic
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.1,
    max_tokens=1000
)

response = llm.invoke("分析这段代码的性能问题")
```

#### 本地模型
```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama2",
    temperature=0.7
)

response = llm.invoke("解释什么是深度学习")
```

### 使用场景
- **文本生成**：内容创作、代码生成
- **问答系统**：知识问答、客服机器人
- **翻译服务**：多语言翻译
- **摘要生成**：文档摘要、会议记录

---

## 聊天模型 (Chat Models)

### 基础使用
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 初始化聊天模型
chat_model = ChatOpenAI(
    model="gpt-4",
    temperature=0.1
)

# 发送消息
messages = [
    SystemMessage(content="你是一个有用的AI助手"),
    HumanMessage(content="你好，请介绍一下自己")
]

response = chat_model.invoke(messages)
print(response.content)

# 添加AI回复到对话
messages.append(AIMessage(content=response.content))
messages.append(HumanMessage(content="你能做什么？"))
response = chat_model.invoke(messages)
```

### 流式响应
```python
# 流式处理
for chunk in chat_model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 批量处理
```python
# 批量处理多个消息
batch_messages = [
    [HumanMessage(content="问题1")],
    [HumanMessage(content="问题2")],
    [HumanMessage(content="问题3")]
]

responses = chat_model.batch(batch_messages)
for response in responses:
    print(response.content)
```

### 使用场景
- **对话系统**：聊天机器人、虚拟助手
- **多轮对话**：复杂问题解决
- **实时交互**：流式响应应用

---

## 嵌入模型 (Embeddings)

### 支持的提供商

#### OpenAI Embeddings
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# 生成嵌入
text = "这是一个示例文本"
embedding = embeddings.embed_query(text)
print(f"嵌入维度: {len(embedding)}")

# 批量嵌入
texts = ["文本1", "文本2", "文本3"]
embeddings_list = embeddings.embed_documents(texts)
print(f"批量嵌入数量: {len(embeddings_list)}")
```

#### 本地嵌入模型
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 生成嵌入
embedding = embeddings.embed_query("示例文本")
```

#### 缓存嵌入
```python
from langchain.cache import InMemoryCache
import langchain

# 启用缓存
langchain.cache = InMemoryCache()

# 重复查询会使用缓存
embedding1 = embeddings.embed_query("相同文本")
embedding2 = embeddings.embed_query("相同文本")  # 使用缓存
```

### 使用场景
- **语义搜索**：基于相似度的文档检索
- **推荐系统**：内容推荐、用户画像
- **聚类分析**：文档分类、主题发现
- **相似度计算**：文本相似度匹配

---

## 向量存储 (Vector Stores)

### 支持的向量数据库

#### Chroma
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 准备文档
documents = [
    Document(page_content="机器学习是人工智能的一个分支"),
    Document(page_content="深度学习使用神经网络进行学习"),
    Document(page_content="自然语言处理处理人类语言")
]

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 相似性搜索
query = "什么是机器学习？"
docs = vectorstore.similarity_search(query, k=3)
for doc in docs:
    print(doc.page_content)

# 带分数的搜索
docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in docs_and_scores:
    print(f"内容: {doc.page_content}, 相似度: {score}")
```

#### Pinecone
```python
from langchain_community.vectorstores import Pinecone
import pinecone

# 初始化Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")

vectorstore = Pinecone.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name="your-index-name"
)

# 搜索
results = vectorstore.similarity_search("查询内容")
```

#### FAISS
```python
from langchain_community.vectorstores import FAISS

# 创建FAISS索引
vectorstore = FAISS.from_documents(documents, embeddings)

# 保存和加载
vectorstore.save_local("faiss_index")
loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)

# 搜索
results = vectorstore.similarity_search("查询内容")
```

### 高级功能

#### 混合搜索
```python
# 结合关键词和语义搜索
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 创建不同的检索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(documents)

# 组合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# 使用组合检索器
results = ensemble_retriever.get_relevant_documents("查询内容")
```

### 使用场景
- **RAG系统**：检索增强生成
- **知识库**：企业知识管理
- **搜索引擎**：语义搜索
- **推荐系统**：内容推荐

---

## 文档加载器 (Document Loaders)

### 支持的文档类型

#### PDF文档
```python
from langchain_community.document_loaders import PyPDFLoader

# 加载PDF
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()

# 分页加载
pages = loader.load_and_split()
print(f"总页数: {len(pages)}")

# 获取元数据
for doc in documents:
    print(f"页码: {doc.metadata.get('page')}")
    print(f"内容长度: {len(doc.page_content)}")
```

#### Word文档
```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("path/to/document.docx")
documents = loader.load()

# 处理文档内容
for doc in documents:
    print(doc.page_content[:200])  # 显示前200个字符
```

#### 网页内容
```python
from langchain_community.document_loaders import WebBaseLoader

# 加载单个网页
loader = WebBaseLoader("https://example.com")
documents = loader.load()

# 加载多个网页
urls = ["https://example1.com", "https://example2.com"]
loader = WebBaseLoader(urls)
documents = loader.load()

# 获取网页元数据
for doc in documents:
    print(f"URL: {doc.metadata.get('source')}")
    print(f"标题: {doc.metadata.get('title')}")
```

#### 数据库
```python
from langchain_community.document_loaders import SQLDatabaseLoader
from sqlalchemy import create_engine

# 创建数据库连接
engine = create_engine("sqlite:///database.db")

# 从数据库加载
loader = SQLDatabaseLoader(
    db=engine,
    query="SELECT title, content FROM articles"
)
documents = loader.load()
```

#### 社交媒体
```python
from langchain_community.document_loaders import TwitterTweetLoader

# 加载Twitter推文
loader = TwitterTweetLoader.from_bearer_token(
    oauth2_bearer_token="your-token",
    tweet_ids=["123456789"]
)
documents = loader.load()
```

### 使用场景
- **文档处理**：企业文档管理
- **数据提取**：从各种来源提取信息
- **内容聚合**：多源内容整合
- **知识库构建**：构建企业知识库

---

## 文本分割器 (Text Splitters)

### 分割策略

#### 递归字符分割
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# 分割文档
texts = text_splitter.split_documents(documents)
print(f"分割后文档数量: {len(texts)}")

# 分割单个文本
text = "这是一个很长的文本..."
chunks = text_splitter.split_text(text)
```

#### 标记分割
```python
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)
```

#### 语义分割
```python
from langchain.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text_splitter = SemanticChunker(embeddings)

texts = text_splitter.split_documents(documents)
```

#### 代码分割
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 针对代码的分割器
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
    keep_separator=True
)

# 分割代码文件
code_chunks = code_splitter.split_documents(code_documents)
```

### 使用场景
- **长文档处理**：大文档的预处理
- **上下文管理**：控制输入长度
- **批处理优化**：提高处理效率
- **代码分析**：代码文件的分割和处理
