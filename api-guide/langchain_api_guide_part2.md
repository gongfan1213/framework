# LangChain 常用API及使用场景完整指南 - 第二部分

## 链 (Chains)

### 基础链

#### LLMChain
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 创建提示模板
prompt = PromptTemplate(
    input_variables=["product"],
    template="为{product}写一个营销文案，要求简洁有力，突出产品特点。"
)

# 创建链
llm = ChatOpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)

# 执行
result = chain.run("智能手机")
print(result)

# 批量执行
products = ["笔记本电脑", "无线耳机", "智能手表"]
results = chain.batch([{"product": p} for p in products])
for product, result in zip(products, results):
    print(f"{product}: {result['text']}")
```

#### 对话链
```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

# 多轮对话
conversation.predict(input="你好，我叫张三")
conversation.predict(input="今天天气怎么样？")
conversation.predict(input="你还记得我的名字吗？")

# 获取对话历史
print(conversation.memory.buffer)
```

### 高级链

#### 检索问答链
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

# 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",  # 其他选项: "map_reduce", "refine", "map_rerank"
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 提问
query = "什么是机器学习？"
result = qa_chain({"query": query})
print(f"答案: {result['result']}")
print(f"来源文档: {len(result['source_documents'])} 个")

# 使用不同的链类型
# Map-Reduce: 适合处理大量文档
map_reduce_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever
)

# Refine: 逐步改进答案
refine_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever
)
```

#### 摘要链
```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# 摘要提示
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="请总结以下文本的要点，要求简洁明了：\n\n{text}"
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# 处理长文本
long_text = "这是一个很长的文本内容..."
summary = summary_chain.run(long_text)
print(summary)

# 分块摘要
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_text(long_text)

summaries = []
for chunk in chunks:
    summary = summary_chain.run(chunk)
    summaries.append(summary)

# 合并摘要
final_summary = summary_chain.run("\n".join(summaries))
```

#### 翻译链
```python
# 翻译提示
translation_prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template="将以下文本翻译成{target_language}，保持原意：\n{text}"
)

translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

# 翻译
text = "Hello, how are you?"
translated = translation_chain.run(text=text, target_language="中文")
print(translated)
```

### 顺序链
```python
from langchain.chains import SimpleSequentialChain, SequentialChain

# 简单顺序链
chain1 = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["topic"],
    template="为{topic}写一个标题"
))

chain2 = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["title"],
    template="基于标题'{title}'写一篇文章"
))

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
result = overall_chain.run("人工智能")

# 复杂顺序链
title_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["topic"], template="为{topic}写一个标题"),
    output_key="title"
)

content_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["title"], template="基于标题'{title}'写内容"),
    output_key="content"
)

summary_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["content"], template="总结内容：{content}"),
    output_key="summary"
)

full_chain = SequentialChain(
    chains=[title_chain, content_chain, summary_chain],
    input_variables=["topic"],
    output_variables=["title", "content", "summary"],
    verbose=True
)

result = full_chain({"topic": "机器学习"})
```

### 使用场景
- **问答系统**：基于知识库的问答
- **内容生成**：营销文案、文章创作
- **文档处理**：摘要、翻译、分析
- **对话系统**：聊天机器人

---

## 代理 (Agents)

### 基础代理

#### 零样本ReAct代理
```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# 初始化工具
search = DuckDuckGoSearchRun()

# 创建代理
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 执行任务
try:
    result = agent.run("搜索最新的AI技术发展，并总结主要趋势")
    print(result)
except Exception as e:
    print(f"代理执行出错: {e}")
```

#### 结构化聊天代理
```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("分析当前市场趋势")
```

#### 对话代理
```python
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# 添加记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 多轮对话
agent.run("你好")
agent.run("搜索Python编程教程")
agent.run("你能记住我们之前的对话吗？")
```

### 自定义代理
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# 自定义提示
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""你是一个有用的AI助手，可以使用工具来帮助用户。

可用工具:
{tools}

使用以下格式:
问题: 用户的问题
思考: 你应该思考要做什么
行动: 要使用的工具名称
行动输入: 工具的输入
观察: 工具的结果
... (可以重复思考-行动-观察)
思考: 我现在知道最终答案
最终答案: 对用户问题的最终答案

问题: {input}
{agent_scratchpad}"""
)

# 创建代理
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行
result = agent_executor.invoke({"input": "帮我分析这个数据"})
```

### 多工具代理
```python
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import PythonREPLTool

# 创建多个工具
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
python_tool = PythonREPLTool()

# 创建代理
agent = initialize_agent(
    tools=[search, wikipedia, python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 执行复杂任务
result = agent.run("搜索Python的最新版本，然后计算从Python 2.7到最新版本的版本号差异")
```

### 使用场景
- **任务自动化**：复杂任务分解和执行
- **工具集成**：调用外部API和服务
- **问题解决**：多步骤问题处理
- **智能助手**：多功能AI助手

---

## 工具 (Tools)

### 内置工具

#### 搜索工具
```python
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# DuckDuckGo搜索
search = DuckDuckGoSearchRun()
result = search.run("Python编程教程")
print(result)

# Wikipedia搜索
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wikipedia.run("人工智能")
print(result)

# 使用工具链
from langchain.agents import load_tools

# 加载多个搜索工具
tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
```

#### 计算工具
```python
from langchain.tools import PythonREPLTool

# Python代码执行
python_tool = PythonREPLTool()
result = python_tool.run("print('Hello, World!')")
print(result)

# 数学计算
math_result = python_tool.run("import math; print(math.pi)")
print(math_result)

# 数据处理
data_result = python_tool.run("""
import pandas as pd
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
df = pd.DataFrame(data)
print(df.to_string())
""")
```

#### 文件工具
```python
from langchain.tools import FileReadTool, FileWriteTool

# 读取文件
read_tool = FileReadTool()
content = read_tool.run("path/to/file.txt")
print(content)

# 写入文件
write_tool = FileWriteTool()
write_tool.run({"file_path": "output.txt", "text": "Hello World"})

# 检查文件是否存在
import os
if os.path.exists("output.txt"):
    print("文件创建成功")
```

#### 网络工具
```python
from langchain.tools import RequestsGetTool

# HTTP GET请求
get_tool = RequestsGetTool()
response = get_tool.run("https://api.github.com/users/octocat")
print(response)
```

### 自定义工具
```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str

class WeatherTool(BaseTool):
    name = "weather"
    description = "获取指定城市的天气信息"
    args_schema = WeatherInput
    
    def _run(self, city: str) -> str:
        # 这里可以集成真实的天气API
        # 示例实现
        weather_data = {
            "北京": "晴天，温度25°C",
            "上海": "多云，温度28°C",
            "广州": "雨天，温度30°C"
        }
        return weather_data.get(city, f"无法获取{city}的天气信息")
    
    def _arun(self, city: str) -> str:
        # 异步实现
        return self._run(city)

# 使用自定义工具
weather_tool = WeatherTool()
result = weather_tool.run("北京")
print(result)

# 在代理中使用
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("查询北京的天气")
```

### 高级工具

#### 函数调用工具
```python
from langchain.tools import tool

@tool
def calculate_area(length: float, width: float) -> float:
    """计算矩形的面积"""
    return length * width

@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 使用装饰器工具
result1 = calculate_area.run({"length": 10, "width": 5})
result2 = get_current_time.run({})
print(f"面积: {result1}")
print(f"时间: {result2}")
```

#### 工具链
```python
from langchain.tools import BaseTool
from langchain.agents import Tool

# 创建工具链
def process_data(data: str) -> str:
    """处理数据的工具"""
    return f"处理后的数据: {data.upper()}"

def analyze_data(data: str) -> str:
    """分析数据的工具"""
    return f"分析结果: 数据长度为{len(data)}"

# 包装为LangChain工具
process_tool = Tool(
    name="process_data",
    func=process_data,
    description="处理输入数据"
)

analyze_tool = Tool(
    name="analyze_data",
    func=analyze_data,
    description="分析数据"
)

# 在代理中使用
agent = initialize_agent(
    tools=[process_tool, analyze_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("处理并分析文本'hello world'")
```

### 使用场景
- **API集成**：调用外部服务
- **数据处理**：文件读写、数据转换
- **计算任务**：数学计算、代码执行
- **信息检索**：搜索、查询

---

## 记忆 (Memory)

### 记忆类型

#### 对话记忆
```python
from langchain.memory import ConversationBufferMemory

# 基础对话记忆
memory = ConversationBufferMemory()
memory.save_context({"input": "你好"}, {"output": "你好！有什么可以帮助你的吗？"})
memory.save_context({"input": "我叫张三"}, {"output": "很高兴认识你，张三！"})

# 获取记忆
memory_variables = memory.load_memory_variables({})
print(memory_variables)

# 在链中使用
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="你还记得我的名字吗？")
```

#### 对话窗口记忆
```python
from langchain.memory import ConversationBufferWindowMemory

# 限制记忆窗口大小
memory = ConversationBufferWindowMemory(k=3)  # 只记住最近3轮对话

memory.save_context({"input": "第一轮"}, {"output": "回复1"})
memory.save_context({"input": "第二轮"}, {"output": "回复2"})
memory.save_context({"input": "第三轮"}, {"output": "回复3"})
memory.save_context({"input": "第四轮"}, {"output": "回复4"})

# 只保留最近3轮
variables = memory.load_memory_variables({})
print(variables)
```

#### 对话摘要记忆
```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# 摘要记忆
llm = ChatOpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=llm)

# 添加对话
memory.save_context({"input": "你好"}, {"output": "你好！"})
memory.save_context({"input": "今天天气怎么样？"}, {"output": "今天天气很好！"})
memory.save_context({"input": "适合做什么？"}, {"output": "适合户外活动！"})

# 获取摘要
summary = memory.load_memory_variables({})
print(summary)
```

#### 向量存储记忆
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 向量存储记忆
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 保存记忆
memory.save_context({"input": "用户偏好"}, {"output": "喜欢科技产品"})

# 检索相关记忆
memory_variables = memory.load_memory_variables({"input": "用户喜欢什么？"})
print(memory_variables)
```

### 在链中使用记忆
```python
from langchain.chains import ConversationChain

# 创建带记忆的对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 对话
conversation.predict(input="我的名字是张三")
conversation.predict(input="你还记得我的名字吗？")
conversation.predict(input="我喜欢什么颜色？")
conversation.predict(input="你还记得我喜欢什么颜色吗？")

# 查看记忆内容
print(conversation.memory.buffer)
```

### 高级记忆功能

#### 实体记忆
```python
from langchain.memory import ConversationEntityMemory

# 实体记忆
entity_memory = ConversationEntityMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=entity_memory,
    verbose=True
)

# 对话中会记住实体信息
conversation.predict(input="我叫张三，今年25岁")
conversation.predict(input="我住在北京")
conversation.predict(input="你还记得我的信息吗？")
```

#### 知识图谱记忆
```python
from langchain.memory import ConversationKGMemory

# 知识图谱记忆
kg_memory = ConversationKGMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=kg_memory,
    verbose=True
)

# 构建知识图谱
conversation.predict(input="苹果是一家科技公司")
conversation.predict(input="苹果生产iPhone")
conversation.predict(input="苹果和iPhone有什么关系？")
```

### 使用场景
- **对话系统**：保持对话上下文
- **个性化服务**：记住用户偏好
- **长期交互**：跨会话记忆
- **上下文管理**：复杂对话流程

---

## 提示模板 (Prompts)

### 基础提示模板
```python
from langchain_core.prompts import PromptTemplate

# 简单提示
prompt = PromptTemplate(
    input_variables=["name"],
    template="你好，{name}！"
)

# 使用
result = prompt.format(name="张三")
print(result)

# 多个变量
prompt = PromptTemplate(
    input_variables=["name", "age", "city"],
    template="我叫{name}，今年{age}岁，住在{city}。"
)

result = prompt.format(name="李四", age="30", city="上海")
print(result)
```

### 聊天提示模板
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# 聊天提示
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，专门帮助用户解决问题。"),
    ("human", "请帮我{task}")
])

# 使用
messages = chat_prompt.format_messages(task="写一篇文章")
for message in messages:
    print(f"{message.type}: {message.content}")

# 多轮对话模板
conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手。"),
    ("human", "你好"),
    ("ai", "你好！有什么可以帮助你的吗？"),
    ("human", "{user_input}")
])

messages = conversation_prompt.format_messages(user_input="今天天气怎么样？")
```

### 部分提示
```python
from langchain_core.prompts import PromptTemplate

# 部分填充
prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="告诉我一个{adjective}的{content}故事"
)

# 部分填充
prompt_partial = prompt.partial(adjective="有趣的")
result = prompt_partial.format(content="科幻")
print(result)

# 多个部分填充
prompt_partial2 = prompt.partial(adjective="感人的", content="爱情")
result = prompt_partial2.format()
print(result)
```

### 提示组合
```python
from langchain_core.prompts import PromptTemplate

# 组合提示
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="关于{topic}的要点："
)

prompt2 = PromptTemplate(
    input_variables=["points"],
    template="基于这些要点写一篇文章：{points}"
)

# 组合
combined_prompt = prompt1 + prompt2
result = combined_prompt.format(topic="人工智能", points="AI技术发展迅速")
print(result)
```

### 高级提示功能

#### 少样本学习
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 示例
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "fast", "antonym": "slow"}
]

# 示例格式
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="单词: {word}\n反义词: {antonym}"
)

# 少样本提示
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出以下单词的反义词：",
    suffix="单词: {input_word}\n反义词:",
    input_variables=["input_word"]
)

result = few_shot_prompt.format(input_word="big")
print(result)
```

#### 条件提示
```python
from langchain_core.prompts import PromptTemplate

# 条件提示
def create_conditional_prompt(style):
    if style == "formal":
        template = "请以正式的语气回答：{question}"
    elif style == "casual":
        template = "请以轻松的语气回答：{question}"
    else:
        template = "请回答：{question}"
    
    return PromptTemplate(
        input_variables=["question"],
        template=template
    )

# 使用
formal_prompt = create_conditional_prompt("formal")
casual_prompt = create_conditional_prompt("casual")

print(formal_prompt.format(question="什么是机器学习？"))
print(casual_prompt.format(question="什么是机器学习？"))
```

### 使用场景
- **内容生成**：文章、代码、创意内容
- **问答系统**：结构化问答
- **翻译服务**：多语言翻译
- **分析任务**：数据分析、总结
