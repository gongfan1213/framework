# LangChain 常用API及使用场景完整指南 - 第三部分

## 输出解析器 (Output Parsers)

### 基础解析器

#### Pydantic解析器
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Recipe(BaseModel):
    title: str = Field(description="菜谱标题")
    ingredients: List[str] = Field(description="食材列表")
    steps: List[str] = Field(description="制作步骤")
    cooking_time: int = Field(description="烹饪时间（分钟）")
    difficulty: str = Field(description="难度等级：简单/中等/困难")

# 创建解析器
parser = PydanticOutputParser(pydantic_object=Recipe)

# 在提示中使用
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="回答用户问题。\n{format_instructions}\n问题: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 解析输出
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
_input = prompt.format_prompt(question="如何制作番茄炒蛋？")
output = llm.invoke(_input)

try:
    recipe = parser.parse(output.content)
    print(f"菜谱标题: {recipe.title}")
    print(f"食材: {recipe.ingredients}")
    print(f"步骤: {recipe.steps}")
    print(f"烹饪时间: {recipe.cooking_time}分钟")
    print(f"难度: {recipe.difficulty}")
except Exception as e:
    print(f"解析失败: {e}")
```

#### 列表解析器
```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
prompt = PromptTemplate(
    template="列出{subject}的5个例子。\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

_input = prompt.format_prompt(subject="编程语言")
output = llm.invoke(_input)

try:
    languages = parser.parse(output.content)
    print("编程语言列表:")
    for i, lang in enumerate(languages, 1):
        print(f"{i}. {lang}")
except Exception as e:
    print(f"解析失败: {e}")
```

#### 结构化输出解析器
```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# 定义响应模式
response_schemas = [
    ResponseSchema(name="answer", description="问题的答案"),
    ResponseSchema(name="confidence", description="答案的置信度，0-1之间"),
    ResponseSchema(name="sources", description="信息来源列表"),
    ResponseSchema(name="explanation", description="答案的解释")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 使用解析器
prompt = PromptTemplate(
    template="回答用户问题。\n{format_instructions}\n问题: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

_input = prompt.format_prompt(question="什么是人工智能？")
output = llm.invoke(_input)

try:
    result = parser.parse(output.content)
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']}")
    print(f"来源: {result['sources']}")
    print(f"解释: {result['explanation']}")
except Exception as e:
    print(f"解析失败: {e}")
```

### 高级解析器

#### 重试解析器
```python
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryOutputParser

# 基础解析器
base_parser = PydanticOutputParser(pydantic_object=Recipe)

# 重试解析器
retry_parser = RetryOutputParser.from_llm(
    parser=base_parser,
    llm=llm
)

# 使用重试解析器
try:
    recipe = retry_parser.parse_with_prompt(output.content, prompt)
    print("解析成功:", recipe)
except Exception as e:
    print(f"重试解析失败: {e}")
```

#### 修复解析器
```python
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser

# 修复解析器
fix_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm
)

# 使用修复解析器
try:
    recipe = fix_parser.parse(output.content)
    print("修复解析成功:", recipe)
except Exception as e:
    print(f"修复解析失败: {e}")
```

### 使用场景
- **结构化输出**：JSON、XML格式输出
- **数据提取**：从文本中提取结构化信息
- **API集成**：格式化输出用于API调用
- **数据分析**：解析分析结果

---

## 回调 (Callbacks)

### 回调类型

#### 日志回调
```python
from langchain.callbacks import FileCallbackHandler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 文件日志
handler = FileCallbackHandler("langchain.log")
llm = ChatOpenAI(callbacks=[handler])

# 执行操作
response = llm.invoke("你好")
```

#### 流式回调
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 流式输出
llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 流式响应
response = llm.invoke("写一个Python函数")
```

#### 自定义回调
```python
from langchain.callbacks.base import BaseCallbackHandler
import time

class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        print(f"LLM开始处理，输入: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"LLM处理完成，耗时: {duration:.2f}秒")
        print(f"输出: {response}")
    
    def on_llm_error(self, error, **kwargs):
        print(f"LLM处理出错: {error}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"链开始执行，输入: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        print(f"链执行完成，输出: {outputs}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"工具开始执行: {serialized['name']}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"工具执行完成，输出: {output}")

# 使用自定义回调
custom_handler = CustomCallbackHandler()
llm = ChatOpenAI(callbacks=[custom_handler])

# 执行操作
response = llm.invoke("解释什么是机器学习")
```

### 高级回调功能

#### 性能监控回调
```python
class PerformanceCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.metrics = {
            'llm_calls': 0,
            'total_tokens': 0,
            'total_time': 0,
            'errors': 0
        }
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.metrics['llm_calls'] += 1
        self.start_time = time.time()
    
    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        self.metrics['total_time'] += duration
        
        # 估算token数量
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('usage', {})
            self.metrics['total_tokens'] += usage.get('total_tokens', 0)
    
    def on_llm_error(self, error, **kwargs):
        self.metrics['errors'] += 1
    
    def get_metrics(self):
        return self.metrics

# 使用性能监控
perf_handler = PerformanceCallbackHandler()
llm = ChatOpenAI(callbacks=[perf_handler])

# 执行多个操作
for i in range(5):
    llm.invoke(f"这是第{i+1}个问题")

# 获取性能指标
metrics = perf_handler.get_metrics()
print(f"LLM调用次数: {metrics['llm_calls']}")
print(f"总token数: {metrics['total_tokens']}")
print(f"总耗时: {metrics['total_time']:.2f}秒")
print(f"错误次数: {metrics['errors']}")
```

#### 链式回调
```python
from langchain.callbacks import CallbackManager

# 创建回调管理器
callback_manager = CallbackManager([
    StreamingStdOutCallbackHandler(),
    CustomCallbackHandler(),
    PerformanceCallbackHandler()
])

# 使用回调管理器
llm = ChatOpenAI(callbacks=callback_manager)
```

### 使用场景
- **调试**：跟踪执行过程
- **监控**：性能监控和日志记录
- **流式处理**：实时输出
- **错误处理**：异常捕获和处理

---

## 实际应用场景

### 1. 智能客服系统
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class CustomerServiceBot:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        
        # 自定义提示
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""你是一个专业的客服代表，请根据以下对话历史和用户问题提供帮助。

对话历史:
{history}

用户问题: {input}

请以友好、专业的态度回答用户问题。如果遇到无法解决的问题，请建议用户联系人工客服。"""
        )
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )
    
    def handle_query(self, user_input):
        """处理用户查询"""
        try:
            response = self.conversation.predict(input=user_input)
            return response
        except Exception as e:
            return f"抱歉，处理您的问题时出现了错误。请稍后重试或联系人工客服。错误信息: {str(e)}"
    
    def get_conversation_history(self):
        """获取对话历史"""
        return self.memory.buffer

# 使用示例
bot = CustomerServiceBot()

# 处理用户查询
queries = [
    "你好，我想了解你们的产品",
    "如何申请退款？",
    "你们的营业时间是什么时候？",
    "谢谢你的帮助"
]

for query in queries:
    print(f"用户: {query}")
    response = bot.handle_query(query)
    print(f"客服: {response}")
    print("-" * 50)

# 查看对话历史
print("对话历史:")
print(bot.get_conversation_history())
```

### 2. 文档问答系统
```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class DocumentQASystem:
    def __init__(self, documents_path):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.1)
        self.vectorstore = None
        self.qa_chain = None
        self.load_documents(documents_path)
    
    def load_documents(self, documents_path):
        """加载和处理文档"""
        # 加载文档
        loader = PyPDFLoader(documents_path)
        documents = loader.load()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 创建向量存储
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # 创建问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    
    def ask_question(self, question):
        """提问"""
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
            }
        except Exception as e:
            return {
                "answer": f"抱歉，处理您的问题时出现了错误: {str(e)}",
                "sources": []
            }
    
    def search_similar(self, query, k=5):
        """相似性搜索"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# 使用示例
qa_system = DocumentQASystem("path/to/documents.pdf")

# 提问
questions = [
    "文档的主要内容是什么？",
    "有哪些重要的概念？",
    "如何应用这些知识？"
]

for question in questions:
    print(f"问题: {question}")
    result = qa_system.ask_question(question)
    print(f"答案: {result['answer']}")
    print(f"来源: {len(result['sources'])} 个相关片段")
    print("-" * 50)
```

### 3. 内容生成系统
```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class ContentGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.chains = self._create_chains()
    
    def _create_chains(self):
        """创建各种内容生成链"""
        chains = {}
        
        # 营销文案链
        marketing_prompt = PromptTemplate(
            input_variables=["product", "target_audience", "tone"],
            template="""为{product}写一个针对{target_audience}的营销文案，语气要{tone}。

要求：
1. 突出产品特点
2. 吸引目标受众
3. 包含行动号召
4. 长度控制在200字以内"""
        )
        chains['marketing'] = LLMChain(llm=self.llm, prompt=marketing_prompt)
        
        # 文章生成链
        article_prompt = PromptTemplate(
            input_variables=["topic", "style", "length"],
            template="""写一篇关于{topic}的{style}风格文章，长度约{length}字。

要求：
1. 结构清晰
2. 内容充实
3. 语言流畅
4. 符合{style}风格"""
        )
        chains['article'] = LLMChain(llm=self.llm, prompt=article_prompt)
        
        # 社交媒体内容链
        social_prompt = PromptTemplate(
            input_variables=["platform", "topic", "hashtags"],
            template="""为{platform}平台写一条关于{topic}的帖子。

要求：
1. 符合平台特点
2. 内容有趣
3. 包含相关标签
4. 鼓励互动

标签: {hashtags}"""
        )
        chains['social'] = LLMChain(llm=self.llm, prompt=social_prompt)
        
        return chains
    
    def generate_marketing_copy(self, product, target_audience, tone="专业"):
        """生成营销文案"""
        return self.chains['marketing'].run({
            "product": product,
            "target_audience": target_audience,
            "tone": tone
        })
    
    def generate_article(self, topic, style="科普", length="1000"):
        """生成文章"""
        return self.chains['article'].run({
            "topic": topic,
            "style": style,
            "length": length
        })
    
    def generate_social_post(self, platform, topic, hashtags=""):
        """生成社交媒体内容"""
        return self.chains['social'].run({
            "platform": platform,
            "topic": topic,
            "hashtags": hashtags
        })

# 使用示例
generator = ContentGenerator()

# 生成营销文案
marketing_copy = generator.generate_marketing_copy(
    product="智能手表",
    target_audience="年轻白领",
    tone="时尚"
)
print("营销文案:")
print(marketing_copy)
print("-" * 50)

# 生成文章
article = generator.generate_article(
    topic="人工智能的发展趋势",
    style="技术分析",
    length="800"
)
print("文章:")
print(article)
print("-" * 50)

# 生成社交媒体内容
social_post = generator.generate_social_post(
    platform="微博",
    topic="Python编程技巧",
    hashtags="#Python #编程 #技术"
)
print("社交媒体内容:")
print(social_post)
```

### 4. 数据分析助手
```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

class DataAnalysisAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.tools = [PythonREPLTool()]
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """创建数据分析代理"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def analyze_data(self, data_description, analysis_request):
        """分析数据"""
        prompt = f"""
        数据描述: {data_description}
        
        分析请求: {analysis_request}
        
        请使用Python进行数据分析，包括：
        1. 数据加载和预处理
        2. 基本统计分析
        3. 可视化分析
        4. 结果解释
        
        请提供完整的Python代码和结果解释。
        """
        
        try:
            result = self.agent.run(prompt)
            return result
        except Exception as e:
            return f"分析过程中出现错误: {str(e)}"
    
    def create_visualization(self, data_description, chart_type):
        """创建可视化"""
        prompt = f"""
        数据描述: {data_description}
        
        请创建一个{chart_type}图表来可视化这些数据。
        使用matplotlib或seaborn库，并提供代码和图表说明。
        """
        
        try:
            result = self.agent.run(prompt)
            return result
        except Exception as e:
            return f"创建可视化时出现错误: {str(e)}"

# 使用示例
assistant = DataAnalysisAssistant()

# 数据分析
data_desc = "一个包含销售数据的CSV文件，包含日期、产品、销售额、地区等字段"
analysis_req = "分析销售趋势，找出最畅销的产品和地区"

result = assistant.analyze_data(data_desc, analysis_req)
print("数据分析结果:")
print(result)
print("-" * 50)

# 创建可视化
viz_result = assistant.create_visualization(data_desc, "折线图")
print("可视化结果:")
print(viz_result)
```

### 5. 多语言翻译服务
```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class TranslationService:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1)
        self.translation_chain = self._create_translation_chain()
        self.supported_languages = {
            "中文": "Chinese",
            "英文": "English",
            "日文": "Japanese",
            "韩文": "Korean",
            "法文": "French",
            "德文": "German",
            "西班牙文": "Spanish",
            "俄文": "Russian"
        }
    
    def _create_translation_chain(self):
        """创建翻译链"""
        prompt = PromptTemplate(
            input_variables=["text", "target_language", "style"],
            template="""将以下文本翻译成{target_language}，保持原意和风格。

原文: {text}
目标语言: {target_language}
翻译风格: {style}

要求：
1. 准确传达原意
2. 保持原文风格
3. 符合目标语言习惯
4. 专业术语准确翻译"""
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def translate(self, text, target_language, style="标准"):
        """翻译文本"""
        if target_language not in self.supported_languages:
            return f"不支持的目标语言: {target_language}"
        
        try:
            result = self.translation_chain.run({
                "text": text,
                "target_language": self.supported_languages[target_language],
                "style": style
            })
            return result
        except Exception as e:
            return f"翻译过程中出现错误: {str(e)}"
    
    def batch_translate(self, texts, target_language, style="标准"):
        """批量翻译"""
        results = []
        for text in texts:
            result = self.translate(text, target_language, style)
            results.append(result)
        return results
    
    def get_supported_languages(self):
        """获取支持的语言列表"""
        return list(self.supported_languages.keys())

# 使用示例
translator = TranslationService()

# 单文本翻译
text = "Hello, how are you? I hope you're doing well."
translated = translator.translate(text, "中文", "友好")
print(f"原文: {text}")
print(f"译文: {translated}")
print("-" * 50)

# 批量翻译
texts = [
    "Welcome to our company!",
    "We provide excellent service.",
    "Thank you for your business."
]

batch_results = translator.batch_translate(texts, "中文", "正式")
for i, (original, translated) in enumerate(zip(texts, batch_results)):
    print(f"文本{i+1}:")
    print(f"原文: {original}")
    print(f"译文: {translated}")
    print()

# 查看支持的语言
print("支持的语言:")
for lang in translator.get_supported_languages():
    print(f"- {lang}")
```

---

## 最佳实践

### 1. 错误处理
```python
from langchain.callbacks import BaseCallbackHandler
import logging
from typing import Optional

class ErrorHandler(BaseCallbackHandler):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
    
    def on_llm_error(self, error, **kwargs):
        self.error_count += 1
        self.logger.error(f"LLM错误 #{self.error_count}: {error}")
        
        # 记录错误详情
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "context": kwargs
        }
        self.logger.error(f"错误详情: {error_details}")
    
    def on_chain_error(self, error, **kwargs):
        self.logger.error(f"链执行错误: {error}")
    
    def on_tool_error(self, error, **kwargs):
        self.logger.error(f"工具执行错误: {error}")

# 使用错误处理
error_handler = ErrorHandler()
llm = ChatOpenAI(callbacks=[error_handler])

# 包装函数以处理异常
def safe_llm_call(prompt, max_retries=3):
    """安全的LLM调用，包含重试机制"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # 指数退避
```

### 2. 性能优化
```python
from langchain.cache import InMemoryCache
import langchain
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 启用缓存
langchain.cache = InMemoryCache()

# 批量处理
def batch_process(texts, batch_size=10):
    """批量处理文本"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = llm.batch(batch)
        results.extend(batch_results)
    return results

# 异步处理
async def async_process(texts):
    """异步处理文本"""
    tasks = [llm.ainvoke(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results

# 并行处理
def parallel_process(texts, max_workers=4):
    """并行处理文本"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(llm.invoke, text) for text in texts]
        results = [future.result() for future in futures]
    return results

# 性能监控
import time
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result
    return wrapper

@performance_monitor
def optimized_llm_call(prompt):
    """优化的LLM调用"""
    return llm.invoke(prompt)
```

### 3. 安全考虑
```python
from pydantic import BaseModel, validator
import re

class SafeInput(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > 1000:
            raise ValueError("文本长度不能超过1000字符")
        
        # 检查恶意内容
        malicious_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:'
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("检测到潜在的安全风险")
        
        return v

# 输出过滤
def sanitize_output(output):
    """过滤输出内容"""
    if not output:
        return output
    
    # 移除HTML标签
    output = re.sub(r'<[^>]+>', '', output)
    
    # 移除脚本内容
    output = re.sub(r'<script.*?>.*?</script>', '', output, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除危险URL
    output = re.sub(r'javascript:', '', output, flags=re.IGNORECASE)
    output = re.sub(r'data:text/html', '', output, flags=re.IGNORECASE)
    
    return output

# 安全的LLM调用
def safe_llm_invoke(prompt):
    """安全的LLM调用"""
    try:
        # 验证输入
        safe_input = SafeInput(text=prompt)
        
        # 调用LLM
        response = llm.invoke(safe_input.text)
        
        # 过滤输出
        safe_response = sanitize_output(response.content)
        
        return safe_response
    except ValueError as e:
        return f"输入验证失败: {str(e)}"
    except Exception as e:
        return f"处理过程中出现错误: {str(e)}"
```

### 4. 监控和日志
```python
import logging
from langchain.callbacks import BaseCallbackHandler
import json
from datetime import datetime

class MonitoringCallback(BaseCallbackHandler):
    def __init__(self, log_file="langchain_monitor.log"):
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.setup_logging()
        
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_time': 0,
            'errors': []
        }
    
    def setup_logging(self):
        """设置日志"""
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.metrics['total_calls'] += 1
        self.start_time = time.time()
        
        self.logger.info(f"LLM调用开始 - 输入长度: {len(str(prompts))}")
    
    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        self.metrics['total_time'] += duration
        self.metrics['successful_calls'] += 1
        
        # 估算token数量
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('usage', {})
            self.metrics['total_tokens'] += usage.get('total_tokens', 0)
        
        self.logger.info(f"LLM调用完成 - 耗时: {duration:.2f}秒")
    
    def on_llm_error(self, error, **kwargs):
        self.metrics['failed_calls'] += 1
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': type(error).__name__
        })
        
        self.logger.error(f"LLM调用失败: {error}")
    
    def get_metrics(self):
        """获取性能指标"""
        if self.metrics['total_calls'] > 0:
            success_rate = self.metrics['successful_calls'] / self.metrics['total_calls']
            avg_time = self.metrics['total_time'] / self.metrics['total_calls']
        else:
            success_rate = 0
            avg_time = 0
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'avg_time': avg_time
        }
    
    def export_metrics(self, filename="metrics.json"):
        """导出指标到文件"""
        metrics = self.get_metrics()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

# 使用监控
monitor = MonitoringCallback()
llm = ChatOpenAI(callbacks=[monitor])

# 执行操作
for i in range(5):
    llm.invoke(f"这是第{i+1}个测试请求")

# 获取指标
metrics = monitor.get_metrics()
print("性能指标:")
print(f"总调用次数: {metrics['total_calls']}")
print(f"成功率: {metrics['success_rate']:.2%}")
print(f"平均耗时: {metrics['avg_time']:.2f}秒")
print(f"总token数: {metrics['total_tokens']}")

# 导出指标
monitor.export_metrics()
```

### 5. 配置管理
```python
import os
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class LangChainConfig:
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    cache_enabled: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=os.getenv("LANGCHAIN_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LANGCHAIN_MAX_TOKENS", "1000")),
            cache_enabled=os.getenv("LANGCHAIN_CACHE", "true").lower() == "true",
            log_level=os.getenv("LANGCHAIN_LOG_LEVEL", "INFO")
        )
    
    @classmethod
    def from_yaml(cls, file_path: str):
        """从YAML文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, file_path: str):
        """保存配置到YAML文件"""
        config_data = {
            'openai_api_key': self.openai_api_key,
            'anthropic_api_key': self.anthropic_api_key,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'cache_enabled': self.cache_enabled,
            'log_level': self.log_level
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def create_llm(self):
        """创建LLM实例"""
        if self.cache_enabled:
            langchain.cache = InMemoryCache()
        
        return ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

# 使用配置
config = LangChainConfig.from_env()
llm = config.create_llm()

# 保存配置
config.to_yaml("langchain_config.yaml")

# 从文件加载配置
loaded_config = LangChainConfig.from_yaml("langchain_config.yaml")
```

---

## 总结

LangChain提供了丰富的API和组件，支持构建各种复杂的AI应用。通过合理组合这些组件，可以创建功能强大、可扩展的LLM驱动系统。

### 关键要点
1. **模块化设计**：组件可以灵活组合
2. **多模型支持**：支持各种LLM提供商
3. **丰富的集成**：与外部工具和服务深度集成
4. **生产就绪**：支持部署、监控和扩展

### 学习路径
1. 从基础组件开始（LLM、Prompt、Chain）
2. 学习高级功能（Agent、Memory、Tools）
3. 实践实际应用场景
4. 掌握最佳实践和性能优化

### 常见应用场景
- **智能客服**：自动回答用户问题
- **文档问答**：基于知识库的问答系统
- **内容生成**：营销文案、文章创作
- **数据分析**：数据分析和可视化
- **翻译服务**：多语言翻译

通过本文档的学习，您应该能够熟练使用LangChain构建各种AI应用，并根据具体需求选择合适的组件和架构。
