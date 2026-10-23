"""
前面loader.py文件中的函数把知识库中的内容切分成了文本块，形成文本块列表
例如：
     [
        "Python 是一种高级编程语言……",
        "FastAPI 是一个基于 Python 的 Web 框架……",
        "RAG 的英文全称是 Retrieval-Augmented Generation……",
        "MySQL 是一种关系型数据库管理系统……"
                ]
本文件要把这些文本转换成数字向量，这个过程叫 Embedding（文本向量化）
"""

"""
embedding:文本向量化模块。
    主要职责：
        1. 读取 .env 中的 DashScope API Key。
        2. 配置 Embedding 模型。
        3. 返回可用于文本向量化的对象。
"""

import os   #导入 Python 的 os 模块，后面用它读取环境变量中的 API Key
from dotenv import load_dotenv      # 导入 load_dotenv() 函数，用来读取 .env 文件中的配置
from langchain_openai import OpenAIEmbeddings


load_dotenv()   #读取项目根目录下的 .env 文件

### 1、 创建 Embedding 模型对象
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("dashscope_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False    #使用 OpenAIEmbeddings 对接非 OpenAI 官方、但兼容 OpenAI 接口的第三方大模型服务
)                                       #（如阿里云百炼 DashScope、vLLM、Ollama、OpenRouter 等）时一个非常关键的“避坑”配置
# OpenAIEmbeddings 默认可能把文本转换成 Token ID 再发送；
# DashScope 的这个兼容接口要求输入文本字符串。设置为 False 后，会关闭这套自动分词和长度处理流程，直接发送文本。

"""
之前学的：
    embeddings = DashScopeEmbeddings(             
                model="text-embedding-v3",                
                dashscope_api_key=os.getenv("dashscope_api_key")
                )
    和上面是等效的。
    
OpenAIEmbeddings(...)：通过 DashScope 的 OpenAI 兼容接口调用模型。
DashScopeEmbeddings(...)：通过 DashScope 专用集成调用模型。
"""
