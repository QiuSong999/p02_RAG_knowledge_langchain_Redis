"""
模块分工：
    loader.py：读取知识库并切分文本。
    retriever.py：将文档存入 Redis，并提供向量检索功能。
    generator.py：检索相关文档，将问题和资料交给大模型生成答案。
"""
"""
大模型答案生成模块。
    主要职责：
        1. 接收用户问题。
        2. 接收检索到的知识库文档。
        3. 将问题和文档交给大模型。
        4. 返回大模型生成的答案。
"""


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough

from langchain_core.runnables import RunnableLambda
# 用于把普通 Python 函数包装成 LangChain 的 Runnable 对象

from rag.retriever import retrieve_documents


load_dotenv()       # 读取 .env 中的环境变量

### 1. 创建聊天模型对象
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("dashscope_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

### 2. 创建检索器：根据用户问题，检索最相关的 2 个文档
# ⭐使用 RunnableLambda 将 lambda 函数包装成 Runnable 对象，供 LCEL 链调用。
retriever = RunnableLambda(         #RunnableLambda(...)：把这个retrieve_documents(question, k=2)   函数包装成 LangChain 的 Runnable 对象。
    lambda question: retrieve_documents(question, k=2)      #lambda question：定义一个匿名函数，接收一个参数 question，表示用户的问题。
)               #retrieve_documents(question, k=2)：调用检索函数，根据用户的问题，从 Redis 向量库中检索最相关的 2 个 Document 文档块。
# 普通 Python 函数也可以直接放在 LCEL 的 | 链中，由 LangChain 自动转换为 Runnable。
# 这里使用 RunnableLambda，是为了将匿名函数显式包装成 Runnable，并在调用检索函数时指定 k=2。


### 3. 创建提示词模板
prompt = ChatPromptTemplate.from_template(
    """
    你是一个知识库问答助手。
    请根据知识库资料回答用户的问题。
    如果资料中没有相关信息，请明确说明，不要编造答案。

    知识库资料：
    {context}

    用户问题：
    {question}
    """
)

### 4. 定义文档格式化函数：只提取正文，避免把 metadata 等信息传给大模型
def format_docs(docs):  #docs是要上传的检索到的Document对象，它是一个 list[Document]
    return "\n\n".join(doc.page_content for doc in docs)
    #doc.page_content for doc in docs：生成器表达式，用于遍历 docs，每次取出一个Document的正文
    # "\n\n"：连续两个换行符，通常用于在两段文本之间空一行。
    """
    ⭐⭐⭐
    用法：'连接符'.join(可迭代对象) 
        把一个可迭代对象（比如列表、元组）里的所有字符串元素，用指定的字符串连接起来
        
        # 用空字符串连接（直接拼在一起）
            words = ["我", "喜欢", "吃", "苹果"]
            result = "".join(words)
            print(result)  # 输出: 我喜欢吃苹果
        
        # 用逗号连接
            result2 = ",".join(words)
            print(result2) # 输出: 我,喜欢,吃,苹果
        
        # 用换行符连接
            result3 = "\n".join(["第一行", "第二行"])
            print(result3)
                # 输出:
                # 第一行
                # 第二行
    """
rag_chain = (
    {
        # 将用户问题传给检索函数，并把检索到的 Document 列表格式化成文本。
        "context": retriever | format_docs,
        #format_docs 是普通函数，可以直接写在 LCEL 的 | 后面，由 LangChain 自动转换为 Runnable。
        "question": RunnablePassthrough(),  # 原样保留用户问题，作为提示词中的 question。
    }
    | prompt
    | llm
)
"""
为啥都是在chain中，retrieve_documents函数需要包装成Runnable，而format_docs函数不需要 ？
retriever 连在链条的最前端：它直接面对的是外部输入的 str（字符串），而 retrieve_documents 需要明确传参 k=2，
如果不显式用 RunnableLambda 包装并定义好参数传递关系，LangChain 无法直接把字符串灌给它。

format_docs 处在链条的中间：它紧跟在 retriever 后面，上一步输出的 list[Document] 可以无缝、直接传给 format_docs(docs)，
所以 LangChain 允许你偷懒，由框架在底层自动完成了隐式转换。
"""

### 5. 定义答案生成函数
def generate_answer(question: str) -> str:
    """输入用户问题，返回 RAG 生成的答案。"""
    response = rag_chain.invoke(question)

    return response.content

if __name__ == '__main__':
    # print(generate_answer("你是谁"))
    print(generate_answer("简单介绍RAG"))

"""
对比	                    只检索	    RAG（检索 + 大模型）
返回内容	                原始知识文本	根据资料生成的答案
是否调用大模型	            否	            是
响应成本	                较低	            较高
能否归纳、解释、组织语言	有限	            可以
是否适合直接返回精确记录	适合	          不一定需要
"""

