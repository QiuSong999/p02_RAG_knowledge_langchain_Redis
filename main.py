
"""总入口"""

"""
RAG 知识库问答项目入口。
    执行流程：
        1. 接收用户问题。
        2. 检索知识库中的相关文本。
        3. 将问题和检索结果交给大模型。
        4. 输出最终答案。
同时提供 FastAPI 服务
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.generator import generate_answer


### 1.创建 FastAPI 应用对象，负责接收 HTTP 请求
app = FastAPI(title="RAG 知识库问答系统")


### 2. 定义用户请求的数据结构
class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, description="用户提出的问题")


### 3. 定义 RAG 问答入口函数
def rag_qa(question: str) -> str:
    """接收用户问题，返回知识库问答结果。"""

    # 调用生成模块，内部会完成检索和大模型回答
    answer = generate_answer(question)

    return answer

### 4. # 定义 POST 接口，接收用户问题并返回知识库回答
@app.get("/")
def root():
    return {"introduce":"welcome to root"}

@app.post("/ask")
def rag_endpoint(request: QuestionRequest):
    # 获取用户提交的问题
    question = request.question

    # 调用 RAG 问答函数
    answer = rag_qa(question)

    # 将答案作为 JSON 数据返回
    return {"question": question, "answer": answer}


### 5. 直接运行当前文件时，启动 FastAPI 服务
if __name__ == "__main__":
    import uvicorn

    # 启动 FastAPI 服务，允许本机浏览器访问
    uvicorn.run(app, host="127.0.0.1", port=8000)
