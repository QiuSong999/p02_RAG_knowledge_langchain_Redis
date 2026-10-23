
"""把知识库和向量存入Redis 向量库；根据用户问题检索相关文本"""
"""
知识库检索模块。
    主要职责：
        1. 调用 loader.py，读取知识库文件并切分成 Document 文档块。
        2. 使用 Embedding 模型将文档块转换成向量，并将文档及向量存入 Redis 向量库。
        3. 根据用户问题进行向量相似度检索，返回相关的 Document 文档块。
"""

### 1. 导入所需的模块和对象
from langchain_core.documents import Document
# 每个 Document 对象主要包含两个部分：
# page_content：文档的正文文本；
# metadata：文档的附加信息，例如来源、编号、分类等。

from langchain_redis import RedisConfig, RedisVectorStore
# RedisConfig：用于配置 Redis 向量存储，例如 Redis 地址、索引名称等。
# RedisVectorStore：LangChain 提供的 Redis 向量存储类，
# 负责文档向量化后的存储，以及后续的相似度检索。

from rag.loader import load_knowledge   #导入加载、切分知识库的函数。该函数返回切分后的 Document 列表
from rag.embedding import embeddings    #导入向量模型


### 2. 加载知识库Document块
chunks = load_knowledge()   #类型是 list[Document]
# 引用 loader 文件中的 load_knowledge()，
# 该函数现在返回的是切分好的 Document 对象列表，而不是字符串列表。
"""
此时：
    chunks = [
        Document(
            page_content="Python 是一种高级编程语言……",
            metadata={"source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt"}
        ),
        Document(
            page_content="FastAPI 是一个基于 Python 的 Web 框架……",
            metadata={"source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt"}
        ),
        ...
    ]
"""

### 3. 为每个 Document 文档块的 元数据 添加一个 segment_id 字段
# 需要为每个 Document 添加 segment_id，标识文本块的编号。
for index, document in enumerate(chunks, start=1):
    # 遍历 Document 列表，为每个文本块生成从 1 开始的编号。
    # document 是一个 Document 对象，不是字符串。

    document.metadata["segment_id"] = str(index)    # 在原有 metadata 的基础上添加 segment_id。
    # 使用 str(index) 将编号转换成字符串。 这样可以保留 TextLoader 提供的 source 等已有元数据。

documents = chunks  # 将编号后的 Document 列表赋值给 documents，后续 Redis 入库逻辑继续使用 documents。
"""
每个切分后的文档块分配一个标识，方便区分和追踪不同的文档块
作用：
    区分文档块：为每个文档块分配一个序号。
    辅助调试：检索结果返回后，可以查看命中了哪个文档块
    辅助追踪：结合 source，可以了解文档块来自哪个文件、对应哪个切分序号。
"""
"""
处理后的数据示例：
    [
        Document(
                page_content="Python 是一种高级编程语言……",
                metadata={
                        "source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt",
                        "segment_id": "1"       ###为metadata新增了"segment_id"
                          }
                ),
        Document(
                page_content="FastAPI 是一个基于 Python 的 Web 框架……",
                metadata={
                    "source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt",
                    "segment_id": "2"
                          }
                ),
        ...
    ]
    
    注意：
        1. page_content 不需要重新生成，保留原来的文本内容。
        2. metadata 原有的 source 等字段继续保留。
        3. segment_id 是本项目额外添加的文本块编号。
        4. documents 仍然是 list[Document]，可以直接交给 RedisVectorStore。
    """


### 4. 创建 Redis 配置对象
config = RedisConfig(
    index_name="rag_v3_knowledge",           # 指定本项目 V3 使用的向量索引名称
    redis_url="redis://192.168.88.130:6379"  # Redis 服务器的连接地址
)
#如果我们在创建 RedisConfig 时没有显式指定 key_prefix（键名前缀），它就会自动默认将 key_prefix 设置为与 index_name 完全相同
"""
127.0.0.1:6379> keys *
rag_v3_knowledge:01M56CX233W8HG7QK7Y74VXJKG     #rag_v3_knowledge是键的前缀
rag_v3_knowledge:01M56CX233W8HG7QK7Y74VXJKF     #输出的整体是键
rag_v3_knowledge:01M56CX233W8HG7QK7Y74VXJKH
rag_v3_knowledge:01M56CX233W8HG7QK7Y74VXJKJ
127.0.0.1:6379> FT._LIST
rag_v3_knowledge    #这查看的才是索引，只不过设置索引的后，默认键名前缀和索引保持相同
"""

### 5. 定义创建 Redis 向量存储并写入文档的函数
# embedding 使用项目中已经初始化好的 embeddings 对象。
# 将入库操作封装成函数，避免导入本模块时自动执行写入。
# 只有主动调用 build_knowledge_base() 时，才会执行文档向量化和写入 Redis。
def build_knowledge_base():
    vector_store = RedisVectorStore.from_documents(
        documents=documents,  # 已经封装好的 Document 文档列表
        embedding=embeddings,  # 使用项目中初始化好的 Embedding 模型
        config=config,  # 指定 Redis 地址和本项目的索引名称
    )
    return vector_store  # 返回向量存储对象RedisVectorStore，供后续操作使用

"""from_documents() 负责从文档创建并写入向量存储；from_existing_index() 负责连接已有索引"""

### 6. 连接 Redis 中已经存在的向量索引index（而不是重新将文档写入 Redis）
# 如果你只想查询知识库，就可以连接已有索引，不必每次重新入库。
def connect_knowledge_base():
    # 连接已有索引，不重新写入整批文档。
    vector_store = RedisVectorStore.from_existing_index(
        index_name="rag_v3_knowledge",
        embedding=embeddings,
        redis_url="redis://192.168.88.130:6379",
    )
    return vector_store #返回一个 RedisVectorStore 对象
# 它与前面 build_knowledge_base() 返回的对象类型相同，但用途不同：
# - build_knowledge_base()：将文档向量化并写入 Redis。
# - connect_knowledge_base()：连接已有索引，供后续检索使用。
"""
⭐⭐⭐
from_existing_index() 用于连接已有索引，不会像 from_documents() 那样重新写入整批文档。
    index_name: 要连接的 Redis 向量索引名称，必须与之前创建索引时的名称一致。
    embedding：Embedding 模型对象，用于将用户的问题转换成向量，
              再与 Redis 中已经存储的文档向量进行相似度检索。
    redis_url：Redis 服务器的连接地址，格式通常为 redis://主机地址:端口。

注意：
    Redis 中必须已经存在对应的索引，否则连接时会报 Unknown index name。
    因此，不能在模块顶层直接执行 from_existing_index()，
    而是将连接操作封装为函数，在真正需要检索时再调用。
    
    在进行向量检索（无论是通过 RedisVectorStore.from_existing_index 还是其他向量数据库的连接方式）时，index_name（索引名）是不可或缺的核心凭证。
    像 Redis（结合 RediSearch 模块）这类向量数据库，它的内部可以同时存放成百上千个不同的业务数据集：
    如果你只告诉数据库：“我要找最相关的文本”，但不告诉它去哪个索引（抽屉）里找，数据库根本不知道该去比对哪一批向量数据。
    只有传入了正确的 index_name（比如你代码里的 "rag_v3_knowledge"），数据库才能瞬间锁定对应的底层 HNSW 向量索引结构，去执行数学相似度计算。
"""


### 7. 根据用户问题检索相关文档
def retrieve_documents(question: str, k: int = 2) -> list[Document]:
    """根据问题检索最相关的 k 个文本块。"""
    """
    根据用户的问题，在向量数据库中检索最相关的 k 个文本块。
    参数解释:
        question (str): 用户的查询问题文本。
        k (int, optional): 返回的最相关文本块数量，默认为 2。

    返回内容:
        list[Document]: 匹配到的 Document 对象列表（包含 page_content 和 metadata）。

    Note (核心机制说明):
        此处的 vector_store 实例在初始化时（无论是通过 from_documents() 写入，
        还是通过 from_existing_index() 加载），就已经与特定的 index_name
        （例如 "rag_v3_knowledge"）进行了强绑定。
        因此，在调用 similarity_search() 时不需要再次传入索引名，
        它会自动去其所属的 Redis 索引命名空间中执行向量相似度计算。
    """

    ##7.1 连接 Redis 向量库
    vector_store = connect_knowledge_base()     # 连接 Redis 中已经存在的索引。
    #调用上一步,返回 RedisVectorStore 对象。这里不会重新加载知识库，也不会重新写入文档。
    # 注意：只有索引创建成功后，才能执行这一步。

    #7.2 执行相似度检索
    results = vector_store.similarity_search(
        query=question,
        k=k,
    )
    return results


### 8. 单独运行本文件时，测试知识库入库和检索功能
if __name__ == "__main__":
    #如果第一次执行，要主动创建向量存储，并将知识库文档写入 Redis。
    # vector_store = build_knowledge_base()

    #如果非第一次执行，redis已有数据，则用
    vector_store = connect_knowledge_base()

    # 第二步：测试刚刚写入的知识库能否正常检索。
    # 直接使用 build_knowledge_base() / connect_knowledge_base()返回的对象，不需要再次连接已有索引。
    results = vector_store.similarity_search(
        query="FastAPI 是什么？",
        k=2,
    )
    print(results)
    """
    result的内容是：
        [
          Document(metadata={'segment_id': '2'}, 
                  page_content='FastAPI 基础知识\n\nFastAPI 是一个基于 Python 的 Web 框架，常用于开发 Web API。
                               它支持根据类型注解校验请求数据，并能自动生成交互式 API 文档。\nFastAPI 通常使用路由函数处理 HTTP 请求。
                               例如，使用 @app.get("/items") 可以定义一个处理 GET 请求的接口。\n\nRAG 基础知识'), 
          Document(metadata={'segment_id': '1'}, 
                 page_content='Python 基础知识\n\nPython 是一种高级编程语言，具有语法简洁、易于学习、生态丰富等特点。
                 Python 常用于 Web 开发、数据分析、人工智能和自动化脚本开发。\nPython 使用变量保存数据。变量不需要提前声明类型，
                 Python 会根据赋值内容确定变量的类型。例如，name = "张三" 创建一个字符串变量，age = 18 创建一个整数变量。\nPython 
                 使用函数组织可重复执行的代码。通过 def 关键字可以定义函数，函数可以接收参数，也可以使用 return 返回结果。\n\nFastAPI 基础知识')
        ]
    """


    # 逐个遍历查询到的检索结果，检查返回的 Document 对象。
    for index, result in enumerate(results, start=1):
        print(f"\n--- 第 {index} 个检索结果 ---")
        print("数据类型：", type(result))
        print("文本内容：", result.page_content)
        print("元数据：", result.metadata)
