"""
该模块负责：使用 TextLoader 读取知识库文件，
并使用 LangChain 将文本切分成多个 Document。
"""
"""
知识库加载模块。
    主要职责：
        1. 读取 knowledge/knowledge.txt 文件。
        2. 使用 LangChain 的文本切分器，将文本拆分成多个文本块。
        3. Document 列表，供后续 Embedding 和向量检索使用。
"""

from pathlib import Path    #Path 是 Python 标准库 pathlib 中的类，用来处理文件路径
from langchain_community.document_loaders import TextLoader     #导入文本加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter
#导入 LangChain 的文本切分器(递归按字符分割文本)
# RecursiveCharacterTextSplitter:作用是把较长的文本拆成较小的文本块，以便后续生成向量和检索

### 1、定义文本加载切割函数
# 函数最终返回切分后的 Document 列表
# 流程：读取知识库文件 → 切分 Document → 返回 Document 列表
def load_knowledge():
    """读取知识库，并返回切分后的 Document 列表"""
    ## 1.1 获取加载知识库的地址（非固定写死）
    rag_dir = Path(__file__).resolve().parent   # 获取当前 loader.py 所在的 上一级 rag 目录
    #★★★ rag_dir 保存的是一个 Path 对象，表示 rag 文件夹的路径。
    """
    __file__：      表示当前文件（loader.py）路径, 等价于 "D:/AI项目/p02_RAG_knowledge_langchain/rag/loader.py"
    Path(__file__)：把这个文件路径转换成 Path 对象，以便操作路径(Path 对象就是专门用来处理文件路径的对象，可以获取目录、拼接路径、检查文件是否存在等)
    .resolve()：    将路径解析为绝对路径
    .parent：       获取当前路径的上一级目录
    """
    project_dir = rag_dir.parent    #获取 rag 文件夹的上一级目录，也就是项目根目录 p02_RAG_knowledge_langchain。
    knowledge_path = project_dir / "knowledge" / "knowledge.txt"    # 拼接出知识库文件knowledge.txt的完整路径

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 1.2 使用 TextLoader 读取知识库内容
    loader = TextLoader(        #TextLoader： LangChain 提供的文本文件加载器，负责读取文本文件。
        str(knowledge_path),    #把知识库文件的路径对象转换成字符串。
        encoding="utf-8"        #指定文件编码，正确读取中文。
    )   #只是创建加载器对象，还没有真正读取文件内容
    """
    文本文件加载器（属于文档加载器 Document Loader）。
    它的核心作用非常纯粹：把电脑本地的一个普通文本文件（如 .txt 文件），读取并转换成 LangChain 认识的 Document 对象。
    ⭐⭐⭐
    TextLoader(
            file_path="str/Path",         #必填：要加载的文本文件的内容或路径（可以是相对路径或绝对路径）
            encoding="str"                #选填：文件的字符编码格式(例如"utf-8", "gbk"),，默认None
            autodetect_encoding="bool"    #选填：是否开启自动检测文件编码功能，默认为False
            ) 
    """

    documents = loader.load()   #load()：读取文件并返回Document列表。
    # 对于当前这个 TXT 文件，通常先得到一个 Document。
    """
    此时的：
        documents = [
                    Document(
                        page_content="Python 是一种高级编程语言………………",
                        metadata={"source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt"}
                            )
                    ]
    其中：
        documents[0]	            第一个 Document 对象
        documents[0].page_content	文件读取到的完整文本内容
        documents[0].metadata	    文件的附加信息，通常包含 source 文件路径
    """
    """
    TextLoader().load()文档加载器的方法和上一版本的方法区别：
    
    对比	                Path.read_text()	TextLoader().load()
    ---                 ----------------     ------------------
    所属	                Python 标准库	        LangChain
    主要作用	            读取文件文本	        将文件加载为 LangChain 文档
    返回类型	            str	                    list[Document]
    是否自动创建Document	否	                       是
    是否自动切分文本	    否	                       否
    常见后续操作	        调用 split_text()	调用 split_documents()
    """

    ## 1.3 创建 LangChain 文本切分器，设置切分规则
    text_splitter = RecursiveCharacterTextSplitter(     #RecursiveCharacterTextSplitter（递归字符文本切分器）
        chunk_size=300,         # 每个文本块的目标长度
        chunk_overlap=50,       # 相邻文本块保留 50 个字符的重叠
        length_function=len,    # 使用 Python 的 len() 计算文本长度
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],    #“从哪里下刀”（切割优先级）
    )   #设置切分时优先考虑的分隔符，先尝试段落换行，再尝试换行、中文标点等；"" 表示必要时可以逐字符切分


    #⭐.split_documents(documents)才是真正执行切分的步骤；前面创建 text_splitter 只是准备好切分规则

    ## 1.4 利用利用文本切分器， 将 Document 切分成多个 Document（真正的切割）
    chunks = text_splitter.split_documents(documents)
    # split_documents()接收Document列表。返回的仍然是 Document 列表。
    """
    和上一版本相比：
     方法	                        输入类型	        返回类型	             作用
     split_text(text)	             str		    list[str]      将字符串切分成多个字符串文本块
     split_documents(documents)     list[Document]	list[Document] 将文档列表中的正文切分成多个 Document 文档块
    """

    return chunks       # 返回切分后的 Document 列表，交给后续的向量入库流程。
    """
    此时：
        chunks = [
                   Document(
                       page_content="Python 是一种高级编程语言……",
                       metadata={"source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt"}
                       ),
                   Document(
                       page_content="FastAPI 是一个基于 Python 的 Web 框架……",
                       metadata={"source": "知识库文件路径"}
                       ),
                   Document(
                       page_content="RAG 的英文全称是 Retrieval-Augmented Generation……",
                       metadata={"source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt"}
                       ),
                   Document(
                       page_content="MySQL 是一种关系型数据库管理系统……",
                       metadata={"source": "D:\\AI项目\\p02_RAG_knowledge_langchain_Redis\\knowledge\\knowledge.txt"}
                       )
                ]
        ~~~~~
        注意：
            1. chunks 的类型是 list[Document]。
            2. page_content 保存每个文本块的正文。
            3. metadata 保存文档的附加信息，例如文件来源。
            4. 实际 metadata 的内容以 TextLoader 和文本切分器返回的结果为准。
    """
    """
    本文件流程:
         knowledge.txt
         知识库文件
             ↓
         read_text()
         返回完整字符串 str
             ↓
         split_text(content)
         返回 list[str]，即多个文本块
    """

