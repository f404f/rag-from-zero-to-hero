# 导入文档加载所需模块（LangChain内置）
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载本地TXT文档（替换为你的文档路径）
loader = TextLoader(".\\test_data\\test_rag.txt", encoding="utf-8")
# 加载文档内容
documents = loader.load()

# 2. 拆分文档（关键：避免文本过长，影响向量化和后续检索效果）
# 拆分规则：按字符递归拆分，_chunk_size=200（每段200字），chunk_overlap=20（两段重叠20字，保证上下文连贯）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)
# 执行拆分，得到拆分后的文档片段
split_docs = text_splitter.split_documents(documents)
# 打印拆分结果（验证加载和拆分成功）
print("文档加载成功，拆分后片段数：", len(split_docs))
print("\n拆分后的第一个片段：")
for i, doc in enumerate(split_docs):
    print(f"----{i + 1}----")
    print(doc.page_content)
