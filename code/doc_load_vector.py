# 导入文档加载所需模块（LangChain内置）
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载本地TXT文档（替换为你的文档路径）
loader = TextLoader(r".\test_data\test_rag.txt", encoding="utf-8")
# 加载文档内容
documents = loader.load()

# 2. 拆分文档（关键：避免文本过长，影响向量化和后续检索效果）
# 拆分规则：按字符递归拆分，_chunk_size=200（每段200字），chunk_overlap=20（两段重叠20字，保证上下文连贯）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
# 执行拆分，得到拆分后的文档片段
split_docs = text_splitter.split_documents(documents)
# 打印拆分结果（验证加载和拆分成功）
print("文档加载成功，拆分后片段数：", len(split_docs))
print("\n拆分后的第一个片段：")
for i, doc in enumerate(split_docs):
    print(f"----文档：{i + 1}----")
    print(doc.page_content)



# 导入向量化模型模块
from sentence_transformers import SentenceTransformer
# 初始化bge-small-zh模型（轻量中文模型，本地可跑，无需联网调用API）
# 第一次运行会自动下载模型（约300MB，国内网络可正常下载，若慢可改用清华源镜像）
embedding_model = SentenceTransformer('BAAI/bge-small-zh')
# 3. 文本向量化：将拆分后的文档片段转化为向量（维度：512维，bge-small-zh默认输出）
# 提取文档片段的文本内容
doc_texts = [doc.page_content for doc in split_docs]
# 执行向量化（生成向量列表，每个向量对应一个文档片段）
doc_embeddings = embedding_model.encode(doc_texts)
# 打印向量化结果（验证向量化成功）
print("\n文本向量化成功，向量数量：", len(doc_embeddings))
print("每个向量维度：", len(doc_embeddings[0]))  # 输出512，即为成功
print("第一个文档片段的向量（前10位）：", doc_embeddings[0][:10])


# 导入Milvus相关模块（Day1已安装pymilvus）
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
# 4. 连接Milvus向量数据库（衔接Day2的部署，参数和Day2验证时一致）
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
# 5. 在Milvus中创建集合（类似数据库的表，用于存储文档向量和对应文本）
# 集合名称：rag_test_collection（可自定义，建议语义化）
collection_name = "rag_test_collection"
# 定义集合的字段（3个核心字段：主键ID、文档文本、文档向量）
fields = [
    # 主键ID：唯一标识，自增
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 文档文本：存储拆分后的文档片段（方便后续检索后展示原文）
    FieldSchema(name="doc_text", dtype=DataType.VARCHAR, max_length=600),  # 最大长度600，可根据需求调整
    # 文档向量：存储向量化后的向量，维度512（和bge-small-zh模型输出一致）
    FieldSchema(name="doc_embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]

# 定义集合 schema
schema = CollectionSchema(fields=fields, description="RAG测试集合：存储文档片段和对应向量")
# 检查集合是否已存在，若存在则删除（避免重复创建，新手可直接执行）
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
# 创建集合
collection = Collection(name=collection_name, schema=schema)
# 6. 准备存入Milvus的数据（格式：列表，对应集合的3个字段，id自增无需传入）
milvus_data = [
    doc_texts,  # 对应doc_text字段
    doc_embeddings.tolist()  # 对应doc_embedding字段，转化为列表格式（Milvus要求）
]
# 7. 将数据插入Milvus集合
collection.insert(milvus_data)
# 8. 创建索引（关键：后续检索需要索引，否则检索速度极慢）
index_params = {
    "index_type": "IVF_FLAT",  # 基础索引类型，新手首选，简单易操作
    "metric_type": "L2",       # 距离度量方式，L2为欧氏距离（适合向量匹配）
    "params": {"nlist": 128}   # 索引参数，默认128即可
}
# 为向量字段创建索引
collection.create_index(field_name="doc_embedding", index_params=index_params)
# 加载集合到内存（检索前必须加载）
collection.load()
# 打印存入结果（验证向量存入成功）
print("\n向量存入Milvus成功！")
print("Milvus集合名称：", collection_name)
print("存入的向量数量：", collection.num_entities)  # 和文档片段数、向量数量一致，即为成功
# 关闭Milvus连接（可选，后续开发可保持连接）
# connections.disconnect("default")