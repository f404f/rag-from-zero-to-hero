# 从零搭建RAG系统第4天：问题向量化 + Milvus检索匹配
# 作者：老赵全栈实战
# 环境：Miniconda（rag-env）+ Milvus v2.6.9

# 1. 导入所需模块（复用前三天的依赖，无需额外安装）
from pymilvus import connections, Collection, utility

# 2. 连接Milvus向量数据库（和Day2、Day3的连接参数一致）
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 3. 验证Milvus中的集合和数据（确认Day3存入的数据可用）
collection_name = "rag_test_collection"  # 和Day3创建的集合名称一致

# 检查集合是否存在
if not utility.has_collection(collection_name):
    print("Error：Milvus中未找到集合！请先执行Day3的代码，确保向量存入成功。")
else:
    # 加载集合到内存（检索前必须加载）
    collection = Collection(name=collection_name)
    collection.load()

    # 查看集合中的向量数量（和Day3存入的数量一致，即为成功）
    vector_count = collection.num_entities
    print(f"Milvus集合验证成功！")
    print(f"集合名称：{collection_name}")
    print(f"存入的向量数量：{vector_count}")

# 4. 初始化bge-small-zh模型（复用Day3的模型，无需重新下载）
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('BAAI/bge-small-zh')

# 5. 模拟用户问题（可自定义3-5个，贴合Day3存入的文档内容）
user_questions = [
    "RAG是什么？",
    "为什么通用的基础大模型基本无法满足实际业务需求？",
    "RAG解决了什么问题？",
]

# 6. 选择一个问题进行测试（先测试1个，后续可批量处理）
test_question = user_questions[0]
print(f"\n测试用户问题：{test_question}")

# 7. 问题向量化（和文档向量化逻辑一致，维度512）
question_embedding = embedding_model.encode(test_question)

print(f"问题向量化成功！向量维度：{len(question_embedding)}")  # 输出512，即为成功
print(f"问题向量（前10位）：{question_embedding[:10]}")

# 8. 配置Milvus检索参数（新手可直接复用，无需修改）
search_params = {
    "metric_type": "L2",  # 距离度量方式，和Day3创建索引时一致（L2欧氏距离）
    "params": {"nprobe": 10}  # 检索参数，nprobe越小，检索越快；越大，匹配越精准（新手设10即可）
}

# 9. 执行检索（核心代码）
# 参数说明：
# - data：问题向量（需用列表包裹，Milvus要求格式）
# - anns_field：检索的向量字段（和Day3集合定义的向量字段一致）
# - limit：检索返回的最相关片段数量（新手设3-5个即可，太多易冗余）
# - param：检索参数（上面配置的search_params）
# - output_fields：检索返回的字段（需包含doc_text，用于查看匹配的文档内容）
results = collection.search(
    data=[question_embedding],
    anns_field="doc_embedding",
    limit=3,
    param=search_params,
    output_fields=["doc_text"]
)

# 10. 解析检索结果（提取匹配的文档片段和匹配度）
print(f"\n检索到与问题最相关的{len(results[0])}个文档片段：")
for i, result in enumerate(results[0]):
    # 匹配度：L2距离越小，匹配度越高（距离为0时完全匹配）
    distance = result.distance
    # 匹配的文档文本
    doc_text = result.entity.get("doc_text")
    print(f"\n第{i+1}个匹配片段（距离：{distance:.4f}）：")
    print(f"文档内容：{doc_text}")