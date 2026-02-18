# qa_pipeline.py - 端到端示例（检索 + 生成）
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from mfj_langchain.custom_streaming_handler import CustomStreamingHandler


def retrieve(question, top_k=3):
    connections.connect(alias='default', host='localhost', port='19530')
    collection_name = 'rag_test_collection'
    collection = Collection(collection_name)
    collection.load()
    embedder = SentenceTransformer('BAAI/bge-small-zh')
    q_emb = embedder.encode(question)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=[q_emb], anns_field='doc_embedding', limit=top_k, param=search_params,
                                output_fields=['doc_text'])
    connections.disconnect('default')
    optimized_results = [r.entity.get('doc_text') for r in results[0] if r.distance <= 0.6]
    return optimized_results


def generate_answer_local(question, contexts):
    model = OllamaLLM(base_url="http://localhost:11434",
                      model="qwen3:4b",
                      temperature=0.0,
                      callbacks=[CustomStreamingHandler()])

    prompt = build_prompt_template()
    chain = prompt | model
    chain.invoke({"question": question, "ctx_text": "\n\n".join(contexts)})


def build_prompt_template():
    template = """"你是一个专业的技术文档助手。参考下面的上下文片段回答问题，回答要简明扼要。\n\n
    上下文片段：\n{ctx_text}\n\n
    用户问题：{question}\n\n
    要求：用中文回答，若证据不足请说明。最后标注来源片段编号。"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt


if __name__ == '__main__':
    q = "什么是RAG？"
    contexts = retrieve(q, top_k=3)
    if not contexts:
        print("未检索到足够相关内容，无法生成答案。")
    else:
        generate_answer_local(q, contexts)
