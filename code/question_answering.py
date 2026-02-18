from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from mfj_langchain.custom_streaming_handler import CustomStreamingHandler

template = """"你是一个专业的技术文档助手。参考下面的上下文片段回答问题，回答要简明扼要。\n\n
上下文片段：\n{ctx_text}\n\n
用户问题：{question}\n\n
要求：用中文回答，若证据不足请说明。最后标注来源片段编号。"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(base_url="http://localhost:11434",
                  model="llama3.2:latest",
                  callbacks=[CustomStreamingHandler()])

chain = prompt | model

optimized_results = [
    {"匹配度": 0.12, "文档片段": "RAG 是检索增强生成框架，将检索到的相关文档作为上下文提供给 LLM。"},
    {"匹配度": 0.33, "文档片段": "RAG 可以降低模型幻觉，提高事实性回答。"}
]
contexts = [r['文档片段'] for r in optimized_results[:3]]
chain.invoke({"question": "什么是RAG？", "ctx_text": "\n\n".join(contexts)})
