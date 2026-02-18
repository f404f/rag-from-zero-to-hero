# 演示如何以stream形式使用LangChain
from langchain_classic.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 1. 初始化Ollama LLM，启用流式输出
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="qwen3:8b",
    callbacks=[StreamingStdOutCallbackHandler()],  # 流式输出回调
    temperature=0.7
)

print("=== 测试1：基本流式输出 ===")
print("问题：请用中文列出 RAG 流程步骤")
print("回答：")

# 2. 执行流式调用
response = llm.invoke("请用中文列出 RAG 流程步骤")
print("\n")



print("=== 测试2：使用LangChain的流式生成器 ===")
import sys


# 自定义流式回调处理器
class CustomStreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        sys.stdout.write(token)
        sys.stdout.flush()


# 初始化带自定义回调的LLM
custom_llm = Ollama(
    model="qwen3:8b",
    callbacks=[CustomStreamingHandler()],
    temperature=0.7
)

print("问题：什么是检索增强生成(RAG)？")
print("回答：")
custom_llm.invoke("什么是检索增强生成(RAG)？")
print("\n")

print("=== 测试3：在RAG流程中使用流式输出 ===")

# 构建提示模板
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="基于以下上下文回答问题：\n\n{context}\n\n问题：{question}\n\n回答："
)

# 创建LLM Chain
chain = LLMChain(
    llm=custom_llm,
    prompt=prompt_template
)

# 模拟检索到的上下文
context = "检索增强生成（RAG）是指对大语言模型输出进行优化，使其能够在生成响应之前引用训练数据来源之外的权威知识库。"
question = "RAG的主要作用是什么？"

print(f"问题：{question}")
print("回答：")
chain.run(context=context, question=question)
print("\n")

print("=== 测试4：使用异步流式输出 ===")
import asyncio
from langchain_community.llms import Ollama


async def async_streaming_example():
    llm = Ollama(
        model="qwen3:8b",
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.7
    )

    print("问题：RAG有哪些优势？")
    print("回答：")
    await llm.agenerate(["RAG有哪些优势？"])


# 运行异步示例
asyncio.run(async_streaming_example())
print("\n")

print("=== 总结 ===")
print("使用LangChain实现流式输出的关键步骤：")
print("1. 初始化LLM时添加StreamingStdOutCallbackHandler或自定义回调")
print("2. 直接调用LLM或通过Chain调用，会自动启用流式输出")
print("3. 可根据需要自定义回调处理器来控制输出格式")
print("4. 支持异步流式输出，使用agenerate方法")
