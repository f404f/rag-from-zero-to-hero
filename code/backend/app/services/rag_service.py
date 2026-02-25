import asyncio
from typing import Tuple, List

import requests

from app.core.config import settings
from app.services.vector_store import VectorStore


async def _call_ollama(prompt: str, temperature: float) -> str:
    """调用Ollama API"""
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=payload, timeout=30)
        )
        response.raise_for_status()
        data = response.json()
        return data.get('response', '生成答案失败')
    except Exception as e:
        return f"调用语言模型时出错: {str(e)}"


class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()

    async def process_question(
            self,
            question: str,
            top_k: int = 3,
            temperature: float = 0.0,
    ) -> Tuple[str, List[str], List[float]]:
        """
        处理用户问题的完整流程
        """
        # 1. 向量化问题
        question_vector = self.vector_store.encode_text(question)

        # 2. 检索相关文档
        results = self.vector_store.search_similar(
            question_vector,
            top_k=top_k,
            threshold=settings.SIMILARITY_THRESHOLD
        )

        if not results:
            return "抱歉，我没有找到相关的文档内容。", [], []

        # 3. 构造上下文
        contexts = [result['content'] for result in results]
        context_text = "\n\n".join(contexts)

        # 4. 调用LLM生成答案
        prompt = self._build_prompt(context_text, question)
        answer = await _call_ollama(prompt, temperature)

        # 5. 准备返回数据
        sources = [result.get('filename', f'文档{i + 1}') for i, result in enumerate(results)]
        scores = [result['score'] for result in results]

        return answer, sources, scores

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        """构建Prompt模板"""
        return f"""你是一个专业的技术文档助手。请基于以下文档内容回答问题：

文档内容：
{context}

问题：{question}

要求：
1. 用中文回答问题
2. 回答要准确、简洁
3. 如果文档中没有相关信息，请说明"未检索到足够相关内容"
4. 在回答末尾标注使用的文档来源"""
