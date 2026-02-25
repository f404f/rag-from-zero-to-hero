from fastapi import APIRouter, HTTPException
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat"])
rag_service = RAGService()


@router.post("", response_model=ChatResponse, summary="智能问答")
async def chat_endpoint(request: ChatRequest):
    """
    基于RAG的智能问答接口

    - **question**: 用户提问
    - **top_k**: 检索文档数量，默认3
    - **temperature**: 模型温度参数，默认0.0
    """
    try:
        answer, sources, scores = await rag_service.process_question(
            request.question,
            request.top_k,
            request.temperature
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            similarity_scores=scores
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时发生错误: {str(e)}"
        )
