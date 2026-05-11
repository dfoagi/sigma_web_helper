from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.qa_engine import get_answer
from app.core.security import verify_api_key

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: str = Depends(verify_api_key)):
    answer = await get_answer(req.message)
    return ChatResponse(answer=answer)
