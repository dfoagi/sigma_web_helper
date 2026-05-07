from fastapi import APIRouter
from pydantic import BaseModel
from app.services.qa_engine import get_answer

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    answer = await get_answer(req.message)
    return ChatResponse(answer=answer)
