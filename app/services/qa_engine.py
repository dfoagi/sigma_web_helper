import asyncio

from app.services.qdrant import get_relevant_chunks
from app.services.embeddings import get_embedding
from app.services.llm import ask_llm

# MODEL = "gpt-5-mini-2025-08-07"
MODEL = "gemini-3.1-flash-lite-preview"
TOP_K = 5


def get_current_topk():
    return TOP_K


async def get_answer(user_question: str):

    question_vector = await get_embedding(user_question)

    top_k = get_current_topk()
    model = MODEL

    relevant_chunks = get_relevant_chunks(
        question_vector=question_vector,
        top_k=top_k
    )

    chapter_ids = " ".join(str(p.id) for p in relevant_chunks.points)
    chapter_scores = " ".join(str(p.score) for p in relevant_chunks.points)
    context_text = "\n\n".join(p.payload['text'] for p in relevant_chunks.points)

    answer, prompt_tokens, response_tokens, used_model = await ask_llm(
        context=context_text,
        user_question=user_question,
        model=model
    )

    print(f"prompt_tokens: {prompt_tokens}, response_tokens: {response_tokens}, used_model: {used_model}")

    return answer
    # return answer, chapter_ids, chapter_scores, prompt_tokens, response_tokens, used_model


if __name__ == "__main__":
    question = input()
    answer = asyncio.run(get_answer(question))
    print(answer)
