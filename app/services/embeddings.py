from app.services.llm_clients import openai_client
import asyncio


async def get_embedding(text: str) -> list[float]:
    embedding_response = await asyncio.to_thread(
        lambda: openai_client.embeddings.create(  # только для embedding
            model="text-embedding-3-large",
            input=text
        )
    )

    return embedding_response.data[0].embedding
