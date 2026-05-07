from qdrant_client import QdrantClient
from app.core.config import QDRANT_COLLECTION, QDRANT_URL, QDRANT_API_KEY


def get_qdrant_client(host: str, api_key: str = "") -> QdrantClient:
    return QdrantClient(url=host, api_key=api_key)


client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)


def get_relevant_chunks(question_vector: list[float], top_k: int=5):

    return client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=question_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
