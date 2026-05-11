import os
from dotenv import load_dotenv

load_dotenv(override=True)

PROXYAPI_KEY = os.getenv("PROXY_API_KEY")
OPENAI_BASE_URL = os.getenv("PROXY_API_BASE_URL_OPENAI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION=os.getenv("COLLECTION_NAME")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
