import asyncio

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHROMA_DB_PATH = "chroma_db"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

# Создание клиента ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection(label: str):
    """Возвращает или создает коллекцию для указанного label."""
    collection_name = f"label_{label}"  # Имя коллекции на основе label
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,  # используем SentenceTransformer
    )


async def upsert_to_collection(collection, documents, metadatas, ids):
    await asyncio.to_thread(collection.upsert, documents=documents, metadatas=metadatas, ids=ids)
