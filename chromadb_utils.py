import chromadb
from chromadb.utils import embedding_functions

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # ИЗМЕНЕНО
CHROMA_DB_PATH = "chroma_db"  # Папка, где будет храниться база данных ChromaDB

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME)  # ИЗМЕНЕНО

# Создание клиента ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection(label: str):
    """Возвращает или создает коллекцию для указанного label."""
    collection_name = f"label_{label}"  # Имя коллекции на основе label
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,  # используем SentenceTransformer
    )
