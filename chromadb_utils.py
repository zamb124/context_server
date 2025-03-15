import asyncio
import sqlite3
from typing import Dict, Any

import chromadb
from chromadb.utils import embedding_functions

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHROMA_DB_PATH = "chroma_db"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

# Создание клиента ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_db_connection():
    """
    Получает соединение с базой данных SQLite.
    """
    conn = sqlite3.connect(f'{CHROMA_DB_PATH}/chroma.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn


def get_collection(label: str):
    """Возвращает или создает коллекцию для указанного label."""
    collection_name = f"label_{label}"  # Имя коллекции на основе label
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,  # используем SentenceTransformer
    )


async def upsert_to_collection(collection, documents, metadatas, ids):
    await asyncio.to_thread(collection.upsert, documents=documents, metadatas=metadatas, ids=ids)


def _process_contains(key: str, search_string: str) -> Dict[str, Any]:
    """
    Выполняет поиск в базе данных по заданному ключу и части значения.
    Возвращает условие $in для поиска.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = f"""
        SELECT DISTINCT
            CASE
                WHEN string_value IS NOT NULL THEN string_value
                WHEN int_value IS NOT NULL THEN int_value
                WHEN float_value IS NOT NULL THEN float_value
                WHEN bool_value IS NOT NULL THEN bool_value
            END AS value
        FROM embedding_metadata
        WHERE key = ?
          AND (
              lower(string_value) LIKE lower(?) OR
              CAST(int_value AS TEXT) LIKE lower(?) OR
              CAST(float_value AS TEXT) LIKE lower(?) OR
              CAST(bool_value AS TEXT) LIKE lower(?)
          )
    """
    cursor.execute(query, (key, f'%{search_string}%', f'%{search_string}%', f'%{search_string}%', f'%{search_string}%'))
    values = [row['value'] for row in cursor.fetchall() if row['value'] is not None]
    conn.close()

    if values:
        return {"$in": values}
    else:
        # Если ничего не найдено, возвращаем условие, которое не может быть выполнено
        return {"$eq": "impossible_value"}


def _transform_recursive(condition: Any) -> Any:
    """
    Рекурсивно преобразует условие, заменяя оператор $contains на $in
    и обрабатывая операторы $and и $or.
    """
    if isinstance(condition, dict):
        new_condition = {}
        for key, value in condition.items():
            if isinstance(value, dict) and "$contains" in value:
                # Обработка оператора $contains
                search_string = value["$contains"]
                transformed_value = _process_contains(key, search_string)
                if transformed_value != {"$eq": "impossible_value"}:
                    new_condition[key] = transformed_value
            elif key in ("$and", "$or"):
                # Рекурсивная обработка списков условий
                transformed_list = [_transform_recursive(item) for item in value]
                filtered_list = [item for item in transformed_list if item]  # Фильтрация пустых условий
                if filtered_list:
                    new_condition[key] = filtered_list
            else:
                new_condition[key] = _transform_recursive(value)

        # Упрощаем конструкцию $and, если остался только один элемент
        if "$and" in new_condition and len(new_condition["$and"]) == 1:
            return new_condition["$and"][0]
        elif not new_condition:
            return None
        else:
            return new_condition
    else:
        return condition


def transform_where_clause(where: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует условия поиска:
    - Заменяет оператор $contains на $in рекурсивно.
    - Оборачивает условия в $and, если их более одного.
    """
    if len(where) > 1 and "$and" not in where and "$or" not in where:
        transformed_where = _transform_recursive({'$and': [{k: v} for k, v in where.items()]})
    else:
        transformed_where = _transform_recursive(where)

    return transformed_where if transformed_where else {}
