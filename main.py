# --- START OF FILE main.py ---
# --- Дополнительные библиотеки ---
import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any, List

import nltk

nltk.download('bcp47')
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI, HTTPException, Header, status, Body, \
    Depends

from chromadb_utils import get_collection, upsert_to_collection
from config import config  # Убедитесь, что config.py содержит CHAT_TOKEN и TELEGRAM_BOT_TOKEN
from models import ValidLabels, DocumentBase, Query, ContextResponse, ForceSaveResponse
# Import the new module
from telegram_integration import telegram_integration
from text_utils import split_text_semantically

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# --- Импорт Transformers и Pipeline ---
from transformers import pipeline
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('bcp47')
# --- Конфигурация ---
CHROMA_DB_PATH = "chroma_db"  # Папка, где будет хранится база данных ChromaDB
TEMP_STORAGE_PATH = "temp_telegram_data"  # Папка для временного хранения данных Telegram
SAVE_INTERVAL_SECONDS = 69  # Интервал сохранения в секундах (10 минут)

# --- Токен авторизации ---
CHAT_TOKEN = config.CHAT_TOKEN
if not CHAT_TOKEN:
    raise ValueError("Не установлена переменная окружения CHAT_TOKEN. Пожалуйста, настройте ее в файле .env.")

TELEGRAM_BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не установлена переменная окружения TELEGRAM_BOT_TOKEN. Пожалуйста, настройте ее в файле .env.")

# --- Инициализация логирования ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    """
    # Start telegram integration
    asyncio.create_task(telegram_integration.start())  # Call the method to get a coroutine
    yield
    # Stop telegram integration
    await telegram_integration.stop()


app = FastAPI(lifespan=lifespan)


# --- Зависимость для аутентификации по токену ---
async def verify_token(authorization: Optional[str] = Header(None)):
    """
    Проверяит токен из заголовка Authorization (схема Bearer).
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует заголовок Authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверная схема аутентификации. Используйте Bearer.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный формат заголовка Authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token != CHAT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный токен аутентификации",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


class AddDocumentRequest(BaseModel):
    """
    Модель запроса для добавления дакумента.
    """
    text: str
    label: ValidLabels
    document_id: Optional[str] = None
    metadata: DocumentBase
    chunk: Optional[bool] = True
    author: Optional[str] = None


@app.post("/add_document/", dependencies=[Depends(verify_token)])
async def add_document(request_data: AddDocumentRequest = Body(...)):
    """
        Добавляет текст как дакумент в векторное хранилище, принимая JSON payload.

        - **text**: Текст документа для добавления.
        - **label**: Указание источника дакумента (hubspot, telegram, wiki, startrek). Обязательное поле.
        - **document_id**: Уникальный идентификатор дакумента. Если не указан, генерируется автоматически.
        - **metadata**: Словарь с метаданными дакумента.  Дополнительные поля будут сохранены.
    """
    try:
        text = request_data.text
        label = request_data.label
        document_id = request_data.document_id
        metadata = request_data.metadata

        # Generate document_id if not provided
        document_id = document_id or f"doc_{label}_{hash(text)}"

        collection = get_collection(label.value)
        # Удаляем старые чанки дакумента, если они есть
        collection.delete(where={"source_document_id": document_id})

        combined_metadata = json.loads(metadata.model_dump_json())

        if request_data.chunk:  # По умолчанию разбиваем на чанки, если не указано "chunk": false
            # Разбиваем на чанки
            chunks = await split_text_semantically(text)
            meta_end_text = '[META: ' + ','.join([f'{k}: {v}' for k, v in combined_metadata.items()]) + ']'
            chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]

            metadatas = []
            for i in range(len(chunks)):
                chunks[i] = chunks[i] + '\n' + meta_end_text
                chunk_metadata = combined_metadata
                chunk_metadata["source_document_id"] = document_id
                metadatas.append(chunk_metadata)
            await upsert_to_collection(collection, chunks, metadatas, chunk_ids)
        else:
            # Сохраняем файл целиком
            combined_metadata["source_document_id"] = document_id  # Добавляем source_document_id
            await upsert_to_collection(collection, [text], [combined_metadata], [document_id])

        return {"message": f"Дакумент успешно обработан и добавлен с label {label.value} и ID {document_id}."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def create_prompt(question, context):
    """Создает prompt для фильтрации контекста."""
    prompt = f"""
    You are provided with a question and context derived from the knowledge base.
    Your task is to select from the context only the information that is most relevant to answer the question.
    Leave only meaningful suggestions and exclude unnecessary details.
    question: {question}
    context: {context}
    relevant context:
    """
    return prompt


# --- New endpoint for context compression ---
class CompressContextRequest(BaseModel):
    question: str
    context: str


class CompressContextResponse(BaseModel):
    compressed_context: str


def chunk_text(text: str, max_words: int = 350) -> List[str]:
    """Splits the text into chunks of maximum words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def get_db_connection():
    conn = sqlite3.connect(f'{CHROMA_DB_PATH}/chroma.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn


def transform_where_clause(where: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует операторы $contains в $in рекурсивно, обрабатывая $and и $or.
    Удаляет "impossible_value" и упрощает $and, если это необходимо.
    Если в where больше 1 параметра, оборачивает их в $and.
    """

    def transform_recursive(condition: Any) -> Any:
        if isinstance(condition, dict):
            new_condition = {}
            for key, value in condition.items():
                if isinstance(value, dict) and "$contains" in value:
                    # Обрабатываем $contains
                    search_string = value["$contains"]
                    transformed_value = process_contains(key, search_string)
                    if transformed_value != {"$eq": "impossible_value"}:
                        new_condition[key] = transformed_value
                elif key in ("$and", "$or"):
                    # Рекурсивно обрабатываем $and и $or
                    transformed_list = [transform_recursive(item) for item in value]
                    # Фильтруем "impossible_value" из списков $and и $or
                    filtered_list = [item for item in transformed_list if item]  # Пустые словари считаются False
                    if filtered_list:
                        new_condition[key] = filtered_list
                else:
                    # Рекурсивно обрабатываем другие ключи
                    new_condition[key] = transform_recursive(value)

            # Упрощаем $and, если в нем остался только один элемент
            if "$and" in new_condition and len(new_condition["$and"]) == 1:
                return new_condition["$and"][0]
            elif not new_condition:
                # Если словарь пустой, возвращаем None, чтобы его можно было отфильтровать
                return None
            else:
                return new_condition
        else:
            # Возвращаем значение как есть, если это не словарь
            return condition

    def process_contains(key: str, search_string: str) -> Dict[str, Any]:
        """
        Выполняет поиск в базе данных и возвращает условие $in.
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
        cursor.execute(query, (key, '%' + search_string + '%', '%' + search_string + '%', '%' + search_string + '%',
                               '%' + search_string + '%'))
        values = [row['value'] for row in cursor.fetchall() if row['value'] is not None]

        conn.close()

        if values:
            return {"$in": values}
        else:
            # Если ничего не найдено, возвращаем невозможное условие
            return {"$eq": "impossible_value"}

    if len(where) > 1 and "$and" not in where and "$or" not in where:
        # Оборачиваем условия в $and
        transformed_where = transform_recursive({'$and': [ {k: v} for k, v in where.items()]})
    else:
        transformed_where = transform_recursive(where)

    # Если верхний уровень стал None, возвращаем пустой словарь
    return transformed_where if transformed_where else {}


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


async def filter_context_with_summarization(question: str, context: str) -> str:
    """
    Фильтрует контекст, генерируя краткое резюме, отвечающее на вопрос.
    """
    # Скомбинируйте вопрос и контекст для подачи в модель
    input_text = f"###leave contact information, phone, email if available.\n Question: {question} Context: {context}"

    # Calculate max_length dynamically
    input_length = len(input_text.split())  # Approximate word count
    max_length = min(int(input_length * 0.5), 2000)  # Set max_length to half the input length, but no more than 2000
    min_length = min(30, int(input_length * 0.1))  # Ensure min_length is not greater than 30 or 10% of input length

    # Важно: использовать модель асинхронно, если это возможно.  В данном случае pipeline не асинхронный,
    # поэтому мы используем asyncio.to_thread, чтобы запустить его в отдельном потоке.
    summary = await asyncio.to_thread(
        summarizer, input_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    return summary[0]['summary_text']


async def parallel_summarize(question: str, contexts: List[str]) -> List[str]:
    """
    Параллельно фильтрует список контекстов, используя asyncio.gather.
    """
    tasks = [filter_context_with_summarization(question, context) for context in contexts]
    filtered_results = await asyncio.gather(*tasks)
    return filtered_results


@app.post("/query", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def query_documents(query: Query):
    """Выполняет поиск дакументов в векторном хранилище и фильтрует контекст локальной LLM."""
    all_results = []
    for label in query.labels:  # Итерируемся по списку labels
        try:
            collection = get_collection(label.value)  # Получаем коллекцию для label

            # Transform the where clause
            transformed_where = transform_where_clause(query.where)

            results = collection.query(
                query_texts=[query.text],
                n_results=query.n_results,
                where=transformed_where,
                include=["documents", "metadatas", "distances"]
            )

            extracted_results = []
            for i in range(len(results['ids'][0])):
                extracted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'label': label  # Добавляем label
                })
            all_results.extend(extracted_results)

        except Exception as e:
            logging.error(f"Ошибка при запросе label {label}: {e}")  # Логируем ошибку, но продолжаем
            traceback.print_exc()

    # Сортируем результаты по расстоянию (по возрастанию)
    all_results.sort(key=lambda x: x['distance'])

    # Group chunks by source_document_id and combine them
    grouped_results = {}
    for item in all_results:
        if item['distance'] < 10: # Фильтруем по максимальному расстоянию
            source_document_id = item['metadata']['source_document_id']
            if source_document_id not in grouped_results:
                grouped_results[source_document_id] = {
                    'document': '',
                    'metadata': item['metadata'],
                    'label': item['label'],
                    'chunks': []  # List to store chunks in order
                }
            grouped_results[source_document_id]['chunks'].append(item)

    # Sort chunks within each group by their ID postfix (e.g., _0, _1, _2)
    for source_document_id, group_data in grouped_results.items():
        group_data['chunks'].sort(key=lambda x: int(x['id'].split('_')[-1]))  # Extract and sort by the number after the last underscore

    # Combine the sorted chunks into a single document
    combined_documents = []
    for source_document_id, group_data in grouped_results.items():
        combined_document = ''
        for chunk in group_data['chunks']:
            combined_document += chunk['document'] + '\n'
        combined_documents.append(combined_document)


    if query.summarize:
        # Параллельная фильтрация контекста
        combined_documents = await parallel_summarize(query.text, combined_documents)

    # Возвращаем только отфильтрованный контекст
    return ContextResponse(results=combined_documents)


@app.post("/summarize_context", response_model=CompressContextResponse, dependencies=[Depends(verify_token)])
async def summarize_context_endpoint(request: CompressContextRequest):
    """Compresses the given context based on the question using a summarization model.
    Splits the context into chunks suitable for the summarization model.
    """
    compress_context_done = await filter_context_with_summarization(request.question, request.context)
    return await CompressContextResponse(compressed_context=compress_context_done)


@app.delete("/delete_document/{document_id}/{label}", dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, label: str):
    """Удаляет дакумент или фрагмент дакумента из векторного хранилища по ID и label."""
    try:
        collection = get_collection(label)  # Получаем коллекцию для label
        # !ВАЖНО: Удаляем все фрагменты, связанные с исходным document_id
        collection.delete(where={"source_document_id": document_id})
        return {"message": f"Дакумент с ID {document_id} и все его фрагменты с label {label} успешно удалены"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/force_save/", response_model=ForceSaveResponse, dependencies=[Depends(verify_token)])
async def force_save_messages():
    """
    Endpoint to force save the telegram messages to ChromaDB.
    """
    message = await telegram_integration.save_telegram_messages_to_chromadb()
    return ForceSaveResponse(message=message)


# --- Signal Handling and atexit ---
async def handle_exit():
    logging.info("Завершение работы: сохранение данных...")
    await telegram_integration.sync_save_all_telegram_messages_to_files()
    logging.info("Данные сохранены.")


# Register the exit handler
# atexit.register(handle_exit)

def register_signal_handlers():
    """Registers signal handlers for SIGINT and SIGTERM."""

    async def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}. Exiting...")
        try:
            await telegram_integration.sync_save_all_telegram_messages_to_files()  # Save data before exiting
            await telegram_integration.save_telegram_messages_to_chromadb()
            logging.info("Data saved successfully.");
        except Exception as e:
            logging.error(f"Error saving data during shutdown: {e}")
            traceback.print_exc()
        finally:
            sys.exit(0)

    def handle_signal(sig, frame):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(signal_handler(sig, frame))

    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Termination signal (e.g., from Docker)


# --- Main ---
if __name__ == "__main__":
    register_signal_handlers()

    uvicorn.run(app, host="0.0.0.0", port=8001)