import asyncio
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import traceback
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Dict, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor
import time

import fasttext
import nltk
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header, status, Body, Depends
from pydantic import BaseModel
from starlette.requests import Request
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from chromadb_utils import get_collection, upsert_to_collection
from config import config
from models import ValidLabels, DocumentBase, Query, ContextResponse, ForceSaveResponse
from telegram_integration import telegram_integration
from text_utils import split_text_semantically, split_text_semantically_sync

# --- Конфигурация ---
CHROMA_DB_PATH = "chroma_db"
TEMP_STORAGE_PATH = "temp_telegram_data"
SAVE_INTERVAL_SECONDS = 600

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

# --- Загрузка моделей ---
fasttext_model_path = "lid.176.bin"
lang_model = fasttext.load_model(fasttext_model_path)

nltk.data.path.append('nltk')
try:
    stop_words = set(stopwords.words('english'))  # Или 'english'
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  # Или 'english'

# --- FastAPI App ---
app = FastAPI()

# --- Утилиты ---
def get_db_connection():
    """
    Получает соединение с базой данных SQLite.
    """
    conn = sqlite3.connect(f'{CHROMA_DB_PATH}/chroma.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn


def clean_text(text, lang='en'):
    """
    Очистка текста от стоп-слов и специальных символов.

    Args:
        text: Текст для очистки.
        lang: Язык текста (по умолчанию 'en').

    Returns:
        Очищенный текст.
    """
    stop_words = set(stopwords.words('english'))  # Или 'english', в зависимости от языка
    word_tokens = word_tokenize(text)
    cleaned_tokens = [w for w in word_tokens if w.lower() not in stop_words and w.isalnum()]
    return " ".join(cleaned_tokens)


def _process_contains(key: str, search_string: str) -> Dict[str, Any]:
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


def _transform_recursive(condition: Any) -> Any:
    """
    Рекурсивно преобразует условие, заменяя $contains на $in и обрабатывая $and и $or.
    """
    if isinstance(condition, dict):
        new_condition = {}
        for key, value in condition.items():
            if isinstance(value, dict) and "$contains" in value:
                # Обрабатываем $contains
                search_string = value["$contains"]
                transformed_value = _process_contains(key, search_string)
                if transformed_value != {"$eq": "impossible_value"}:
                    new_condition[key] = transformed_value
            elif key in ("$and", "$or"):
                # Рекурсивно обрабатываем $and и $or
                transformed_list = [_transform_recursive(item) for item in value]
                # Фильтруем "impossible_value" из списков $and и $or
                filtered_list = [item for item in transformed_list if item]  # Пустые словари считаются False
                if filtered_list:
                    new_condition[key] = filtered_list
            else:
                # Рекурсивно обрабатываем другие ключи
                new_condition[key] = _transform_recursive(value)

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


def transform_where_clause(where: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует операторы $contains в $in рекурсивно, обрабатывая $and и $or.
    Удаляет "impossible_value" и упрощает $and, если это необходимо.
    Если в where больше 1 параметра, оборачивает их в $and.
    """
    if len(where) > 1 and "$and" not in where and "$or" not in where:
        # Оборачиваем условия в $and
        transformed_where = _transform_recursive({'$and': [{k: v} for k, v in where.items()]})
    else:
        transformed_where = _transform_recursive(where)

    # Если верхний уровень стал None, возвращаем пустой словарь
    return transformed_where if transformed_where else {}


async def filter_context_with_summarization(question: str, context: str, summarizer) -> str:
    """
    Фильтрует контекст, генерируя краткое резюме, отвечающее на вопрос,
    и удаляет слова из вопроса из ответа. Использует чанкинг для обработки больших контекстов.
    """
    start_time = perf_counter()  # Засекаем время
    logging.info(f"Начала filter_context_with_summarization. Длина контекста: {len(context)}")

    def detect_language(context: str) -> str:
        """Определение языка текста."""
        cleaned_text = " ".join(context.split())  # Убираем \n и лишние пробелы
        # Предполагается, что lang_model определен где-то в глобальной области видимости
        # и имеет метод predict.
        try:
            lang = lang_model.predict([cleaned_text])[0][0][0].replace('__label__', '')
        except Exception as e:
            logging.error(f"Error predicting language: {e}")
            return 'en'  # Default to English if language detection fails
        logging.info(f"Язык определен: {lang}")
        return lang

    # Очистка текста от метаданных и технических деталей (ВАЖНО!)
    context = context.replace("[META:", "").replace("]", "")  # Пример, настройте под свои метаданные

    lang = detect_language(context)
    cleaned_question = clean_text(question, lang=lang)
    prompts = {
        'en': f"""Shorten it a little, if possible, without loss of meaning to help: \n""",
        'ru': "Текст нужно сократить без потери смысла и очистить от технических данных: \n"
    }
    prompt = prompts.get(lang, prompts['en'])

    # Разбиваем контекст на чанки
    chunks = split_text_semantically_sync(context, chunk_size=3000, chunk_overlap=200)
    summarized_chunks = []

    for i, chunk in enumerate(chunks):
        input_text = f"{prompt}{chunk}"
        logging.info(f"Входной текст для summarizer (чанк {i}), длина: {len(input_text)}")

        # Вычисление max_length динамически
        input_length = len(input_text.split())
        max_length = min(int(input_length * 1.2), 1024)  # Увеличиваем max_length
        min_length = int(input_length * 0.7)  # Немного увеличиваем min_length
        logging.info(f"max_length: {max_length}, min_length: {min_length}")

        try:
            summary = summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
            )[0]['generated_text']
        except Exception as e:
            logging.error(f"Ошибка при вызове summarizer: {e}")
            raise

        logging.info(f"Результат summarizer (чанк), длина: {len(summary)}")
        summarized_chunks.append(summary)

    # Объединяем результаты суммирования чанков
    full_summary = " ".join(summarized_chunks)

    # Post-processing: Удаление слов из вопроса из ответа
    summary_tokens = word_tokenize(full_summary)
    question_tokens = word_tokenize(cleaned_question)
    filtered_summary_tokens = [w for w in summary_tokens if w.lower() not in question_tokens]
    filtered_summary = " ".join(filtered_summary_tokens)
    filtered_summary = filtered_summary.replace(prompt[:-3], "")  # Удаляем prompt

    # Extract metadata using regex
    date_match = re.search(r"create_date:\s*([^\n]+)", context)
    title_match = re.search(r"title:\s*([^\n]+)", context)
    type_match = re.search(r"type:\s*([^\n]+)", context)

    date = date_match.group(1).strip() if date_match else "N/A"
    title = title_match.group(1).strip() if title_match else "N/A"
    event_type = type_match.group(1).strip() if type_match else "N/A"

    # Create intro string
    intro = f"Date of the event: {date}, Type of event: {event_type}, Title of event: {title}.\n"

    filtered_summary = intro + filtered_summary

    end_time = perf_counter()
    logging.info(f"Завершение filter_context_with_summarization. Время выполнения: {end_time - start_time:.4f} секунд")
    return filtered_summary


# --- Модели ---
class AddDocumentRequest(BaseModel):
    """
    Модель запроса для добавления документа.
    """
    text: str
    label: ValidLabels
    document_id: Optional[str] = None
    metadata: DocumentBase
    chunk: Optional[bool] = True
    author: Optional[str] = None


class CompressContextRequest(BaseModel):
    question: str = ''
    contexts: list


# --- Зависимости ---
async def verify_token(authorization: Optional[str] = Header(None)):
    """
    Проверяет токен из заголовка Authorization (схема Bearer).
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


# --- Эндпоинты ---
@app.post("/add_document/", dependencies=[Depends(verify_token)])
async def add_document(request_data: AddDocumentRequest = Body(...)):
    """
    Добавляет текст как документ в векторное хранилище, принимая JSON payload.

    - **text**: Текст документа для добавления.
    - **label**: Указание источника документа (hubspot, telegram, wiki, startrek). Обязательное поле.
    - **document_id**: Уникальный идентификатор документа. Если не указан, генерируется автоматически.
    - **metadata**: Словарь с метаданными документа. Дополнительные поля будут сохранены.
    """
    try:
        text = request_data.text
        label = request_data.label
        document_id = request_data.document_id
        metadata = request_data.metadata

        # Generate document_id if not provided
        document_id = document_id or f"doc_{label}_{hash(text)}"

        collection = get_collection(label.value)
        # Удаляем старые чанки документа, если они есть
        collection.delete(where={"source_document_id": document_id})

        combined_metadata = json.loads(metadata.model_dump_json())

        if request_data.chunk:  # По умолчанию разбиваем на чанки, если не указана "chunk": false
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

        return {"message": f"Документ успешно обработан и добавлен с label {label.value} и ID {document_id}."}

    except Exception as e:
        logging.error(f"Ошибка при добавлении документа: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



async def summarize_documents(docs: List[str], question: str, summarizer) -> List[str]:
    """Summarizes a list of documents using the summarizer."""
    start_time = perf_counter()
    total_input_length = sum(len(doc) for doc in docs)
    logging.info(f"Начало summarize_documents. Количество документов: {len(docs)}, Общее количество символов на входе: {total_input_length}")

    summarized_docs = []
    for i, doc in enumerate(docs):
        doc_start_time = perf_counter()
        try:
            logging.info(f"Суммирование документа {i+1}/{len(docs)}. Длина документа: {len(doc)}")
            summarized_text = await filter_context_with_summarization(question, doc, summarizer)
            summarized_docs.append(summarized_text)
            doc_end_time = perf_counter()
            logging.info(f"Документ {i+1} успешно суммирован за {doc_end_time - doc_start_time:.4f} секунд. Длина результата: {len(summarized_text)}")

        except Exception as e:
            logging.error(f"Ошибка при суммировании документа {i+1}: {e}")
            traceback.print_exc()
            summarized_docs.append("")  # Handle error by returning an empty string

    total_output_length = sum(len(doc) for doc in summarized_docs)
    end_time = perf_counter()
    total_time = end_time - start_time
    logging.info(f"Завершено summarize_documents. Общее время выполнения: {total_time:.4f} секунд, Общее количество символов на выходе: {total_output_length} было: {total_input_length}")
    return summarized_docs

@app.post("/query", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def query_documents(query: Query):
    """
    Выполняет поиск документов в векторном хранилище и фильтрует контекст локальной LLM.
    """
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
            logging.error(f"Ошибка при запросе label {label}: {e}")  # Логируем ошибку
            traceback.print_exc()
            # Возвращаем ошибку клиенту
            raise HTTPException(status_code=500, detail=f"Ошибка при запросе label {label}: {e}")

    # Сортируем результаты по расстоянию (по возрастанию)
    all_results.sort(key=lambda x: x['distance'])

    # Group chunks by source_document_id and combine them
    grouped_results = {}
    for item in all_results:
        if item['distance'] < 10:  # Фильтруем по максимальному расстоянию
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
        group_data['chunks'].sort(
            key=lambda x: int(x['id'].split('_')[-1]))  # Extract and sort by the number after the last underscore

    # Combine the sorted chunks into a single document
    combined_documents = []
    for source_document_id, group_data in grouped_results.items():
        combined_document = ''
        for chunk in group_data['chunks']:
            combined_document += chunk['document'] + '\n'
        combined_documents.append(combined_document)

    if query.summarize:
        # Use the summarization model to compress the combined documents
        summarizer = app.state.summarizer
        combined_documents = await summarize_documents(combined_documents, query.text, summarizer)

    # Возвращаем только отфильтрованный контекст
    return ContextResponse(results=combined_documents)


@app.post("/summarize_context", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def summarize_context_endpoint(query: CompressContextRequest):
    """Compresses the given context based on the question using a summarization model.
    Splits the context into chunks suitable for the summarization model.
    """
    summarizer = app.state.summarizer  # Access the summarizer from app state

    combined_documents = await summarize_documents(query.contexts, query.question, summarizer)
    return ContextResponse(results=combined_documents)


@app.delete("/delete_document/{document_id}/{label}", dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, label: str):
    """Удаляет документ или фрагмент документа из векторного хранилища по ID и label."""
    try:
        collection = get_collection(label)  # Получаем коллекцию для label
        # !ВАЖНО: Удаляем все фрагменты, связанные с исходным document_id
        collection.delete(where={"source_document_id": document_id})
        return {"message": f"Документ с ID {document_id} и все его фрагменты с label {label} успешно удалены"}
    except Exception as e:
        logging.error(f"Ошибка при удалении документа: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}


@app.post("/force_save/", response_model=ForceSaveResponse, dependencies=[Depends(verify_token)])
async def force_save_messages():
    """
    Endpoint to force save the telegram messages to ChromaDB.
    """
    message = await telegram_integration.save_telegram_messages_to_chromadb()
    return ForceSaveResponse(message=message)


@app.post("/hubspot")
async def hubspot_webhook(request: Request):
    """
    Endpoint to receive webhooks from HubSpot.
    Logs the entire payload for inspection.
    """
    logging.info("Received HubSpot webhook payload:")
    try:
        data = await request.json()
    except Exception:
        data = request.json()
    logging.info(json.dumps(data, indent=2))  # Log the payload with pretty printing
    return {"message": "HubSpot webhook received and logged successfully."}


# --- Lifespan ---
async def init_summarizer(model_name: str):
    """Инициализирует summarizer асинхронно."""
    try:
        summarizer = pipeline("text2text-generation", model=model_name, device="cpu")
        logging.info(f"Model initialized: {model_name}")
        return summarizer
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    """
    # Initialize the summarizer
    try:
        app.state.summarizer = await init_summarizer("t5-base")
    except Exception as e:
        logging.error(f"Failed to initialize summarizer: {e}")
        # Handle the error appropriately, possibly exiting the application
        raise

    # Start telegram integration
    #asyncio.create_task(telegram_integration.start()) #TODO: раскоментировать

    yield

    # Stop telegram integration
    #await telegram_integration.stop()

    logging.info("Application shutdown.")


app.router.lifespan_context = lifespan  # Assign lifespan to the app


# --- Signal Handling ---
async def handle_exit():
    """
    Обработчик завершения работы приложения.
    Сохраняет данные перед выходом.
    """
    logging.info("Завершение работы приложения. Сохранение данных...")
    try:
        await telegram_integration.sync_save_all_telegram_messages_to_files()
        logging.info("Данные Telegram сохранены.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных Telegram: {e}")
        traceback.print_exc()

    try:
        await telegram_integration.save_telegram_messages_to_chromadb()
        logging.info("Данные Telegram сохранены в ChromaDB.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных Telegram в ChromaDB: {e}")
        traceback.print_exc()

    logging.info("Данные сохранены.")


def register_signal_handlers():
    """Registers signal handlers for SIGINT and SIGTERM."""

    async def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}. Exiting...")
        await handle_exit()  # Используем общую функцию для сохранения данных
        sys.exit(0)

    def handle_signal(sig, frame):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(signal_handler(sig, frame))

    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Termination signal (e.g., from Docker)


# --- Main ---
if __name__ == "__main__":
    torch.set_default_device('cpu')  # Установить CPU как устройство по умолчанию
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Установить параллелизм для токенизаторов
    nltk.download('bcp47', quiet=True)
    nltk.download('stopwords')
    nltk.download('punkt')
    register_signal_handlers()
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=4)  # Adjust workers as needed