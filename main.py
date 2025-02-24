# --- START OF FILE main.py ---
# --- Дополнительные библиотеки ---
import asyncio
import io
import json
import logging
import signal
import sqlite3
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any, Tuple, List

from pydantic import BaseModel

from chromadb_utils import client
import uvicorn
from chromadb import PersistentClient
from fastapi import FastAPI, HTTPException, Header, status, UploadFile, File, Query as FastAPIQuery, Request, Body, Depends

from chromadb_utils import get_collection, upsert_to_collection
from config import config  # Убедитесь, что config.py содержит CHAT_TOKEN и TELEGRAM_BOT_TOKEN
from file_utils import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx, \
    extract_text_from_json, extract_text_from_xlsx
from models import ValidLabels, DocumentBase, Query, ContextResponse, ForceSaveResponse
# Import the new module
from telegram_integration import telegram_integration
from text_utils import split_text_semantically

# --- Импорт Transformers и Pipeline ---
from transformers import pipeline, AutoTokenizer

# --- Конфигурация ---
CHROMA_DB_PATH = "chroma_db"  # Папка, где будет храниться база данных ChromaDB
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

@app.post("/add_document/", dependencies=[Depends(verify_token)])
async def add_document(request_data: AddDocumentRequest = Body(...)):
    """
        Добавляет текст как документ в векторное хранилище, принимая JSON payload.

        - **text**: Текст документа для добавления.
        - **label**: Указание источника документа (hubspot, telegram, wiki, startrek). Обязательное поле.
        - **document_id**: Уникальный идентификатор документа. Если не указан, генерируется автоматически.
        - **metadata**: Словарь с метаданными документа.  Дополнительные поля будут сохранены.
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

        if request_data.chunk:  # По умолчанию разбиваем на чанки, если не указано "chunk": false
            # Разбиваем на чанки
            chunks = await split_text_semantically(text)
            chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]

            metadatas = []
            for i in range(len(chunks)):
                chunk_metadata = combined_metadata
                chunk_metadata["source_document_id"] = document_id
                metadatas.append(chunk_metadata)
            await upsert_to_collection(collection, chunks, metadatas, chunk_ids)
        else:
            # Сохраняем файл целиком
            combined_metadata["source_document_id"] = document_id  # Добавляем source_document_id
            await upsert_to_collection(collection, [text], [combined_metadata], [document_id])

        return {"message": f"Документ успешно обработан и добавлен с label {label} и ID {document_id}."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def process_results(results: List[Dict[str, Any]], label: ValidLabels, query: Query) -> List[Dict[str, Any]]:
    """
    Обрабатывает результаты поиска, заменяя чанки целым документом, если доля чанков превышает определенный порог.
    """
    source_document_ids = {}
    for result in results:
        source_document_id = result['metadata'].get('source_document_id')
        if source_document_id:
            source_document_ids[source_document_id] = source_document_ids.get(source_document_id, 0) + 1

    for source_document_id, count in source_document_ids.items():
        if count > 5:
            # Remove the chunks from results
            results = [result for result in results if result['metadata'].get('source_document_id') != source_document_id]

            # Fetch the entire document by source_document_id
            collection = get_collection(label.value)
            full_document_results = collection.get(
                where={"source_document_id": source_document_id},
                include=["documents", "metadatas"]
            )

            if full_document_results and full_document_results['documents']:
                # Create result entries for each document found
                for i in range(len(full_document_results['ids'])):
                    results.append({
                        'id': full_document_results['ids'][i],
                        'document': full_document_results['documents'][i],
                        'metadata': full_document_results['metadatas'][i],
                        'distance': None,  # Distance not applicable for full documents
                        'label': label
                    })
            break  # Only process one document at a time

    return results


# # --- Инициализация pipeline для фильтрации контекста (ВЫНЕСТИ ЗА ПРЕДЕЛЫ ФУНКЦИИ, ЧТОБЫ НЕ ПЕРЕИНИЦИАЛИЗИРОВАТЬ КАЖДЫЙ РАЗ) ---
# try:
#     context_filter_pipeline = pipeline("text2text-generation", model="facebook/bart-base", device="cuda:0")
#     print("BART-base loaded successfully with CUDA")
# except Exception as e:
#     print(f"Error loading BART-base with CUDA: {e}")
#     context_filter_pipeline = pipeline("text2text-generation", model="facebook/bart-base", device="cpu")
#     print("BART-base loaded successfully on CPU")

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


def filter_context_with_local_llm(question, context):
    max_context_length = 2048
    if len(context) > max_context_length:
        context = context[:max_context_length]
    prompt = create_prompt(question, context)

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", model_max_length=2048) # Ensure correct tokenizer

    # Tokenize the prompt and check token IDs
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"][0]
    vocab_size = tokenizer.vocab_size
    for token_id in input_ids:
        if token_id >= vocab_size:
            print(f"Error: Token ID {token_id} is out of vocabulary range (0-{vocab_size-1})")
            return ""  # or handle the error in another way

    filtered_context = context_filter_pipeline(prompt, max_length=2048, truncation=True)[0]['generated_text']
    return filtered_context

@app.post("/query", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def query_documents(query: Query):
    """Выполняет поиск документов в векторном хранилище и фильтрует контекст локальной LLM."""
    all_results = []
    for label in query.labels:  # Итерируемся по списку labels
        try:
            collection = get_collection(label.value)  # Получаем коллекцию для label
            results = collection.query(
                query_texts=[query.text],
                n_results=1,
                where=query.where,  # Фильтры по метаданным
                include=["documents", "metadatas", "distances"]  # Добавляем distances в include
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

    # Сортируем результаты по расстоянию (по возрастанию)
    all_results.sort(key=lambda x: x['distance'])

    # Concatenate documents from all results into a single context string
    context = "\n".join([result['document'] for result in all_results])

    # Filter the context with the local LLM
    #filtered_context = filter_context_with_local_llm(query.text, context)

    # gpt_response = await call_expensive_gpt(query.text, filtered_context)

    # Создаем результат с отфильтрованным контекстом
    # filtered_result = {
    #     'id': 'filtered_context',
    #     'document': filtered_context,
    #     'metadata': {},
    #     'distance': 0,
    #     'label': 'filtered'
    # }

    # Возвращаем только отфильтрованный контекст
    return ContextResponse(results=all_results)



@app.delete("/delete_document/{document_id}/{label}", dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, label: str):
    """Удаляет документ или фрагмент документа из векторного хранилища по ID и label."""
    try:
        collection = get_collection(label)  # Получаем коллекцию для label
        # !ВАЖНО: Удаляем все фрагменты, связанные с исходным document_id
        collection.delete(where={"source_document_id": document_id})
        return {"message": f"Документ с ID {document_id} и все его фрагменты с label {label} успешно удалены"}
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


def get_db_connection():
    conn = sqlite3.connect(f'{CHROMA_DB_PATH}/chroma.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn
# Определите функцию для получения уникальных партнеров прямо из базы данных
def find_partner_strings(search_string):
    """
    Ищет вхождения строки в столбце string_value таблицы embedding_metadata,
    где key = 'partner', без учета регистра.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT DISTINCT string_value
    FROM embedding_metadata
    WHERE key = 'partner'
      AND lower(string_value) LIKE lower(?)
    """

    cursor.execute(query, ('%' + search_string + '%',))  # Используем параметризацию для безопасности

    results = cursor.fetchall()
    conn.close()
    return results

@app.get("/unique_partners/{label}", dependencies=[Depends(verify_token)])
async def endpoint_get_unique_partners(label: ValidLabels, prefix: Optional[str] = None):
    """
    Возвращает список уникальных значений поля "partner" из метаданных документов указанной коллекции (label),
    используя прямой SQL-запрос к таблице embedding_metadata.
    Можно указать префикс для фильтрации значений.
    """
    try:
        partners = find_partner_strings(prefix)
        return {"partners": partners}
    except Exception as e:
        #  Обработайте исключения, например, логирование и возврат ошибки
        print(f"Ошибка при выполнении запроса к базе данных: {e}")
        return {"error": "Произошла ошибка при получении партнеров"} # Верните более информативное сообщение об ошибке

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            logging.info("Data saved successfully.")
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