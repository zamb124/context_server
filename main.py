# --- START OF FILE main.py ---
# --- Дополнительные библиотеки ---
import asyncio
import io
import json
import logging
import signal
import sys
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, status, UploadFile, File
from pydantic import BaseModel

from chromadb_utils import get_collection
from config import config  # Убедитесь, что config.py содержит CHAT_TOKEN и TELEGRAM_BOT_TOKEN
from file_utils import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx, \
    extract_text_from_json, extract_text_from_xlsx
# Import the new module
from telegram_integration import telegram_integration
from text_utils import split_text_semantically

# --- Конфигурация ---
CHROMA_DB_PATH = "chroma_db"  # Папка, где будет храниться база данных ChromaDB
TEMP_STORAGE_PATH = "temp_telegram_data"  # Папка для временного хранения данных Telegram
SAVE_INTERVAL_SECONDS = 600  # Интервал сохранения в секундах (10 минут)

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


# --- Модели данных ---
class DocumentBase(BaseModel):
    id: str  # ID документа
    text: str
    label: str  # Обязательный атрибут label
    metadata: Dict[str, str] = {}  # Добавляем метаданные


class Query(BaseModel):
    text: str
    labels: List[str]  # Обязательный атрибут labels
    n_results: int = 5
    where: Optional[Dict[str, Any]] = None  # Фильтры по метаданным


class ContextResponse(BaseModel):
    results: List[Dict]


class ForceSaveResponse(BaseModel):
    message: str


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    """
    # Start telegram integration
    asyncio.create_task(telegram_integration.start()) # Call the method to get a coroutine
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


# --- API endpoints ---
@app.post("/add_document/", dependencies=[Depends(verify_token)])
async def add_document(
        file: UploadFile = File(...),
        label: str = "default",
        metadata: str = "{}",  # Передаем метаданные как строку JSON
        document_id: str = None  # Allow specifying the document_id in the query
):
    """Добавляет документ в векторное хранилище, извлекая текст из файла.

    Args:
        file: Файл для загрузки (поддерживаются форматы: pdf, txt, docx, json, xlsx).
        label: Label для документа.
        metadata: JSON-строка с метаданными.
        document_id:  Уникальный идентификатор документа (если не указан, будет сгенерирован).
    """
    try:
        file_content = await file.read()
        file_io = io.BytesIO(file_content)
        file_type = file.filename.split(".")[-1].lower()

        if file_type == "pdf":
            text = extract_text_from_pdf(file_io)
        elif file_type == "txt":
            text = extract_text_from_txt(file_io)
        elif file_type == "docx":
            text = extract_text_from_docx(file_io)
        elif file_type == "json":
            text = extract_text_from_json(file_io)
        elif file_type == "xlsx":
            text = extract_text_from_xlsx(file_io)
        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

        # Преобразуем метаданные из строки JSON в словарь
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Некорректный формат метаданных JSON")

        if not document_id:
            document_id = file.filename  # Используем имя файла как ID по умолчанию, если не указан

        collection = get_collection(label)

        # Удаляем старые чанки документа, если они есть
        collection.delete(where={"source_document_id": document_id})

        chunks = split_text_semantically(text)
        chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i in range(len(chunks)):
            chunk_metadata = metadata_dict.copy()
            chunk_metadata["source_document_id"] = document_id
            metadatas.append(chunk_metadata)

        collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
        )

        return {"message": f"Документ {file.filename} успешно обработан и добавлен с label {label} и ID {document_id}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def query_documents(query: Query):
    """Выполняет поиск документов в векторном хранилище."""
    all_results = []
    for label in query.labels:  # Итерируемся по списку labels
        try:
            collection = get_collection(label)  # Получаем коллекцию для label
            results = collection.query(
                query_texts=[query.text],
                n_results=query.n_results,
                where=query.where  # Фильтры по метаданным
            )
            """
                $eq – оператор "равно". Другие операторы:
                $ne: Не равно
                $gt: Больше
                $gte: Больше или равно
                $lt: Меньше
                $lte: Меньше или равно
                $in: Значение находится в списке значений (например, {"category": {"$in": ["blog", "news"]}})
                $nin: Значение отсутствует в списке значений
                $contains: Строка содержит подстроку (например, {"author": {"$contains": "Doe"}})
                $like: Строка соответствует шаблону SQL-стиля LIKE.
            """
            # Добавляем результаты в общий список
            for i in range(len(results['ids'][0])):
                all_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,  # Добавляем расстояние
                    'label': label  # Добавляем label
                })
        except Exception as e:
            logging.error(f"Ошибка при запросе label {label}: {e}")  # Логируем ошибку, но продолжаем
            # raise HTTPException(status_code=500, detail=str(e)) # Можно и выкидывать исключение, но лучше обработать все label

    # Возвращаем все результаты, объединенные в один список
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
