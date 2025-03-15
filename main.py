# main.py

# =======================
# Импорт стандартных библиотек
# =======================
import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

# =======================
# Импорт сторонних библиотек
# =======================
import nltk
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header, status, Body, Depends
from starlette.requests import Request

# =======================
# Импорт локальных модулей
# =======================
from chromadb_utils import get_collection, upsert_to_collection, transform_where_clause
from config import config
from models import Query, ContextResponse, ForceSaveResponse, AddDocumentRequest, \
    CompressContextRequest
from telegram_integration import telegram_integration
from text_utils import split_text_semantically

# =======================
# Конфигурация
# =======================
CHROMA_DB_PATH = "chroma_db"
TEMP_STORAGE_PATH = "temp_telegram_data"
SAVE_INTERVAL_SECONDS = 600
MODEL_PROCESS_COUNT = 1

# =======================
# Токены авторизации
# =======================
CHAT_TOKEN = config.CHAT_TOKEN
if not CHAT_TOKEN:
    raise ValueError("Не установлена переменная окружения CHAT_TOKEN. Пожалуйста, настройте ее в файле .env.")

TELEGRAM_BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не установлена переменная окружения TELEGRAM_BOT_TOKEN. Пожалуйста, настройте ее в файле .env.")

# =======================
# Инициализация логирования
# =======================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# =======================
# Создание приложения FastAPI
# =======================
# Настраиваем логгер
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Создаем обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Настраиваем формат сообщений
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Добавляем обработчик к логгеру
logger.addHandler(console_handler)

# Создаем приложение и добавляем логгер
app = FastAPI()
app.logger = logger


# =============================================================================
#                             Зависимости
# =============================================================================
async def verify_token(authorization: Optional[str] = Header(None)):
    """
    Проверяет токен из заголовка Authorization (Bearer-схема).
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует заголовок Authorization",
            headers={"Authorization": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверная схема аутентификации. Используйте Bearer.",
                headers={"Authorization": "Bearer"},
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный формат заголовка Authorization",
            headers={"Authorization": "Bearer"},
        )

    if token != CHAT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный токен аутентификации",
            headers={"Authorization": "Bearer"},
        )
    return True


# =============================================================================
#                          Эндпоинты API
# =============================================================================
@app.post("/add_document/", dependencies=[Depends(verify_token)])
async def add_document(request_data: AddDocumentRequest = Body(...)):
    """
    Добавляет документ (или его части) в векторное хранилище.
    """
    try:
        text = request_data.text
        label = request_data.label
        document_id = request_data.document_id
        metadata = request_data.metadata

        # Генерируем идентификатор документа, если не передан
        document_id = document_id or f"doc_{label}_{hash(text)}"

        collection = get_collection(label.value)
        # Удаляем предыдущие чанки документа, если они существуют
        collection.delete(where={"source_document_id": document_id})

        combined_metadata = json.loads(metadata.model_dump_json())

        if request_data.chunk:  # По умолчанию текст разбивается на чанки
            # Разбиваем текст на логически завершённые части
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
            # Сохраняем текст целиком
            combined_metadata["source_document_id"] = document_id
            await upsert_to_collection(collection, [text], [combined_metadata], [document_id])

        return {"message": f"Документ успешно обработан и добавлен с label {label.value} и ID {document_id}."}

    except Exception as e:
        logging.error(f"Ошибка при добавлении документа: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def summarize_document(doc: str, question: str, app: FastAPI) -> str:
    """
    Отправляет запрос на суммаризацию текста.
    """
    request_data = {"text": doc, "question": question}
    process_handler = choose_process(app)
    result = await send_model_request(request_data, process_handler)
    if result.get("status") == "success":
        return result.get("summary", "")
    else:
        logging.error(f"Ошибка при суммировании документа: {result.get('error', 'Unknown error')}")
        return ""


async def summarize_documents(docs: list, question: str, app: FastAPI) -> list:
    """
    Выполняет суммаризацию списка документов параллельно.
    """
    start_time = time.time()
    input_chars = sum(len(doc) for doc in docs)

    tasks = [summarize_document(doc, question, app) for doc in docs]
    results = await asyncio.gather(*tasks)

    output_chars = sum(len(result) for result in results)
    compression_ratio = (input_chars - output_chars) / input_chars * 100 if input_chars > 0 else 0

    elapsed_time = time.time() - start_time
    app.logger.info(
        f"Суммаризация завершена за {elapsed_time:.2f} сек.\n"
        f"Символов до обработки: {input_chars:,}\n"
        f"Символов после обработки: {output_chars:,}\n"
        f"Степень сжатия: {compression_ratio:.1f}%"
    )

    return results


@app.post("/query", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def query_documents(query: Query, request: Request):
    """
    Выполняет поиск документов в векторном хранилище и фильтрует контекст по запросу.
    """
    all_results = []
    for label in query.labels:
        try:
            collection = get_collection(label.value)
            # Преобразуем условие поиска
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
                    'label': label
                })
            all_results.extend(extracted_results)

        except Exception as e:
            logging.error(f"Ошибка при запросе для label {label}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Ошибка при запросе для label {label}: {e}")

    # Сортируем результаты по расстоянию (от меньшего к большему)
    all_results.sort(key=lambda x: x['distance'])

    # Группируем чанки по source_document_id
    grouped_results = {}
    for item in all_results:
        if item['distance'] < 10:  # Фильтрация по максимальному расстоянию
            source_document_id = item['metadata']['source_document_id']
            if source_document_id not in grouped_results:
                grouped_results[source_document_id] = {
                    'document': '',
                    'metadata': item['metadata'],
                    'label': item['label'],
                    'chunks': []
                }
            grouped_results[source_document_id]['chunks'].append(item)

    # Сортируем чанки внутри каждой группы по порядковому номеру (из ID)
    for source_document_id, group_data in grouped_results.items():
        group_data['chunks'].sort(key=lambda x: x['id'].split('_')[-1])

    # Объединяем отсортированные чанки в один документ
    combined_documents = []
    for source_document_id, group_data in grouped_results.items():
        combined_document = ''
        for chunk in group_data['chunks']:
            combined_document += chunk['document'] + '\n'
        combined_documents.append(combined_document)

    if query.summarize:
        # Суммаризация объединенных документов
        combined_documents = await summarize_documents(combined_documents, query.text, request.app)

    return ContextResponse(results=combined_documents)


@app.post("/summarize_context", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def summarize_context_endpoint(query: CompressContextRequest, request: Request):
    """
    Сжимает заданный контекст (summarization) на основе вопроса.
    """
    model_queue = request.app.state.model_queue  # Доступ к очереди модели
    combined_documents = await summarize_documents(query.contexts, query.question, model_queue)
    return ContextResponse(results=combined_documents)


@app.delete("/delete_document/{document_id}/{label}", dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, label: str):
    """
    Удаляет документ и его части из векторного хранилища по заданным document_id и label.
    """
    try:
        collection = get_collection(label)
        # Удаляем все фрагменты, связанные с данным document_id
        collection.delete(where={"source_document_id": document_id})
        return {"message": f"Документ с ID {document_id} и все его фрагменты с label {label} успешно удалены"}
    except Exception as e:
        logging.error(f"Ошибка при удалении документа: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Проверка работоспособности сервиса.
    """
    return {"status": "ok"}


@app.post("/force_save/", response_model=ForceSaveResponse, dependencies=[Depends(verify_token)])
async def force_save_messages():
    """
    Принудительное сохранение сообщений Telegram в ChromaDB.
    """
    message = await telegram_integration.save_telegram_messages_to_chromadb()
    return ForceSaveResponse(message=message)


# =============================================================================
#                      Интеграция с HubSpot
# =============================================================================
from hubspot_data_extractor import HubSpotDataExtractor

hubspot_extractor = HubSpotDataExtractor(access_token="YOUR_ACCESS_TOKEN")


@app.post("/hubspot")
async def handle_hubspot_webhook(request: Request):
    """
    Обрабатывает вебхук от HubSpot и обновляет данные компании.
    """
    try:
        payload = await request.json()

        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="Неверный формат payload")

        for event in payload:
            subscription_type = event.get("subscriptionType")
            object_id = event.get("objectId")

            if subscription_type == "contact.creation":
                company_id = get_associated_company_id(hubspot_extractor, object_id)
                if company_id:
                    update_company_data(hubspot_extractor, company_id)

            elif subscription_type == "company.creation":
                update_company_data(hubspot_extractor, object_id)

            elif subscription_type == "deal.creation":
                company_id = get_associated_company_id(hubspot_extractor, object_id, object_type="deal")
                if company_id:
                    update_company_data(hubspot_extractor, company_id)

            elif ".propertyChange" in subscription_type:
                update_company_data(hubspot_extractor, object_id)

        return {"status": "success", "message": "Webhook processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вебхука: {str(e)}")


def get_associated_company_id(extractor: HubSpotDataExtractor, object_id: int, object_type: str = "contact") -> int:
    """
    Получает ID ассоциированной компании через API HubSpot.
    """
    if object_type == "contact":
        associated_data = extractor.get_associated_contacts(object_id)
    elif object_type == "deal":
        associated_data = extractor.get_associated_activities(object_id)
    else:
        associated_data = None

    if associated_data and "companies" in associated_data:
        # Возвращаем ID первой найденной компании
        return associated_data["companies"][0].get("id")
    return None


def update_company_data(extractor: HubSpotDataExtractor, company_id: int):
    """
    Обновляет данные компании с использованием HubSpotDataExtractor.
    """
    company_data = extractor.get_company_data(company_id)
    if company_data:
        rag_store(company_data)


def rag_store(company_data: dict):
    """
    Сохраняет данные компании напрямую в ChromaDB.
    """
    if not company_data:
        raise ValueError("Нет данных для сохранения")

    metadata = {
        "type": "company",
        "category": "sales",
        "partner": company_data.get("name", ""),
        "create_date": company_data.get("createdAt", 0),
        "country": company_data.get("country", ""),
        "chunk": False
    }

    document_request = AddDocumentRequest(
        text=company_data.get("description", ""),
        label="hubspot",
        document_id=str(company_data.get("id")),
        metadata=metadata,
        chunk=False
    )

    add_document(document_request)


# =============================================================================
#                Функции для работы с моделью
# =============================================================================
async def handle_model_requests(process: asyncio.subprocess.Process, queue: asyncio.Queue):
    """
    Обрабатывает запросы к модели:
    - Отправляет запрос в процесс модели через stdin.
    - Ожидает ответа через stdout.
    - Возвращает результат через future.
    """
    try:
        while process.returncode is None:
            request, future = await queue.get()
            try:
                request_str = json.dumps(request) + "\n"
                process.stdin.write(request_str.encode())
                await process.stdin.drain()

                response_line = await process.stdout.readline()
                if not response_line:
                    result = {"status": "error", "error": "No response from process"}
                else:
                    try:
                        result = json.loads(response_line.decode().strip())
                    except Exception as e:
                        logging.error(f"Ошибка при разборе JSON-ответа от процесса {process.pid}: {e}")
                        result = {"status": "error", "error": "Invalid JSON response"}
            except Exception as e:
                logging.error(f"Ошибка при обработке запроса в процессе {process.pid}: {e}")
                result = {"status": "error", "error": str(e)}

            if not future.done():
                future.set_result(result)
            queue.task_done()
    except asyncio.CancelledError:
        logging.info(f"Обработка запросов для процесса {process.pid} отменена.")
    except Exception as e:
        logging.error(f"Ошибка в обработчике запросов для процесса {process.pid}: {e}")


async def send_model_request(request: dict, process_handler: dict) -> dict:
    """
    Отправляет запрос в выбранный процесс модели и возвращает полученный результат.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await process_handler["queue"].put((request, future))
    result = await future
    return result


def choose_process(app: FastAPI) -> dict:
    """
    Выбирает процесс модели для обработки запроса (случайным образом).
    """
    import random
    return random.choice(app.state.model_processes)


async def start_model_process() -> dict:
    """
    Запускает процесс модели:
    - Создаёт процесс и ожидает сигнал "MODEL_READY".
    - Запускает фоновую задачу для обработки запросов.
    Возвращает словарь с процессом и очередью запросов.
    """
    queue = asyncio.Queue()
    process = await asyncio.create_subprocess_exec(
        "python", "/home/viktor-shved/context_server/model_process.py",
        cwd="/home/viktor-shved/context_server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    logging.info(f"Запущен процесс модели с PID: {process.pid}")

    # Ожидаем сигнал готовности от процесса модели
    while True:
        line = await process.stdout.readline()
        if not line:
            logging.error(f"Процесс {process.pid} завершился до получения сигнала готовности.")
            break
        decoded_line = line.decode().strip()
        if decoded_line == "MODEL_READY":
            logging.info(f"Модель в процессе {process.pid} готова к работе.")
            break
        else:
            logging.info(f"Процесс {process.pid} вывод: {decoded_line}")

    asyncio.create_task(handle_model_requests(process, queue))
    return {"process": process, "queue": queue}


async def model_process_monitor(app: FastAPI):
    """
    Мониторит процессы модели и перезапускает их при завершении.
    """
    while True:
        for i, process_handler in enumerate(app.state.model_processes):
            process = process_handler["process"]
            if process.returncode is not None:
                logging.warning(f"Процесс модели {i + 1} (PID {process.pid}) завершился. Перезапуск...")
                new_handler = await start_model_process()
                app.state.model_processes[i] = new_handler
                logging.info(f"Запущен новый процесс модели с PID: {new_handler['process'].pid}")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Lifespan-обработчик приложения:
    - Создает процессы модели и запускает их мониторинг.
    - По завершении приложения корректно завершает процессы.
    """
    _app.state.model_processes = []
    for i in range(MODEL_PROCESS_COUNT):  # MODEL_PROCESS_COUNT = 2
        handler = await start_model_process()
        _app.state.model_processes.append(handler)
        logging.info(f"Процесс модели {i + 1} с PID {handler['process'].pid} добавлен в app.state.model_processes.")

    monitor_task = asyncio.create_task(model_process_monitor(_app))
    yield
    monitor_task.cancel()
    for handler in app.state.model_processes:
        process = handler["process"]
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
    logging.info("Завершение работы приложения.")


app.router.lifespan_context = lifespan  # Назначаем lifespan-обработчик для приложения


# =============================================================================
#                  Обработка сигналов завершения приложения
# =============================================================================
async def handle_exit():
    """
    Обрабатывает завершение работы приложения:
    сохраняет данные перед выходом.
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
    """
    Регистрирует обработчики сигналов SIGINT и SIGTERM.
    """

    async def signal_handler(sig, frame):
        logging.info(f"Получен сигнал {sig}. Завершение работы...")
        await handle_exit()
        sys.exit(0)

    def handle_signal(sig, frame):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(signal_handler(sig, frame))

    signal.signal(signal.SIGINT, handle_signal)  # Обработка Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Обработка сигнала завершения


# Дополнительный endpoint для отладки (проверка работоспособности)
@app.get("/health")
async def health_check_debug():
    """
    Дополнительная проверка работоспособности сервиса (отладка).
    """
    print("Health check called")
    return {"status": "ok"}


# =============================================================================
#                            Основной блок
# =============================================================================
if __name__ == "__main__":
    torch.set_default_device('cpu')  # Устанавливаем использование CPU по умолчанию
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Включаем параллелизм для токенизаторов
    nltk.download('bcp47', quiet=True)
    nltk.download('stopwords')
    nltk.download('punkt')
    register_signal_handlers()
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=4)
