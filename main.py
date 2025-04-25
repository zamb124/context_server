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
import fasttext # <--- Добавлено
from fastapi import FastAPI, HTTPException, Header, status, Body, Depends
from starlette.requests import Request
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM # <--- Добавлено
from nltk.corpus import stopwords # <--- Добавлено
from nltk.tokenize import word_tokenize # <--- Добавлено


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
# MODEL_PROCESS_COUNT = 1 # <-- Больше не нужно

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
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =============================================================================
#                          Логика Модели (Перенесено из model_process.py)
# =============================================================================

# --- Загрузка моделей ---
fasttext_model_path = "lid.176.bin"
lang_model = fasttext.load_model(fasttext_model_path) # Глобальная загрузка, если нужна

nltk.data.path.append('nltk')
try:
    stop_words = set(stopwords.words('english')) # Используем английский по умолчанию
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

def clean_text(text, lang='en'):
    """
    Очистка текста от стоп-слов и специальных символов.
    (Оставлено для совместимости, но в SummarizerModel не используется)
    """
    # Определяем язык, если не задан явно (не используется в SummarizerModel, но может пригодиться)
    # predictions = lang_model.predict(text.replace('\n', ' '), k=1)
    # detected_lang = predictions[0][0].split('__label__')[1]
    # lang_map = {'en': 'english', 'ru': 'russian', ...} # Добавить нужные языки
    # nltk_lang = lang_map.get(detected_lang, 'english')
    nltk_lang = 'english' # Пока хардкод

    try:
        current_stop_words = set(stopwords.words(nltk_lang))
    except OSError: # Если язык не поддерживается NLTK
        logger.warning(f"Stopwords for language '{nltk_lang}' not found, using English.")
        current_stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)
    cleaned_tokens = [w for w in word_tokens if w.lower() not in current_stop_words and w.isalnum()]
    return " ".join(cleaned_tokens)


class SummarizerModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        # self.model = None # Не обязательно загружать отдельно для pipeline
        self.pipeline = None

    def load_model(self):
        """Синхронная загрузка модели."""
        try:
            # AutoTokenizer и pipeline должны справиться с загрузкой нужных компонентов для модели суммирования
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            device = 0 if torch.cuda.is_available() else -1
            # Используем правильный pipeline для задачи
            self.pipeline = pipeline("summarization", model=self.model_name, tokenizer=self.tokenizer, device=device) # Передаем загруженный токенизатор
            device_name = "cuda" if device == 0 else "cpu"
            logger.info(f"Модель {self.model_name} (Summarization) успешно загружена на {device_name}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {self.model_name}: {e}")
            traceback.print_exc()
            raise

    # Метод summarize остается без изменений, т.к. он использует self.pipeline
    def summarize(self, text: str, max_length: int = 1024, min_length: int = 30) -> str:
        # ... (код метода summarize) ...
        if not self.pipeline or not self.tokenizer:
             raise ValueError("Pipeline или токенизатор не загружены.")
        try:
            logger.info(f"Начало суммирования текста. Длина текста (символы): {len(text)}")
            if not text:
                logger.warning("Пустой текст для суммирования.")
                return "" # Возвращаем пустую строку для пустого ввода

            # Определяем максимальную длину входа для модели (обычно 1024 для T5, но pipeline может сам обрабатывать)
            # Для mT5 XLSum, фактический лимит может быть ниже, проверим токенизацией
            # Вместо жесткого лимита 512, попробуем положиться на pipeline, но для очень больших текстов все равно нужно чанкование
            # Давайте оставим логику чанкования, но с лимитом токенов токенизатора
            max_tokens_per_chunk = self.tokenizer.model_max_length # Обычно 512 или 1024

            # Токенизация для проверки длины (без паддинга и усечения здесь)
            inputs = self.tokenizer(text, return_tensors=None, truncation=False) # Не тензоры, просто токены
            num_tokens = len(inputs['input_ids'])
            logger.info(f"Количество токенов в тексте: {num_tokens}")

            # Если текст не превышает лимит, суммаризируем его напрямую
            if num_tokens <= max_tokens_per_chunk:
                 # Используем pipeline напрямую
                summary_result = self.pipeline(text, max_length=max_length, min_length=min_length, truncation=True) # Добавим truncation=True
                summary = summary_result[0]['summary_text']
                logger.info(f"Суммирование завершено (один чанк). Длина summary (символы): {len(summary)}")
                return summary

            # Если текст длиннее, разбиваем его на чанки (простая стратегия по токенам)
            # Важно: нужно разбивать так, чтобы не резать слова/токены пополам.
            # Используем encode/decode для корректного разбиения
            encoded_ids = inputs['input_ids']
            chunks_texts = []
            for i in range(0, num_tokens, max_tokens_per_chunk):
                chunk_ids = encoded_ids[i:i + max_tokens_per_chunk]
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True) # Декодируем чанк обратно в текст
                chunks_texts.append(chunk_text)

            logger.info(f"Текст разбит на {len(chunks_texts)} чанков по ~{max_tokens_per_chunk} токенов")

            # Суммаризация каждого чанка
            summaries = []
            for idx, chunk in enumerate(chunks_texts):
                if not chunk.strip(): # Пропускаем пустые чанки, если вдруг получились
                    continue
                # Используем pipeline для каждого чанка
                chunk_summary_result = self.pipeline(chunk, max_length=max_length, min_length=min_length, truncation=True)
                chunk_summary = chunk_summary_result[0]['summary_text']
                summaries.append(chunk_summary)
                logger.info(f"Чанк {idx + 1}/{len(chunks_texts)} суммаризован. Длина summary: {len(chunk_summary)}")

            # Объединяем все суммаризации в один итоговый текст
            final_summary = " ".join(summaries)
            logger.info(f"Итоговое суммирование завершено. Длина итогового summary: {len(final_summary)}")
            return final_summary

        except Exception as e:
            logger.error(f"Ошибка при суммировании текста: {e}")
            traceback.print_exc()
            # Не пробрасываем ошибку выше, чтобы API не упало, а возвращаем пустую строку или сообщение об ошибке
            return f"[Ошибка суммирования: {str(e)}]"


# =============================================================================
#                             Lifespan Manager
# =============================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Lifespan-обработчик приложения:
    - Загружает модель SummarizerModel при старте.
    - Делает модель доступной через app.state.
    - (Опционально) Освобождает ресурсы при завершении.
    """
    logger.info("Запуск lifespan - Загрузка модели...")
    _app.state.summarizer_model = SummarizerModel(model_name=MODEL_NAME)
    try:
        # Загрузка модели может быть долгой, но делаем это синхронно при старте
        # Если load_model асинхронный, использовать await
        # await _app.state.summarizer_model.load_model()
        _app.state.summarizer_model.load_model() # Синхронный вызов
        logger.info("Модель успешно загружена и готова к работе.")
    except Exception as e:
        logger.error(f"Критическая ошибка: Не удалось загрузить модель при старте: {e}")
        # Можно решить, останавливать ли приложение или работать без модели
        _app.state.summarizer_model = None # Указываем, что модель недоступна
        # raise # Если модель критична, можно остановить старт

    yield # Приложение работает здесь

    # Код очистки при завершении (если нужен)
    logger.info("Завершение lifespan - Очистка ресурсов...")
    _app.state.summarizer_model = None
    # Возможно, нужно очистить память GPU, если используется
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Ресурсы очищены.")

# Создаем приложение FastAPI с новым lifespan
app = FastAPI(lifespan=lifespan)
app.logger = logger # Используем наш настроенный логгер


# =============================================================================
#                             Зависимости
# =============================================================================
async def verify_token(authorization: Optional[str] = Header(None)):
    # ... (код верификации токена остается без изменений)
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
    # ... (код добавления документа остается без изменений)
    try:
        text = request_data.text
        label = request_data.label
        document_id = request_data.document_id
        metadata = request_data.metadata

        document_id = document_id or f"doc_{label.value}_{hash(text)}" # Используем label.value

        collection = get_collection(label.value)
        collection.delete(where={"source_document_id": document_id})

        # Проверяем, есть ли кастомные метаданные перед слиянием
        base_metadata = {}
        if metadata:
             # Убедимся, что metadata - это словарь
            if isinstance(metadata, dict):
                base_metadata = metadata
            else:
                try:
                    # Пытаемся распарсить, если это строка JSON или использовать model_dump, если Pydantic модель
                    if isinstance(metadata, str):
                         base_metadata = json.loads(metadata)
                    elif hasattr(metadata, 'model_dump'):
                         base_metadata = json.loads(metadata.model_dump_json())
                    else:
                         logger.warning(f"Не удалось обработать metadata типа {type(metadata)}")
                         base_metadata = {}
                except (json.JSONDecodeError, TypeError) as e:
                     logger.warning(f"Ошибка парсинга metadata: {e}. Исходные данные: {metadata}")
                     base_metadata = {}


        if request_data.chunk:
            chunks = await split_text_semantically(text)
            meta_end_text_parts = [f'{k}: {v}' for k, v in base_metadata.items() if v is not None]
            meta_end_text = ""
            if meta_end_text_parts:
                 meta_end_text = '\n[META: ' + ', '.join(meta_end_text_parts) + ']'

            chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            metadatas = []
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                processed_chunks.append(chunk_text + meta_end_text)
                chunk_metadata = base_metadata.copy() # Копируем базовые метаданные
                chunk_metadata["source_document_id"] = document_id
                chunk_metadata["chunk_index"] = i # Добавляем индекс чанка
                metadatas.append(chunk_metadata)

            if processed_chunks: # Убедимся, что есть что добавлять
                 await upsert_to_collection(collection, processed_chunks, metadatas, chunk_ids)
            else:
                 logger.warning(f"Нет чанков для добавления для документа {document_id}")

        else:
            # Сохраняем текст целиком
            final_metadata = base_metadata.copy()
            final_metadata["source_document_id"] = document_id
            await upsert_to_collection(collection, [text], [final_metadata], [document_id])

        return {"message": f"Документ успешно обработан и добавлен с label {label.value} и ID {document_id}."}

    except Exception as e:
        logging.error(f"Ошибка при добавлении документа: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Вспомогательные функции для суммирования ---

async def summarize_document(doc: str, question: str, model: SummarizerModel) -> str:
    """
    Выполняет суммирование одного документа с использованием модели.
    Запускает блокирующую операцию в отдельном потоке.
    """
    if not model:
        logger.error("Модель SummarizerModel не загружена или недоступна.")
        return "[Ошибка: Модель не загружена]"
    if not doc:
        return "" # Ничего суммировать

    try:
        # Запускаем синхронный метод model.summarize в отдельном потоке
        summary = await asyncio.to_thread(model.summarize, doc)
        return summary
    except Exception as e:
        logger.error(f"Ошибка во время вызова summarize_document: {e}")
        traceback.print_exc()
        return f"[Ошибка суммирования: {str(e)}]"


async def summarize_documents(docs: list, question: str, model: SummarizerModel) -> list:
    """
    Выполняет суммирование списка документов параллельно.
    """
    if not model:
        logger.error("Модель SummarizerModel не загружена, суммирование невозможно.")
        return ["[Ошибка: Модель не загружена]" for _ in docs]

    start_time = time.time()
    input_chars = sum(len(doc) for doc in docs if doc) # Проверяем на None или пустые строки

    # Создаем задачи для каждого документа
    tasks = [summarize_document(doc, question, model) for doc in docs if doc]
    results = await asyncio.gather(*tasks)

    # Восстанавливаем исходный размер списка, если были пустые документы
    final_results = []
    result_idx = 0
    for doc in docs:
        if doc:
            final_results.append(results[result_idx])
            result_idx += 1
        else:
            final_results.append("") # Возвращаем пустую строку для пустых входов

    output_chars = sum(len(result) for result in final_results)
    compression_ratio = (input_chars - output_chars) / input_chars * 100 if input_chars > 0 else 0

    elapsed_time = time.time() - start_time
    logger.info(
        f"Суммаризация {len(docs)} документов завершена за {elapsed_time:.2f} сек.\n"
        f"Символов до обработки: {input_chars:,}\n"
        f"Символов после обработки: {output_chars:,}\n"
        f"Степень сжатия: {compression_ratio:.1f}%"
    )

    return final_results

# --- Обновленные эндпоинты ---

@app.post("/query", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def query_documents(query: Query, request: Request):
    """
    Выполняет поиск документов в векторном хранилище и фильтрует контекст по запросу.
    Теперь использует модель из app.state для суммирования.
    """
    # ... (логика поиска в ChromaDB остается без изменений) ...
    all_results = []
    for label in query.labels:
        try:
            collection = get_collection(label.value)
            transformed_where = transform_where_clause(query.where)

            results = collection.query(
                query_texts=[query.text],
                n_results=query.n_results,
                where=transformed_where,
                include=["documents", "metadatas", "distances"]
            )

            # Проверка на пустые результаты
            if not results or not results.get('ids') or not results['ids'][0]:
                 logger.info(f"Для label '{label.value}' и запроса '{query.text}' ничего не найдено.")
                 continue # Переходим к следующему лейблу

            extracted_results = []
            for i in range(len(results['ids'][0])):
                extracted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'label': label.value # Сохраняем как строку
                })
            all_results.extend(extracted_results)

        except Exception as e:
            logging.error(f"Ошибка при запросе для label {label.value}: {e}") # Используем label.value
            traceback.print_exc()
            # Не прерываем весь запрос, если ошибка с одним лейблом
            # raise HTTPException(status_code=500, detail=f"Ошибка при запросе для label {label.value}: {e}")

    if not all_results:
        return ContextResponse(results=[]) # Возвращаем пустой список, если ничего не найдено

    # Сортируем результаты по расстоянию
    all_results.sort(key=lambda x: x['distance'])

    # Группируем чанки по source_document_id (если данные есть)
    grouped_results = {}
    min_distance_map = {} # Для отслеживания минимального расстояния для каждого документа
    for item in all_results:
        # Фильтрация по максимальному расстоянию (настраиваемый параметр?)
        distance_threshold = 1.5 # Пример порога, можно сделать параметром
        if item['distance'] <= distance_threshold:
            metadata = item.get('metadata') or {}
            source_document_id = metadata.get('source_document_id')
            if not source_document_id:
                # Если нет source_document_id, считаем сам документ уникальным
                source_document_id = item['id']
                # Добавляем документ как "один чанк"
                if source_document_id not in grouped_results:
                    grouped_results[source_document_id] = {
                         'document': item['document'], # Полный документ
                         'metadata': metadata,
                         'label': item['label'],
                         'chunks': [item] # Сам документ как единственный чанк
                    }
                    min_distance_map[source_document_id] = item['distance']
            else:
                # Стандартная логика группировки чанков
                if source_document_id not in grouped_results:
                    grouped_results[source_document_id] = {
                        'document': '', # Будет собран позже
                        'metadata': metadata, # Метаданные первого встреченного чанка (может быть неточно)
                        'label': item['label'],
                        'chunks': []
                    }
                    min_distance_map[source_document_id] = item['distance']

                grouped_results[source_document_id]['chunks'].append(item)
                # Обновляем минимальное расстояние для документа
                min_distance_map[source_document_id] = min(min_distance_map[source_document_id], item['distance'])

    # Сортируем чанки внутри каждой группы по chunk_index, если есть, иначе по id
    for source_document_id, group_data in grouped_results.items():
         if group_data['chunks']: # Проверка, что список чанков не пуст
            first_chunk_meta = group_data['chunks'][0].get('metadata', {})
            if 'chunk_index' in first_chunk_meta:
                 group_data['chunks'].sort(key=lambda x: int(x.get('metadata', {}).get('chunk_index', 9999)))
            else:
                 # Пытаемся сортировать по последней части ID, если это число
                 group_data['chunks'].sort(key=lambda x: int(x['id'].split('_')[-1]) if x['id'].split('_')[-1].isdigit() else 9999)

    # Объединяем отсортированные чанки в один документ для каждой группы
    combined_documents_data = []
    for source_document_id, group_data in grouped_results.items():
         # Собираем полный текст из чанков
         # Убираем добавленный '[META: ...]' текст перед объединением, если он есть
         full_doc_text = '\n'.join([chunk['document'].split('\n[META:')[0].strip() for chunk in group_data['chunks']])
         combined_documents_data.append({
             "text": full_doc_text,
             "distance": min_distance_map.get(source_document_id, float('inf')) # Добавляем минимальное расстояние
         })

    # Сортируем объединенные документы по минимальному расстоянию
    combined_documents_data.sort(key=lambda x: x['distance'])

    # Оставляем только тексты для суммирования
    final_docs_to_process = [item["text"] for item in combined_documents_data]

    # Опциональное суммирование
    if query.summarize:
        # Получаем модель из состояния приложения
        summarizer_model = request.app.state.summarizer_model
        if summarizer_model:
            final_docs_to_process = await summarize_documents(final_docs_to_process, query.text, summarizer_model)
        else:
            logger.warning("Суммирование запрошено, но модель не загружена.")
            # Можно вернуть оригинальные документы или добавить сообщение об ошибке
            final_docs_to_process = [f"[Суммирование недоступно] {doc}" for doc in final_docs_to_process]


    return ContextResponse(results=final_docs_to_process)


@app.post("/summarize_context", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def summarize_context_endpoint(query: CompressContextRequest, request: Request):
    """
    Сжимает заданный контекст (summarization) на основе вопроса.
    Использует модель из app.state.
    """
    summarizer_model = request.app.state.summarizer_model
    if not summarizer_model:
         logger.error("Запрос на суммирование контекста, но модель не загружена.")
         # Возвращаем ошибку или исходные данные с примечанием
         # return ContextResponse(results=[f"[Суммирование недоступно] {ctx}" for ctx in query.contexts])
         raise HTTPException(status_code=503, detail="Сервис суммирования временно недоступен (модель не загружена).")

    combined_documents = await summarize_documents(query.contexts, query.question, summarizer_model)
    return ContextResponse(results=combined_documents)


@app.delete("/delete_document/{document_id}/{label}", dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, label: str):
    # ... (код удаления остается без изменений) ...
    try:
        collection = get_collection(label)
        # Удаляем все фрагменты, связанные с данным document_id
        collection.delete(where={"source_document_id": document_id})
         # Попытаемся удалить и сам документ, если он был сохранен целиком (без chunking)
        try:
             collection.delete(ids=[document_id])
        except Exception:
             # Ошибка, если ID не найден - это нормально, если документ был разбит на чанки
             pass
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
    model_status = "ok" if app.state.summarizer_model else "model_not_loaded"
    return {"status": "ok", "model_status": model_status}


@app.post("/force_save/", response_model=ForceSaveResponse, dependencies=[Depends(verify_token)])
async def force_save_messages():
    # ... (код форс-сейва остается без изменений) ...
    message = await telegram_integration.save_telegram_messages_to_chromadb()
    return ForceSaveResponse(message=message)

# =============================================================================
#                      Интеграция с HubSpot (Без изменений)
# =============================================================================
# ... (весь код HubSpot остается здесь) ...
from hubspot_data_extractor import HubSpotDataExtractor # Убедитесь, что импорт есть

# Заглушка токена, если не используется реально
# hubspot_extractor = HubSpotDataExtractor(access_token="YOUR_ACCESS_TOKEN_PLACEHOLDER")

# @app.post("/hubspot")
# async def handle_hubspot_webhook(request: Request):
#     # ... ваш код обработки вебхука ...
#     pass

# def get_associated_company_id(extractor: HubSpotDataExtractor, object_id: int, object_type: str = "contact") -> Optional[int]: # Используем Optional
#     # ... ваш код ...
#     return None

# def update_company_data(extractor: HubSpotDataExtractor, company_id: int):
#     # ... ваш код ...
#     pass

# def rag_store(company_data: dict):
#     # ... ваш код ...
#     pass


# =============================================================================
#                Удалены функции для работы с моделью в отдельном процессе
# =============================================================================
# async def handle_model_requests(...) - Удалено
# async def monitor_stdout(...) - Удалено
# async def send_model_request(...) - Удалено
# def choose_process(...) - Удалено
# async def start_model_process(...) - Удалено
# async def model_process_monitor(...) - Удалено


# =============================================================================
#                  Обработка сигналов завершения приложения (Без изменений)
# =============================================================================
# ... (код handle_exit и register_signal_handlers остается здесь) ...
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
    loop = asyncio.get_event_loop()

    async def shutdown(sig):
        logging.info(f"Получен сигнал {sig.name}. Завершение работы...")
        await handle_exit()
        # Отменяем все выполняющиеся задачи (кроме текущей)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        logging.info(f"Ожидание завершения отмененных задач...")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(shutdown(sig)))


# =============================================================================
#                            Основной блок
# =============================================================================
if __name__ == "__main__":
    # Устанавливаем CPU по умолчанию, если нет GPU или для упрощения
    if not torch.cuda.is_available():
         torch.set_default_device('cpu')
         logger.info("CUDA недоступна, используется CPU.")
    else:
         logger.info(f"CUDA доступна. Используется: {torch.cuda.get_device_name(0)}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Часто рекомендуется 'false' при работе в одном процессе с API
    # Загрузка данных NLTK (уже есть выше, но дублирование не повредит)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/bcp47') # Для языков
    except LookupError:
        nltk.download('bcp47', quiet=True)


    # Регистрация обработчиков сигналов теперь не нужна в main, т.к. uvicorn сам их ловит
    # register_signal_handlers() # Убираем, если uvicorn запускается программно

    # Запуск uvicorn
    # workers=1 рекомендуется, если модель тяжелая и занимает много RAM/VRAM,
    # чтобы каждый воркер не грузил свою копию. Для CPU можно пробовать > 1.
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1) # <-- Установил workers=1