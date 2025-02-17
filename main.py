import json
import multiprocessing
import os
import traceback
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Header, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import config


# --- Загрузка переменных окружения ---

# --- Модели данных ---
class DocumentBase(BaseModel):
    text: str
    label: str  # Метка, к которой относится документ


class Query(BaseModel):
    text: str
    labels: List[str]  # Список меток, в которых нужно искать


class ContextResponse(BaseModel):
    context: Dict[str, List[str]]  # Контекст, сгруппированный по метке


class FileUploadResponse(BaseModel):
    filename: str
    label: str
    unique_name: str  # Уникальное имя документа


class MultipleFileUploadResponse(BaseModel):
    files: List[FileUploadResponse]


# --- Конфигурация ---
FAISS_INDEX_PATH = "y_faiss_index.bin"
DOCUMENT_STORE_PATH = "document_store.json"
INDEX_ID_MAP_PATH = "index_id_map.json"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10
MAX_TEXT_LENGTH = 2048
HUBSPOT_DATA_FOLDER = "model_mini/hubspot_company_data"  # Не используется после рефакторинга. Удалите, если точно не нужен.

# --- Токен авторизации ---
CHAT_TOKEN = config.CHAT_TOKEN
if not CHAT_TOKEN:
    raise ValueError("Не установлена переменная окружения CHAT_TOKEN. Пожалуйста, настройте ее в файле .env.")

# --- Глобальные переменные ---
index = None
document_store = {}  # Структура изменена: label: {unique_name: [chunks]}
index_id_map = {}  # Структура изменена: label: {unique_name_chunk_index : faiss_index_id}


# --- Внедрение зависимостей ---
def get_faiss_index():
    global index
    return index


def get_document_store():
    global document_store
    return document_store


def get_index_id_map():
    global index_id_map
    return index_id_map


# --- Утилиты ---
def preprocess_text(text: str) -> str:
    """Предварительная обработка текста: обрезка, удаление не-ASCII символов."""
    text = text.strip()  # Удаляем пробелы в начале и конце
    return text


def chunk_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
    """Разбивает текст на чанки фиксированной длины."""
    chunks = []
    for i in range(0, len(text), max_length):
        chunks.append(text[i:i + max_length])
    return chunks


def add_to_faiss_index(embedding: np.ndarray, faiss_index: faiss.Index):
    """Добавляет одно вложение в индекс FAISS."""
    faiss.normalize_L2(embedding)
    faiss_index.add(embedding.reshape(1, -1))  # Убедитесь, что это вектор-строка


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, document_store, index_id_map

    # --- Фаза запуска ---
    print("Запуск FastAPI приложения...")
    try:
        # Загрузка метаданных
        try:
            with open(DOCUMENT_STORE_PATH, "r") as f:
                document_store = json.load(f)
                # Рассмотрите возможность преобразования ключей (unique_name_chunk_index) в int, если это необходимо
                if isinstance(document_store, dict):  # Добавлена проверка
                    for label, data in document_store.items():
                        if isinstance(data, dict):
                            for unique_name, chunks in data.items():
                                if not isinstance(chunks, list):
                                    print(
                                        f"Предупреждение: Части для {unique_name} в метке {label} не являются списком.  Это вызовет проблемы")
                        else:
                            print(
                                f"Предупреждение: Данные для метки {label} не являются словарем.  Это вызовет проблемы")

            with open(INDEX_ID_MAP_PATH, "r") as f:
                index_id_map = json.load(f)

                if isinstance(index_id_map, dict):
                    for label, data in index_id_map.items():
                        if isinstance(data, dict):
                            for unique_name_chunk_index, faiss_index_id in data.items():
                                if not isinstance(faiss_index_id, int):
                                    print(
                                        f"Предупреждение: ID индекса FAISS для {unique_name_chunk_index} в метке {label} не является целым числом. ПРЕОБРАЗУЕМ!")
                                    try:
                                        index_id_map[label][unique_name_chunk_index] = int(faiss_index_id)
                                    except ValueError:
                                        print(
                                            f"ОШИБКА: Не удалось преобразовать ID индекса FAISS '{faiss_index_id}' в целое число!")
                                        # Рассмотрите возможность выхода из программы здесь, так как данные повреждены
                                    except TypeError:  # Если faiss_index_id вообще None.
                                        print(f"ОШИБКА: faiss_index_id имеет неожиданный тип: {type(faiss_index_id)}")
                                        # Рассмотрите возможность выхода из программы здесь, так как данные повреждены
                        else:
                            print(
                                f"Предупреждение: Данные для метки {label} не являются словарем.  Это вызовет проблемы")

            print("Метаданные (document_store, index_id_map) успешно загружены.")
        except FileNotFoundError:
            print("Файлы метаданных не найдены. Будут использованы пустые значения.")
            document_store = {}
            index_id_map = {}
        except Exception as e:
            print(f"Ошибка при загрузке метаданных: {e}")
            traceback.print_exc()  # <-- ADDED
            document_store = {}
            index_id_map = {}

        # Загрузка индекса FAISS
        # --- Загрузка индекса FAISS ---
        try:
            print(f"Попытка загрузить индекс FAISS из: {FAISS_INDEX_PATH}")
            if os.path.exists(FAISS_INDEX_PATH):
                print(f"Файл индекса FAISS найден: {FAISS_INDEX_PATH}")
                try:
                    index = faiss.read_index(FAISS_INDEX_PATH)
                    print(f"Индекс FAISS успешно загружен из {FAISS_INDEX_PATH}")
                    print(
                        f"index.this.own(): {index.this.own() if hasattr(index, 'this') and hasattr(index.this, 'own') else 'N/A'}")
                except Exception as e:
                    print(f"Ошибка при ЧТЕНИИ индекса FAISS: {e}")
                    traceback.print_exc()
                    index = None
            else:
                print("Индекс FAISS не найден. Будет создан новый.")
                dimension = 768
                print(f"Создание нового индекса с размерностью: {dimension}")
                index = faiss.IndexFlatIP(dimension)
                print(f"Новый пустой индекс FAISS создан с размерностью {dimension}.")

                # Убедитесь, что каталог существует
                directory = os.path.dirname(FAISS_INDEX_PATH)
                if directory:
                    os.makedirs(directory, exist_ok=True)

                try:
                    faiss.write_index(index, FAISS_INDEX_PATH)
                    print(f"Пустой индекс FAISS успешно сохранен в: {FAISS_INDEX_PATH}")
                except Exception as e:
                    print(f"Ошибка при создании или сохранении пустого индекса: {e}")
                    traceback.print_exc()
                    index = None
        except Exception as e:
            print(f"Критическая ошибка при загрузке FAISS индекса: {e}")
            traceback.print_exc()
            index = None

        if index is None:
            print("Критическая ошибка: Индекс FAISS не был загружен!")
        else:
            print(f"Индекс FAISS успешно инициализирован: {index.ntotal} векторов, размерность {index.d}")
        if index is None:  # <-- ADDED
            print("Критическая ошибка: Индекс FAISS не был загружен!")  # <-- ADDED
        else:
            print(f"Индекс FAISS успешно инициализирован: {index.ntotal} векторов, размерность {index.d}")  # <-- ADDED

        yield  # <- Здесь приложение начинает обрабатывать запросы

        # --- Фаза завершения ---
        print("Завершение работы FastAPI приложения...")
        try:
            print(f"Сохранение индекса FAISS в: {FAISS_INDEX_PATH}")
            faiss.write_index(index, FAISS_INDEX_PATH)
            print(f"Индекс FAISS успешно сохранен в: {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"Ошибка при сохранении индекса FAISS: {e}")
            traceback.print_exc()  # <-- ADDED

        try:
            with open(DOCUMENT_STORE_PATH, "w") as f:
                json.dump(document_store, f)
            with open(INDEX_ID_MAP_PATH, "w") as f:
                json.dump(index_id_map, f)
            print("Метаданные (document_store, index_id_map) успешно сохранены.")
        except Exception as e:
            print(f"Ошибка при сохранении метаданных: {e}")
            traceback.print_exc()


    except Exception as e:
        print(f"Общая ошибка при запуске или завершении работы приложения: {e}")
        traceback.print_exc()


# --- Приложение FastAPI ---
# Установите стратегию разделения памяти
torch.multiprocessing.set_sharing_strategy('file_system')

# Ограничьте количество потоков (если проблема в многопоточности)
os.environ["OMP_NUM_THREADS"] = "1"

# Выберите устройство (MPS для M1, CUDA для NVIDIA)
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")

app = FastAPI(lifespan=lifespan)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)


# Определите save_index как асинхронную функцию в области ее вызова
async def save_index():
    global index
    try:
        print(f"Сохранение индекса FAISS в: {FAISS_INDEX_PATH}")
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"Индекс FAISS успешно сохранен в: {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"Ошибка при сохранении индекса FAISS: {e}")


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

@app.get("/ready")  # <--- ADDED
async def ready():
    global index
    if index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен.")
    return {"status": "ok"}


@app.post("/multiple_files/", response_model=MultipleFileUploadResponse, dependencies=[Depends(verify_token)])
async def upload_multiple_files(
        background_tasks: BackgroundTasks,
        label: str,  # Обязательный параметр
        files: List[UploadFile] = File(...),
        document_store: dict = Depends(get_document_store),
        index_id_map: dict = Depends(get_index_id_map),
        faiss_index: faiss.Index = Depends(get_faiss_index)
):
    """Загружает несколько файлов (любого типа) и векторизует их."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    global index

    responses = []
    for file in files:
        try:
            content = await file.read()
            text = content.decode(errors='ignore')  # Обработка ошибок декодирования
            unique_name = file.filename  # Использовать имя файла в качестве unique_name

            text = preprocess_text(text)
            text_chunks = chunk_text(text)

            # --- Обновление существующего или создание нового ---
            if label not in document_store:
                document_store[label] = {}

            if unique_name in document_store[label]:
                # Обновление: Сначала удалите старые вложения из индекса FAISS

                print(f"Обновление существующего документа: {unique_name} в метке: {label}")

                old_chunk_count = len(document_store[label][unique_name])

                # Удаление старых из FAISS
                if label in index_id_map and unique_name in index_id_map[label]:
                    for i in range(old_chunk_count):
                        unique_name_chunk_index = f"{unique_name}_{i}"
                        if unique_name_chunk_index in index_id_map[label]:
                            faiss_index_id = index_id_map[label][unique_name_chunk_index]

                            if faiss_index_id >= 0 and faiss_index_id < index.ntotal:
                                zero_embedding = np.zeros((1, index.d), dtype=np.float32)

                                # **Критическое логирование**
                                print("-------------------------------------")
                                print(
                                    f"Перед index.assign (первый вызов?):")  # Добавлено для дебага. Первый ли это вызов или нет?
                                print(f"  i: {i}")
                                print(f"  label: {label}")
                                print(f"  unique_name: {unique_name}")
                                print(f"  unique_name_chunk_index: {unique_name_chunk_index}")
                                print(f"  faiss_index_id: {faiss_index_id}")
                                print(f"  index.ntotal: {index.ntotal}")
                                print(f"  Shape of zero_embedding: {zero_embedding.shape}")
                                print("-------------------------------------")

                                try:
                                    random_embedding = np.random.rand(1, index.d).astype(np.float32)
                                    faiss.normalize_L2(random_embedding)
                                    index.assign(np.array([faiss_index_id], dtype=np.int64), random_embedding)
                                    print(f"Удален фрагмент {i} с id faiss {faiss_index_id} из индекса.")
                                except Exception as assign_error:
                                    print(f"Ошибка при вызове index.assign: {assign_error}")
                                    traceback.print_exc()

                            else:
                                print(
                                    f"Предупреждение: faiss_index_id {faiss_index_id} находится вне диапазона [0, {index.ntotal}).")

                            if label in index_id_map and unique_name_chunk_index in index_id_map[label]:
                                del index_id_map[label][unique_name_chunk_index]
                            else:
                                print(
                                    f"Предупреждение: unique_name_chunk_index {unique_name_chunk_index} не найден в index_id_map")

                document_store[label][unique_name] = []  # Сброс списка фрагментов

            else:
                print(f"Создание нового документа: {unique_name} в метке: {label}")
                document_store[label][unique_name] = []

            # Векторизация и добавление в FAISS
            for i, chunk in enumerate(text_chunks):
                embedding = embedding_model.encode(chunk, convert_to_tensor=False)
                embedding = np.float32(embedding)
                embedding = embedding.reshape(1, -1)  # Добавляем изменение формы
                add_to_faiss_index(embedding, faiss_index)  # Использовать вспомогательную функцию

                unique_name_chunk_index = f"{unique_name}_{i}"
                if label not in index_id_map:
                    index_id_map[label] = {}
                if unique_name not in index_id_map[label]:
                    index_id_map[label][unique_name] = {}
                index_id_map[label][unique_name_chunk_index] = faiss_index.ntotal - 1  # Правильное присваивание

                document_store[label][unique_name].append(chunk)

            index = faiss_index
            # index = faiss_index #Не нужно
            try:
                await save_index()  # Блокирующий вызов.
            except Exception as e:
                print(f"Ошибка при сохранении индекса: {e}")
                traceback.print_exc()

            responses.append(FileUploadResponse(filename=file.filename, label=label, unique_name=unique_name))

        except Exception as e:
            traceback.print_exc()
            responses.append(FileUploadResponse(filename=file.filename, label=label, unique_name=f"Ошибка: {e}"))

    return MultipleFileUploadResponse(files=responses)


@app.post("/context/", response_model=ContextResponse, dependencies=[Depends(verify_token)])
async def get_context(query: Query, document_store: dict = Depends(get_document_store),
                      index_id_map: dict = Depends(get_index_id_map),
                      faiss_index: faiss.Index = Depends(get_faiss_index)):
    """Получает контекст."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    global index

    query_text = preprocess_text(query.text)  # Предварительная обработка текста запроса
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
    query_embedding = np.float32(query_embedding)  # Ensure it's float32
    query_embedding = query_embedding.reshape(1, -1)

    # Print embedding shape
    print(f"query_embedding shape: {query_embedding.shape}")

    faiss.normalize_L2(query_embedding)

    D, I = faiss_index.search(query_embedding, TOP_K)

    # Print D and I
    print(f"Distances (D): {D}")
    print(f"Indices (I): {I}")

    # Build a dictionary of context documents grouped by label
    context_documents: Dict[str, List[str]] = {}

    # Initialize context_documents with all labels from index_id_map
    for label in query.labels:
        context_documents[label] = []

    for i in range(len(I[0])):
        faiss_index_val = I[0][i]

        # Add this check:
        if faiss_index_val < 0 or faiss_index_val >= faiss_index.ntotal:
            print(f"Внимание: Индекс FAISS {faiss_index_val} вне допустимого диапазона [0, {faiss_index.ntotal}).  Пропускаем.")
            continue

        for label in query.labels: # Изменено
            if label in index_id_map: # Добавлено
                for unique_name_chunk_index, index_val in index_id_map[label].items():  # Добавлено label
                    if index_val == faiss_index_val:  # Найдено соответствие
                        unique_name, chunk_index = unique_name_chunk_index.rsplit("_", 1)  # Разделить unique_name_0 обратно на unique_name, 0

                        if label in document_store and unique_name in document_store[label]:
                            try:
                                chunk_index_int = int(chunk_index)
                                if chunk_index_int < len(document_store[label][unique_name]):
                                    chunk = document_store[label][unique_name][chunk_index_int]
                                    context_documents[label].append(chunk)
                                else:
                                    print(f"Предупреждение: Индекс части {chunk_index} вне диапазона для unique_name {unique_name} в метке {label}.")
                            except ValueError:
                                print(f"Предупреждение: Не удалось преобразовать индекс части {chunk_index} в целое число.")

                            break  # Перейдите к следующему результату faiss, нет смысла искать другие метки после того, как индекс совпал.

                        else:
                            print(f"Предупреждение: unique_name {unique_name} или метка {label} не найдены в document_store.")

    # --- Re-ranking с помощью Cross-Encoder ---
    ranked_context: Dict[str, List[str]] = {}
    for label, docs in context_documents.items():
        if not docs:
            ranked_context[label] = []  # Пустой список, указывающий на отсутствие документов
            continue

        # Add query text to CrossEncoder input
        cross_encoder_input = [[query_text, doc] for doc in docs]
        try:
            cross_encoder_scores = cross_encoder.predict(cross_encoder_input)

            # Сортируем документы по убыванию scores
            ranked_documents = [doc for _, doc in sorted(zip(cross_encoder_scores, docs), reverse=True)]
            ranked_context[label] = ranked_documents

        except Exception as e:
            print(f"Ошибка при работе CrossEncoder: {e}")
            ranked_context[label] = ["Ошибка при перекрестном кодировании."]  # Отметить ошибку

    return ContextResponse(context=ranked_context)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
