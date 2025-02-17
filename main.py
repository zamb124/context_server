import json
import mimetypes
import multiprocessing
import os
import re
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import List, Union, Optional

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Header, status
from pydantic import BaseModel, validator
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import config


# --- Load environment variables ---

# --- Модели данных ---
class DocumentBase(BaseModel):
    id: str = None
    text: str

    @validator('id', pre=True, always=True)
    def set_id(cls, value):
        return value or str(uuid.uuid4())


class Query(BaseModel):
    text: str


class ContextResponse(BaseModel):
    context: str


class BatchDocuments(BaseModel):
    documents: List[DocumentBase]


class FileUploadResponse(BaseModel):
    filename: str
    document_id: str


class MultipleFileUploadResponse(BaseModel):
    files: List[FileUploadResponse]


# --- Конфигурация ---
FAISS_INDEX_PATH = "y_faiss_index.bin"
DOCUMENT_STORE_PATH = "document_store.json"
INDEX_ID_MAP_PATH = "index_id_map.json"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10
MAX_TEXT_LENGTH = 2048  # Увеличиваем длину текста для начала
HUBSPOT_DATA_FOLDER = "model_mini/hubspot_company_data"

# --- Authorization Token ---
CHAT_TOKEN = config.CHAT_TOKEN
if not CHAT_TOKEN:
    raise ValueError("CHAT_TOKEN environment variable not set. Please configure it in .env file.")

# --- Глобальные переменные ---
index = None
document_store = {}
index_id_map = {}


# --- Dependency Injection ---
def get_faiss_index():
    global index
    return index


def get_document_store():
    global document_store
    return document_store


def get_index_id_map():
    global index_id_map
    return index_id_map


# --- Utilities ---
def preprocess_text(text: str) -> str:
    """Предварительная обработка текста: обрезка, удаление не-ASCII символов."""
    # text = text[:MAX_TEXT_LENGTH]  #  УБРАЛИ Обрезаем текст до MAX_TEXT_LENGTH - теперь обрезается при чанкинге, если нужно
    # text = re.sub(r'[^\x00-\x7F]+', '', text)  #  УБРАЛИ Удаляем все не-ASCII символы
    text = text.strip()  # Удаляем пробелы в начале и конце
    return text


def chunk_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
    """Разбивает текст на чанки фиксированной длины."""
    chunks = []
    for i in range(0, len(text), max_length):
        chunks.append(text[i:i + max_length])
    return chunks


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, document_store, index_id_map

    # --- Startup phase ---
    print("Запуск FastAPI приложения...")
    try:
        # Load metadata
        try:
            with open(DOCUMENT_STORE_PATH, "r") as f:
                document_store = json.load(f)
            with open(INDEX_ID_MAP_PATH, "r") as f:
                index_id_map = json.load(f)
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

        # Load Faiss Index
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

                # Ensure the directory exists
                os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

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

        # --- Shutdown phase ---
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
            traceback.print_exc()  # <-- ADDED

    except Exception as e:
        print(f"Общая ошибка при запуске или завершении работы приложения: {e}")
        traceback.print_exc()


# --- FastAPI App ---
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


# Define save_index as an async function within the scope where it is called
async def save_index():
    global index
    try:
        print(f"Сохранение индекса FAISS в: {FAISS_INDEX_PATH}")
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"Индекс FAISS успешно сохранен в: {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"Ошибка при сохранении индекса FAISS: {e}")


# --- Dependency for Token Authentication ---
async def verify_token(authorization: Optional[str] = Header(None)):
    """
    Verify the token from the Authorization header (Bearer scheme).
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Use Bearer.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token != CHAT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
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


@app.post("/documents/", response_model=DocumentBase, dependencies=[Depends(verify_token)])
async def add_document(
        background_tasks: BackgroundTasks,
        document: Union[DocumentBase, None] = None,
        file: UploadFile = File(None, description="Upload a .txt file"),
        document_store: dict = Depends(get_document_store),
        index_id_map: dict = Depends(get_index_id_map),
        faiss_index: faiss.Index = Depends(get_faiss_index)
):
    """Добавляет новый документ (JSON или TXT)."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    global index

    text = None
    doc_id = None

    if document:
        doc_id = document.id
        text = document.text
    elif file:
        try:
            content = await file.read()
            text = content.decode(errors='ignore')  # Обработка ошибок декодирования
            doc_id = str(uuid.uuid4())  # Генерируем ID для файла
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при чтении файла: {e}")
    else:
        raise HTTPException(status_code=400, detail="Необходимо предоставить JSON или TXT файл")

    if doc_id in document_store:
        raise HTTPException(status_code=400, detail="Документ с таким ID уже существует.")

    text = preprocess_text(text)  # Предварительная обработка текста

    # Разбить текст на чанки
    text_chunks = chunk_text(text)

    for chunk in text_chunks:
        try:
            embedding = embedding_model.encode(chunk, convert_to_tensor=False)
            embedding = np.float32(embedding)  # Ensure it's float32
            embedding = embedding.reshape(1, -1)  # Reshape to 2D array
            faiss.normalize_L2(embedding)

            print(embedding.shape)
            faiss_index.add(embedding)  # Добавляем в индекс

            chunk_doc_id = f"{doc_id}_{len(document_store)}"  # Уникальный ID для чанка
            document_store[chunk_doc_id] = chunk
            index_id_map[chunk_doc_id] = faiss_index.ntotal - 1

            index = faiss_index  # Update the global index
            background_tasks.add_task(save_index)  # Use local async function

        except Exception as e:
            print(f"Ошибка при добавлении чанка документа {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при добавлении документа: {e}")

    return DocumentBase(id=doc_id, text=text)  # Вернем исходный документ, а не чанк


@app.post("/multiple_files/", response_model=MultipleFileUploadResponse, dependencies=[Depends(verify_token)])
async def upload_multiple_files(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        document_store: dict = Depends(get_document_store),
        index_id_map: dict = Depends(get_index_id_map),
        faiss_index: faiss.Index = Depends(get_faiss_index)
):
    """Загружает несколько файлов (любого типа) и векторизует их."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    responses = []
    for file in files:
        try:
            content = await file.read()
            text = content.decode(errors='ignore')  # Обработка ошибок декодирования
            doc_id = str(uuid.uuid4())

            text = preprocess_text(text)  # Предварительная обработка текста
            text_chunks = chunk_text(text)

            for chunk in text_chunks:
                embedding = embedding_model.encode(chunk, convert_to_tensor=False)
                embedding = np.float32(embedding)  # Ensure it's float32
                embedding = embedding.reshape(1, -1)
                faiss.normalize_L2(embedding)

                faiss_index.add(embedding)
                chunk_doc_id = f"{doc_id}_{len(document_store)}"
                document_store[chunk_doc_id] = chunk
                index_id_map[chunk_doc_id] = faiss_index.ntotal - 1

            index = faiss_index
            background_tasks.add_task(save_index)

            responses.append(FileUploadResponse(filename=file.filename, document_id=doc_id))

        except Exception as e:
            responses.append(FileUploadResponse(filename=file.filename, document_id=f"Ошибка: {e}"))

    return MultipleFileUploadResponse(files=responses)


@app.post("/load_hubspot_data/", dependencies=[Depends(verify_token)])
async def load_hubspot_data(background_tasks: BackgroundTasks,
                            document_store: dict = Depends(get_document_store),
                            index_id_map: dict = Depends(get_index_id_map),
                            faiss_index: faiss.Index = Depends(get_faiss_index)
                            ):
    """Рекурсивно загружает и векторизует все файлы из папки HUBSPOT_DATA_FOLDER."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    responses = []
    global index

    if not os.path.exists(HUBSPOT_DATA_FOLDER):
        raise HTTPException(status_code=400, detail=f"Папка {HUBSPOT_DATA_FOLDER} не найдена.")

    # Clear the index_id_map and document_store before loading
    index_id_map.clear()
    document_store.clear()

    for root, _, files in os.walk(HUBSPOT_DATA_FOLDER):
        for filename in files:
            if filename == ".DS_Store":  # Игнорируем .DS_Store
                print("Пропускаем файл .DS_Store")
                continue

            filepath = os.path.join(root, filename)
            mime_type, _ = mimetypes.guess_type(filepath)

            try:
                with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
                    text = f.read()

                text = preprocess_text(text)  # Предварительная обработка текста
                text_chunks = chunk_text(text)

                doc_id = str(uuid.uuid4()) # Генерируем ID для исходного файла


                for i, chunk in enumerate(text_chunks):
                    # embedding = embedding_model.encode(text, convert_to_tensor=False)
                    embedding = embedding_model.encode(chunk, convert_to_tensor=False)
                    embedding = np.float32(embedding)  # Ensure it's float32
                    embedding = embedding.reshape(1, -1)  # Reshape to 2D array
                    faiss.normalize_L2(embedding)

                    # --- Add debug prints ---
                    print(f"Файл: {filename}, doc_id: {doc_id}, chunk_id: {i}")
                    print(f"embedding shape: {embedding.shape}")
                    print(f"embedding norm: {np.linalg.norm(embedding)}")
                    # --- End debug prints ---

                    faiss_index.add(embedding)

                    chunk_doc_id = f"{doc_id}_{i}"  # Уникальный ID для чанка
                    document_store[chunk_doc_id] = chunk
                    index_id_map[chunk_doc_id] = faiss_index.ntotal - 1

                    # --- Check consistency ---
                    if chunk_doc_id not in document_store:
                        print(f"Ошибка: doc_id {chunk_doc_id} не найден в document_store после добавления.")
                    if chunk_doc_id not in index_id_map:
                        print(f"Ошибка: doc_id {chunk_doc_id} не найден в index_id_map после добавления.")
                    if index_id_map[chunk_doc_id] != faiss_index.ntotal - 1:
                        print(
                            f"Ошибка: index_id_map[{chunk_doc_id}] ({index_id_map[chunk_doc_id]}) не соответствует faiss_index.ntotal - 1 ({faiss_index.ntotal - 1}).")
                    # --- End check consistency ---

                responses.append(FileUploadResponse(filename=filename, document_id=doc_id)) # ID оригинального файла, а не чанка
                print(f"Файл {filename} успешно векторизован.")

            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"Ошибка при обработке файла {filename}: {e}\n{error_traceback}")
                responses.append(FileUploadResponse(filename=filename, document_id=f"Ошибка: {e}"))

    index = faiss_index
    background_tasks.add_task(save_index)

    return MultipleFileUploadResponse(files=responses)


@app.post("/batch_documents/", dependencies=[Depends(verify_token)])
async def add_batch_documents(
        batch: BatchDocuments,
        background_tasks: BackgroundTasks,
        document_store: dict = Depends(get_document_store),
        index_id_map: dict = Depends(get_index_id_map),
        faiss_index: faiss.Index = Depends(get_faiss_index)
):
    """Добавляет несколько документов (только JSON)."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    global index

    embeddings = []
    ids = []
    for document in batch.documents:
        if document.id in document_store:
            raise HTTPException(status_code=400, detail=f"Документ с ID {document.id} уже существует.")

        text = preprocess_text(document.text)  # Предварительная обработка текста

        text_chunks = chunk_text(text)
        for chunk in text_chunks:
            embedding = embedding_model.encode(chunk, convert_to_tensor=False)
            embedding = np.float32(embedding)  # Ensure it's float32
            embeddings.append(embedding)

    if not embeddings:
        return {"message": "Пакет не содержит документов."}

    embeddings_array = np.array(embeddings).astype('float32')  # Convert list to numpy array

    # Normalize embeddings
    faiss.normalize_L2(embeddings_array)

    # Add embeddings to index
    faiss_index.add(embeddings_array)

    start_index = faiss_index.ntotal - len(batch.documents)
    for i, doc_id in enumerate(ids):
        index_id_map[doc_id] = start_index + i

    index = faiss_index
    background_tasks.add_task(save_index)

    return {"message": f"Добавлено {len(batch.documents)} документов."}


@app.delete("/documents/{document_id}", dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, background_tasks: BackgroundTasks,
                          document_store: dict = Depends(get_document_store),
                          index_id_map: dict = Depends(get_index_id_map),
                          faiss_index: faiss.Index = Depends(get_faiss_index)):
    """Удаляет документ."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    global index

    # Remove chunks associated with the document_id
    keys_to_delete = [key for key in document_store if key.startswith(f"{document_id}_")]
    if document_id not in document_store:
        if not keys_to_delete: # Если нет чанков и нет оригинального ID, значит, нечего удалять
             raise HTTPException(status_code=404, detail="Документ не найден.")
    if document_id in document_store:
        del document_store[document_id]
        if document_id in index_id_map:
             del index_id_map[document_id]


    for key in keys_to_delete:
        del document_store[key]
        if key in index_id_map:
            del index_id_map[key]


    background_tasks.add_task(save_index)


    try:
        with open(DOCUMENT_STORE_PATH, "w") as f:
            json.dump(document_store, f)
        with open(INDEX_ID_MAP_PATH, "w") as f:
            json.dump(index_id_map, f)
    except Exception as e:
        print(f"Ошибка при сохранении index_id_map или document_store: {e}")

    return {"message": f"Документ {document_id} и все его чанки успешно удалены."}


@app.put("/documents/{document_id}", response_model=DocumentBase, dependencies=[Depends(verify_token)])
async def update_document(document_id: str, background_tasks: BackgroundTasks,
                          document: Union[DocumentBase, None] = None,
                          file: UploadFile = File(None, description="Upload a .txt file"),
                          document_store: dict = Depends(get_document_store),
                          index_id_map: dict = Depends(get_index_id_map),
                          faiss_index: faiss.Index = Depends(get_faiss_index)
                          ):
    """Обновляет документ (JSON или TXT)."""
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="Индекс FAISS не загружен. Попробуйте позже.")

    if document and file:
        raise HTTPException(status_code=400, detail="Предоставьте JSON или TXT, но не оба одновременно.")

    if document:
        if document_id != document.id:
            raise HTTPException(status_code=400, detail="ID в URL не совпадает с ID в теле запроса.")
        if document_id not in document_store and not any(key.startswith(f"{document_id}_") for key in document_store): # checking  for original ID and chunks
            raise HTTPException(status_code=404, detail="Документ не найден.")
    elif file:
        if not document_id in document_store and not any(key.startswith(f"{document_id}_") for key in document_store):
            raise HTTPException(status_code=404, detail="Документ не найден.")
    else:
        raise HTTPException(status_code=400, detail="Необходимо предоставить JSON или TXT")

    # Изменим так, чтобы удалялся только старый документ, а новый можно было добавлять
    await delete_document(document_id, background_tasks, document_store, index_id_map, faiss_index)

    text = None  # Initialize text to None
    if document:
        # Добавим новую запись с JSON
        text = preprocess_text(document.text)  # Предварительная обработка текста
        document.text = text  # Обновляем текст в документе
        new_doc = DocumentBase(id=document_id, text=text)
        await add_document(background_tasks=background_tasks, document=new_doc, document_store=document_store,
                           index_id_map=index_id_map, faiss_index=faiss_index)  # Add the updated document
    else:
        # Добавим новую запись с TXT
        content = await file.read()
        text = content.decode(errors='ignore')  # Обработка ошибок декодирования
        text = preprocess_text(text)  # Предварительная обработка текста

        text_chunks = chunk_text(text)

        for chunk in text_chunks:
            embedding = embedding_model.encode(chunk, convert_to_tensor=False)
            embedding = embedding.astype(np.float32)  # Ensure it's float32
            embedding = embedding.reshape(1, -1)  # Reshape to 2D array
            faiss.normalize_L2(embedding)

            faiss_index.add(embedding)
            chunk_doc_id = f"{document_id}_{len(document_store)}"
            document_store[chunk_doc_id] = chunk
            index_id_map[chunk_doc_id] = faiss_index.ntotal - 1

        index = faiss_index
        background_tasks.add_task(save_index)

    return DocumentBase(id=document_id, text=text)  # Return the updated document


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

    context_documents = []
    for i in range(len(I[0])):
        faiss_index_val = I[0][i]

        # Add this check:
        if faiss_index_val < 0 or faiss_index_val >= faiss_index.ntotal:
            print(f"Внимание: Индекс FAISS {faiss_index_val} вне допустимого диапазона [0, {faiss_index.ntotal}).  Пропускаем.")
            continue

        # Find chunk_doc_id instead of doc_id
        chunk_doc_id = next((doc_id for doc_id, index_val in index_id_map.items() if index_val == faiss_index_val), None)

        if chunk_doc_id:
            if chunk_doc_id in document_store:  # Double check
                context_documents.append(document_store[chunk_doc_id])
            else:
                print(f"Внимание: doc_id {chunk_doc_id} найден в index_id_map, но не в document_store. Это неконсистентность данных!")
        else:
            print(f"Внимание: Не удалось найти document_id для индекса {faiss_index_val} в index_id_map. Это серьезная ошибка!")

    # --- Re-ranking с помощью Cross-Encoder ---
    if not context_documents:
        return ContextResponse(context="No relevant documents found.")

    # Add query text to CrossEncoder input
    cross_encoder_input = [[query_text, doc] for doc in context_documents]
    try:
        cross_encoder_scores = cross_encoder.predict(cross_encoder_input)

        # Сортируем документы по убыванию scores
        ranked_documents = [doc for _, doc in sorted(zip(cross_encoder_scores, context_documents), reverse=True)]

        context = "\n\n".join(ranked_documents)
        return ContextResponse(context=context)

    except Exception as e:
        print(f"Ошибка при работе CrossEncoder: {e}")
        return ContextResponse(context="Error during cross-encoding.")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)