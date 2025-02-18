# --- Дополнительные библиотеки для работы с файлами ---
import io
import json
from typing import List, Dict, Optional, Any

import chromadb
import docx
import nltk
import openpyxl
import pdfplumber
import uvicorn
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException, Depends, Header, status, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from config import config

# --- Конфигурация ---
CHROMA_DB_PATH = "chroma_db"  # Папка, где будет храниться база данных ChromaDB
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # ИЗМЕНЕНО

# --- Токен авторизации ---
CHAT_TOKEN = config.CHAT_TOKEN
if not CHAT_TOKEN:
    raise ValueError("Не установлена переменная окружения CHAT_TOKEN. Пожалуйста, настройте ее в файле .env.")


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


# --- Инициализация ChromaDB ---
# Использовать default_embedding_function
# default_ef = embedding_functions.DefaultEmbeddingFunction()  # УДАЛЕНО

# Использование SentenceTransformerEmbeddingFunction
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


# --- Функция разбиения текста на абзацы ---
def split_text_into_paragraphs(text: str) -> List[str]:
    """
    Разбивает текст на абзацы.

    Args:
        text: Исходный текст.

    Returns:
        Список абзацев текста.
    """
    paragraphs = text.split("\n\n")  # Разделение по двойным переносам строк (типичный признак абзаца)
    return [p.strip() for p in paragraphs if p.strip()]  # Убираем пробелы и пустые абзацы


# --- Функция разбиения на предложения ---
def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    """
    Разбивает абзац на предложения с использованием nltk.sent_tokenize.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    return nltk.sent_tokenize(paragraph)


def split_text_semantically(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Разбивает текст на семантические чанки с использованием Langchain.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Порядок важен
        length_function=len,  # Используем стандартную функцию len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# --- FastAPI App ---
app = FastAPI()


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


# --- Функции для извлечения текста из разных форматов файлов ---
def extract_text_from_pdf(file: io.BytesIO) -> str:
    """Извлекает текст из PDF-файла."""
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Добавляем or "" для обработки пустых страниц
    return text


def extract_text_from_txt(file: io.BytesIO) -> str:
    """Извлекает текст из TXT-файла."""
    return file.read().decode("utf-8")


def extract_text_from_docx(file: io.BytesIO) -> str:
    """Извлекает текст из DOCX-файла."""
    doc = docx.Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def extract_text_from_json(file: io.BytesIO) -> str:
    """Извлекает текст из JSON-файла (предполагается, что это просто текстовый JSON)."""
    try:
        data = json.load(file)
        return json.dumps(data, indent=4)  # Форматируем JSON для читаемости
    except json.JSONDecodeError:
        raise ValueError("Некорректный JSON формат.")


def extract_text_from_xlsx(file: io.BytesIO) -> str:
    """Извлекает текст из XLSX-файла."""
    workbook = openpyxl.load_workbook(file)
    text = ""
    for sheet in workbook.sheetnames:
        worksheet = workbook[sheet]
        for row in worksheet.iter_rows():
            row_values = [str(cell.value) for cell in row if
                          cell.value is not None]  # Преобразуем все значения в строки и обрабатываем None
            text += ", ".join(row_values) + "\n"
    return text


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


@app.post("/query/", response_model=ContextResponse, dependencies=[Depends(verify_token)])
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
            print(f"Ошибка при запросе label {label}: {e}")  # Логируем ошибку, но продолжаем
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
