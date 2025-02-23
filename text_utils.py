import asyncio
import functools
from typing import List

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter


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


# Определите вспомогательную функцию run_sync для оборачивания синхронных функций
def run_sync(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    return wrapper


# Создайте синхронную версию split_text_semantically
def split_text_semantically_sync(text: str, chunk_size: int = 5000, chunk_overlap: int = 500) -> List[str]:
    """
    Синхронная версия для разбиения текста на семантические чанки с использованием Langchain.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Порядок важен
        length_function=len,  # Используем стандартную функцию len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Оберните вызов split_text_semantically в asyncio.to_thread
async def split_text_semantically(text: str, chunk_size: int = 5000, chunk_overlap: int = 500) -> List[str]:
    """
    Асинхронно разбивает текст на семантические чанки с использованием Langchain.
    """
    return await asyncio.to_thread(split_text_semantically_sync, text, chunk_size, chunk_overlap)
