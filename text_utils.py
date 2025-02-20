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