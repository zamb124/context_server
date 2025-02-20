# --- START OF FILE file_utils.py ---
import io
import json

import docx
import openpyxl
import pdfplumber


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