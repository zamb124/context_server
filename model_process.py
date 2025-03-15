import asyncio
import json
import logging
import os
import sys
import traceback

import fasttext
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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

# --- Загрузка моделей ---
fasttext_model_path = "lid.176.bin"
lang_model = fasttext.load_model(fasttext_model_path)

nltk.data.path.append('nltk')
try:
    stop_words = set(stopwords.words('english'))  # Или 'english'
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  # Или 'english'


def clean_text(text, lang='en'):
    """
    Очистка текста от стоп-слов и специальных символов.
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    cleaned_tokens = [w for w in word_tokens if w.lower() not in stop_words and w.isalnum()]
    return " ".join(cleaned_tokens)


class SummarizerModel:
    def __init__(self, model_name="csebuetnlp/mT5_multilingual_XLSum"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    async def load_model(self):
        """Асинхронная загрузка модели."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = pipeline("summarization", model=self.model_name, tokenizer=self.model_name, device=device)
            logger.info(f"Модель {self.model_name} успешно загружена на {device} в процессе {os.getpid()}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {self.model_name}: {e}")
            traceback.print_exc()
            raise

    def summarize(self, text: str, max_length: int = 1024, min_length: int = 30) -> str:
        """
        Генерирует summary для текста.
        Если входной текст превышает 512 токенов, разбивает его на чанки по 512 токенов,
        суммаризирует каждый чанк отдельно и объединяет результаты в один итоговый текст.
        """
        if not self.model:
            raise ValueError("Модель не загружена. Сначала вызовите load_model().")
        try:
            logger.info(f"Начало суммирования текста. Длина текста (символы): {len(text)}")
            if not text:
                raise ValueError("Пустой текст для суммаризации")

            max_tokens = 512  # Максимальное количество токенов для одного чанка

            # Токенизация текста с использованием токенизатора модели
            tokens = self.tokenizer.tokenize(text)
            num_tokens = len(tokens)
            logger.info(f"Количество токенов в тексте: {num_tokens}")

            # Если текст не превышает лимит, суммаризируем его напрямую
            if num_tokens <= max_tokens:
                summary = self.model(text, max_length=max_length, min_length=min_length)[0]['summary_text']
                logger.info(f"Суммирование завершено. Длина summary (символы): {len(summary)}")
                return summary

            # Если текст длиннее, разбиваем его на чанки по max_tokens токенов
            chunks = []
            for i in range(0, num_tokens, max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                # Преобразуем список токенов обратно в строку
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
            logger.info(f"Текст разбит на {len(chunks)} чанков по {max_tokens} токенов")

            # Суммаризация каждого чанка
            summaries = []
            for idx, chunk in enumerate(chunks):
                chunk_summary = self.model(chunk, max_length=max_length, min_length=min_length)[0]['summary_text']
                summaries.append(chunk_summary)
                logger.info(f"Чанк {idx + 1}/{len(chunks)} суммаризован. Длина summary: {len(chunk_summary)}")

            # Объединяем все суммаризации в один итоговый текст
            final_summary = " ".join(summaries)
            logger.info(f"Итоговое суммаризация завершена. Длина итогового summary: {len(final_summary)}")
            return final_summary

        except Exception as e:
            logger.error(f"Ошибка при суммировании текста: {e}")
            traceback.print_exc()
            raise


async def main():
    """Основная функция для запуска модели."""
    model = SummarizerModel()
    await model.load_model()
    # Сигнал о готовности модели
    print("MODEL_READY")  # Важно, чтобы основной процесс знал, когда модель готова
    sys.stdout.flush()

    # Ждем сигнала завершения работы
    while True:
        try:
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, sys.stdin.readline)  # Читаем строку из stdin

            if not line:
                logging.info("EOF received on stdin, exiting.")
                break

            try:
                request = json.loads(line.strip())
                text = request.get("text", "")
                question = request.get("question", "")  # Получаем вопрос

                if not text:
                    response = {"status": "error", "error": "Missing 'text' field in request"}
                else:
                    try:
                        summary = model.summarize(text)  # Вызываем summarize
                        response = {"status": "success", "summary": summary}
                    except Exception as e:
                        logging.error(f"Error during summarization: {e}")
                        response = {"status": "error", "error": str(e)}

            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError: {e}")
                response = {"status": "error", "error": f"Invalid JSON: {e}"}
            except Exception as e:
                logging.error(f"Error processing request: {e}")
                traceback.print_exc()
                response = {"status": "error", "error": str(e)}

            try:
                print(json.dumps(response))  # Отправляем JSON-ответ в stdout
                sys.stdout.flush()
            except Exception as e:
                logging.error(f"Error sending response: {e}")
                traceback.print_exc()



        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            traceback.print_exc()
            break


if __name__ == "__main__":
    asyncio.run(main())
