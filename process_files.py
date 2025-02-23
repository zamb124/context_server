import json
import logging
import os
import time

import requests

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_file(filepath):
    """
    Processes a single file, extracts metadata, and uploads it using requests.

    Args:
        filepath (str): The full path to the file.
    """
    logging.info(f"Начинаю обработку файла: {filepath}")

    try:
        logging.info(f"Открываю файл: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            file_content = f.read()  # Read the rest of the file content
        logging.info(f"Файл {filepath} открыт и содержимое прочитано.")

    except FileNotFoundError:
        logging.error(f"Ошибка: Файл не найден: {filepath}")
        return
    except Exception as e:
        logging.error(f"Ошибка при чтении файла {filepath}: {e}")
        return

    # Extract filename
    filename = os.path.splitext(os.path.basename(filepath))[0]
    logging.info(f"Извлечено имя файла: {filename}")

    # Parse metadata
    metadata = {"category": "partner"}
    if "metadata=" in first_line:
        logging.info("Разбираю метаданные из первой строки.")
        metadata_str = first_line.split("metadata=")[1]
        metadata_pairs = metadata_str.split(";")
        for pair in metadata_pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)
                metadata[key.strip()] = value.strip()
        logging.info(f"Извлечены метаданные: {metadata}")

    metadata["partner"] = filename

    final_metadata = {
        "category": "sales",
        "type": "hubspot_profile",
        "city": metadata.get("city", "") if metadata.get("city", "") != 'None' else '',
        "country": metadata.get("country", "") if metadata.get("country", "") != 'None' else '',
        "industry": metadata.get("industry", "") if metadata.get("industry", "") != 'None' else '',
        "author": metadata.get("author", "") if metadata.get("author", "") != 'None' else '',
        "partner": filename or '',
        "market": metadata.get("country", "") if metadata.get("country", "") != 'None' else ''
    }
    logging.info(f"Итоговые метаданные: {final_metadata}")

    # Prepare request
    url = 'https://foodforce.tech/add_document/'
    headers = {'Authorization': f'Bearer {config.CHAT_TOKEN}'}
    params = {
        'label': 'hubspot',
        **final_metadata,
        'document_id': filename + '.txt',
        'id': filename + '.txt'
    }
    files = {'file': (filename + '.txt', file_content)}
    logging.info(f"Подготовлены данные запроса для {filepath}")

    # Send request with retries
    retries = 0
    max_retries = 30  # 5 with increasing + 25 at 20 sec
    delay = 1

    while retries < max_retries:
        retries += 1
        logging.info(f"Отправка POST запроса (попытка {retries}/{max_retries}) для {filepath} с задержкой {delay} сек.")
        try:
            response = requests.post(url, headers=headers, params=params, files=files)
            response.raise_for_status()  # Raise HTTPError for bad responses

            logging.info(f"Файл {filepath} успешно обработан.")
            logging.info(f"Код состояния ответа: {response.status_code}")
            logging.info(f"Содержимое ответа: {response.text}")
            return  # Exit loop if successful

        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при обработке {filepath}: {e}")
            if retries == 5:
                delay = 2
            elif retries == 7:
                delay = 5
            elif retries == 9:
                delay = 10
            elif retries >= 10:
                delay = 20  # Remaining retries at 20 sec
            time.sleep(delay)

    logging.error(f"Не удалось обработать файл {filepath} после {max_retries} попыток.")


def main():
    """
    Main function to iterate through files and process them.
    """
    directory = "hubspot_company_data"

    logging.info(f"Начинаю основной процесс в каталоге: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            process_file(filepath)
    logging.info("Основной процесс завершен.")


if __name__ == "__main__":
    main()
