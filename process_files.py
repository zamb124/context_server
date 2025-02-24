import json
import logging
import os
import time
import re

import requests

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_text(text):
    """Removes all characters except letters and numbers from the text."""
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def process_file(filepath):
    """
    Processes a single JSON file, extracts data, and uploads it to the API.

    Args:
        filepath (str): The full path to the file.
    """
    logging.info(f"Начинаю обработку файла: {filepath}")

    try:
        logging.info(f"Открываю файл: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Файл {filepath} открыт и содержимое прочитано.")

    except FileNotFoundError:
        logging.error(f"Ошибка: Файл не найден: {filepath}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка: Некорректный JSON в файле {filepath}: {e}")
        return
    except Exception as e:
        logging.error(f"Ошибка при чтении файла {filepath}: {e}")
        return

    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Process company data
    company_data = {
        "city": data.get("city", ""),
        "name": data.get("name", "No name"),  # Add name
        "country": data.get("country", ""),
        "industry": data.get("industry", ""),
        "id": data.get("id", ""),
        "domain": data.get("domain", ""),
        "description": data.get("description", ""),
    }
    upload_data(company_data, filename, 'company')

    # Process contacts
    if 'contacts' in data and isinstance(data['contacts'], list):
        for contact in data['contacts']:
            upload_data(contact, filename, 'contact')

    # Process notes
    if 'notes' in data and isinstance(data['notes'], list):
        for note in data['notes']:
            upload_data(note, filename, 'note')

    # Process emails
    if 'emails' in data and isinstance(data['emails'], list):
        for email in data['emails']:
            upload_data(email, filename, 'email')

     # Process calls
    if 'calls' in data and isinstance(data['calls'], list):
        for call in data['calls']:
            upload_data(call, filename, 'call')


    logging.info(f"Файл {filepath} успешно обработан.")


def upload_data(item, filename, item_type):
    """
    Uploads a single item to the API.

    Args:
        item (dict): The data item to upload.
        filename (str): The base filename for metadata.
        item_type (str): The type of the item ('company', 'contact', 'note', 'email', 'call').
    """
    logging.info(f"Отправка данных типа {item_type} для файла {filename}")

    document_id = item['id']

    if item_type != 'company':
        document_id += '_' + str(hash(json.dumps(item)))

    metadata = {
        "category": "sales",
        "type": item_type,
        "author": "system",
        "partner": filename.lower(),
    }

    # Add specific metadata based on item_type
    if item_type == 'company':
        metadata['city'] = item.get("city", "")
        metadata['country'] = item.get("country", "")
        metadata['industry'] = item.get("industry", "")
        metadata['partner_search'] = clean_text(filename.lower())
        metadata['id'] = document_id
    else:
        metadata['city'] = ""
        metadata['country'] = ""
        metadata['industry'] = ""
        metadata['partner_search'] = ""
        metadata['id'] = document_id # Include ID in metadata for non-company types


    data_to_send = {
        "text": json.dumps(item),  # The entire item data as a JSON string
        "label": "hubspot",
        "document_id": document_id,
        "author": "system",
        "type": item_type,
        "chunk": False,
        "metadata": metadata
    }

    url = 'https://foodforce.tech/add_document/'
    headers = {'Authorization': f'Bearer {config.CHAT_TOKEN}'}


    # Send request with retries
    retries = 0
    max_retries = 30
    delay = 1

    while retries < max_retries:
        retries += 1
        logging.info(f"Отправка POST запроса (попытка {retries}/{max_retries}) типа {item_type} для {filename} с задержкой {delay} сек.")
        try:
            response = requests.post(url, headers=headers, json=data_to_send)  # Send JSON payload
            response.raise_for_status()  # Raise HTTPError for bad responses

            logging.info(f"Данные типа {item_type} для {filename} успешно отправлены.")
            logging.info(f"Код состояния ответа: {response.status_code}")
            logging.info(f"Содержимое ответа: {response.text}")
            return  # Exit loop if successful

        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при отправке данных типа {item_type} для {filename}: {e}")
            if retries == 5:
                delay = 2
            elif retries == 7:
                delay = 5
            elif retries == 9:
                delay = 10
            elif retries >= 10:
                delay = 20
            time.sleep(delay)

    logging.error(f"Не удалось отправить данные типа {item_type} для {filename} после {max_retries} попыток.")



def main():
    """
    Main function to iterate through files and process them.
    """
    directory = "hubspot_company_data"

    logging.info(f"Начинаю основной процесс в каталоге: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            process_file(filepath)
    logging.info("Основной процесс завершен.")


if __name__ == "__main__":
    main()