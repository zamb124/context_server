# telegram_integration.py

import asyncio
import json
import logging
import os
import re
import ssl
import traceback
from datetime import datetime
from typing import List, Dict, Optional

import aiofiles
import certifi
import httpx

from chromadb_utils import get_collection
from config import config
from models import ValidatedTelegramMetadata

TEMP_STORAGE_PATH = "temp_telegram_data"  # Папка для временного хранения данных Telegram
SAVE_INTERVAL_SECONDS = 60  # Интервал сохранения в секундах (10 минут)
OFFSET_FILE = "offset.txt"
TELEGRAM_BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
HUBSPOT_API_KEY = config.HUBSPOT_API_KEY

daily_conversations: Dict[str, Dict[str, Dict[str, Dict]]] = {}
deal_cache = {}  # Initialize the deal cache



def clean_text(text):
    """Removes all characters except letters and numbers from the text."""
    return re.sub(r'[^a-zA-Z0-9]', '', text)

class TelegramIntegration:
    def __init__(self):
        self.is_running = False
        self.periodic_save_task = None
        self.collector_task = None
        self.supervisor_task = None  # Добавляем задачу-супервизор

    async def get_telegram_updates(self, offset: int = 0) -> List[Dict]:
        """
        Получает обновления от Telegram API.
        """
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {"offset": offset, "timeout": 60}
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        try:
            async with httpx.AsyncClient(verify=ssl_context) as client:
                response = await client.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                if data.get("ok"):
                    return data["result"]
                else:
                    logging.warning(f"Ошибка при получении обновлений: {data}")
                    return []
        except httpx.HTTPStatusError as e:
            logging.warning(f"Ошибка HTTP: {e}")
            return []
        except httpx.RequestError as e:
            logging.error(f"Ошибка подключения к Telegram API: {e}")
            traceback.print_exc()
            return []
        except Exception as e:
            logging.error(f"Общая ошибка: {e}")
            traceback.print_exc()
            return []

    async def get_chat_details(self, chat_id: int) -> Dict:
        """
        Получает детали чата из Telegram API.
        """
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChat"
        params = {"chat_id": chat_id}
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        try:
            async with httpx.AsyncClient(verify=ssl_context) as client:
                response = await client.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                if data.get("ok"):
                    return data["result"]
                else:
                    logging.warning(f"Ошибка при получении информации о чате: {data}")
                    return {}
        except httpx.HTTPStatusError as e:
            logging.warning(f"Ошибка HTTP при получении информации о чате: {e}")
            return {}
        except httpx.RequestError as e:
            logging.error(f"Ошибка подключения к Telegram API (getChat): {e}")
            traceback.print_exc()
            return {}
        except Exception as e:
            logging.error(f"Общая ошибка при получении информации о чате: {e}")
            traceback.print_exc()
            return {}

    async def get_deal_from_hubspot(self, deal_id: str) -> Optional[Dict]:
        """Gets deal information from HubSpot API."""
        url = f"https://api.hubapi.com/crm/v3/objects/deals/{deal_id}?properties=dealname,company_ids,associations&associations=companies"
        headers = {"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                deal_data = response.json()
                return deal_data
        except httpx.HTTPStatusError as e:
            logging.warning(f"Ошибка HTTP при получении информации о сделке: {e}")
            return None
        except httpx.RequestError as e:
            logging.error(f"Ошибка подключения к HubSpot API (get deal): {e}")
            traceback.print_exc()
            return None
        except Exception as e:
            logging.error(f"Общая ошибка при получении информации о сделке: {e}")
            traceback.print_exc()
            return None

    async def get_company_from_hubspot(self, company_id: str) -> Optional[Dict]:
        """Gets company information from HubSpot API."""
        url = f"https://api.hubapi.com/crm/v3/objects/companies/{company_id}?properties=name"
        headers = {"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                company_data = response.json()
                return company_data
        except httpx.HTTPStatusError as e:
            logging.warning(f"Ошибка HTTP при получении информации о компании: {e}")
            return None
        except httpx.RequestError as e:
            logging.error(f"Ошибка подключения к HubSpot API (get company): {e}")
            traceback.print_exc()
            return None
        except Exception as e:
            logging.error(f"Общая ошибка при получении информации о компании: {e}")
            traceback.print_exc()
            return None


    async def save_telegram_messages_to_chromadb(self):
        """Saves structured Telegram messages to ChromaDB, filtering by date."""
        global daily_conversations

        for chat_title, date_data in list(daily_conversations.items()):
            for date_str, conversations in list(date_data.items()):
                try:
                    for conversation_id, conversation in list(conversations.items()):
                        label = f"telegram_sales"
                        base_document_id = conversation_id  # Use conversation_id as base document_id
                        document_id = base_document_id

                        combined_text = json.dumps(conversation, indent=2, ensure_ascii=False)  # Дампим весь словарь

                        try:
                            collection = get_collection(label)
                            # Check if document_id already exists
                            counter = 1
                            while True:
                                try:
                                    # Проверяем наличие документа с текущим document_id
                                    existing_results = collection.get(ids=[document_id], include=[])
                                    if existing_results and existing_results[
                                        'ids']:  # Если документ с таким ID существует
                                        counter += 1
                                        document_id = f"{base_document_id}_{counter}"  # Добавляем счетчик к ID
                                        continue  # Проверяем следующий ID
                                    else:
                                        break  # ID свободен, выходим из цикла
                                except Exception as check_error:
                                    logging.error(f"Ошибка при проверке ID {document_id}:{check_error}")
                                    traceback.print_exc()
                                    break  # Прерываем цикл, чтобы избежать бесконечного повторения

                            deal_id = None
                            # Создаем словарь для метаданных
                            if not conversation.get("deal_id"):
                                chat_id = conversation.get("chat_id")
                                if not chat_id:
                                    pattern = r"-?\d+"
                                    match = re.search(pattern, conversation_id)
                                    if match:
                                        chat_id = match.group()
                                chat_details = await self.get_chat_details(chat_id)
                                chat_title = chat_details.get("title",
                                                              f"ChatID_{chat_id}") if chat_details else f"ChatID_{conversation_id.split(":")[-2]}"
                                chat_description = chat_details.get("description", "") if chat_details else ""

                                # Extract deal ID from chat title or description
                                deal_id = None
                                match = re.search(r"(\d{11})", chat_title + chat_description)
                                if match:
                                    deal_id = match.group(1)
                                deal_data = await self.get_deal_from_hubspot(deal_id)
                                if deal_data:
                                    deal_title = deal_data["properties"].get("dealname")
                                    company_ids = list(
                                        set([i['id'] for i in deal_data['associations']['companies']['results']]))
                                    if company_ids and len(company_ids) > 0:
                                        company_id = company_ids[0]  # Assuming one company for deal
                                        company_data = await self.get_company_from_hubspot(company_id)
                                        if company_data:
                                            partner = company_data["properties"].get("name")

                                    # Store in cache
                                    deal_cache[deal_id] = {
                                        "deal_title": deal_title,
                                        "company_id": company_id,
                                        "partner": partner
                                    }

                            metadata_dict = {
                                "chat": chat_title,
                                "chat_id": conversation_id.split(":")[-2],
                                "origin_conversation_id": conversation_id,
                                "create_date": datetime.fromisoformat(date_str).timestamp(),
                                "author": conversation.get("author", {}).get("username"),
                                "deal_id": conversation.get("deal_id"),
                                "deal_title": conversation.get("deal_title"),
                                "company_id": conversation.get("company_id", ''),
                                "partner": conversation.get("partner", ''),
                                "partner_search": clean_text(conversation.get("partner", '')).lower(),
                                "chunk": True,  # Обязательное поле, значение по умолчанию
                                "category": "sales",  # Обязательное поле, значение по умолчанию
                                "country": ''  # Обязательное поле, значение по умолчанию
                            }
                            if deal_id:
                                metadata_dict.update(
                                    {
                                        "deal_id": deal_id,
                                        "deal_title": deal_title,
                                        "company_id": company_id,
                                        "partner": partner
                                    }
                                )
                            if not deal_id:
                                logging.warning(f"Deal ID not found for conversation {conversation_id}")
                                continue
                            # Валидируем метаданные с помощью Pydantic Model
                            validated_metadata = ValidatedTelegramMetadata(**metadata_dict)
                            collection.upsert(
                                documents=[combined_text],
                                metadatas=[json.loads(validated_metadata.model_dump_json())],
                                ids=[document_id]
                            )
                            logging.info(
                                f"Conversation {conversation_id} from {chat_title} on {date_str} saved to ChromaDB with ID: {document_id}")
                            # Mark the file as committed
                            filename = self.generate_filename(chat_title, date_str, conversation_id)
                            committed_filename = self.generate_committed_filename(chat_title, date_str, conversation_id)
                            if os.path.exists(os.path.join(TEMP_STORAGE_PATH, filename)):
                                os.rename(os.path.join(TEMP_STORAGE_PATH, filename),
                                          os.path.join(TEMP_STORAGE_PATH, committed_filename))

                            del conversations[conversation_id]  # Remove from memory


                        except Exception as e:
                            logging.error(f"Error saving conversation {conversation_id} to ChromaDB: {e}\n {conversation}")
                            traceback.print_exc()

                        if not conversations:
                            del date_data[date_str]
                except ValueError:
                    logging.error(f"Неверный формат даты: {date_str}. Пропускаем.")
                    continue

            if not date_data:
                del daily_conversations[chat_title]

        return "Сообщения сохранены в ChromaDB."

    def generate_filename(self, chat_title: str, date: str, conversation_id: str) -> str:
        """Generates a filename for temporary storage."""
        return f"telegram_{chat_title}_{date}_{conversation_id}.json"

    def generate_committed_filename(self, chat_title: str, date: str, conversation_id: str) -> str:
        """Generates a filename for committed storage."""
        return f"telegram_{chat_title}_{date}_{conversation_id}_commited.json"

    async def save_all_telegram_messages_to_files(self):
        """Saves all telegram messages to temporary files."""
        global daily_conversations

        # Create the temporary storage directory if it doesn't exist
        if not os.path.exists(TEMP_STORAGE_PATH):
            os.makedirs(TEMP_STORAGE_PATH)

        for chat_title, date_data in daily_conversations.items():
            for date, conversations in date_data.items():
                for conversation_id, conversation in conversations.items():
                    filename = self.generate_filename(chat_title, date, conversation_id)
                    filepath = os.path.join(TEMP_STORAGE_PATH, filename)
                    try:
                        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                            await f.write(
                                json.dumps(conversation, indent=2,
                                           ensure_ascii=False))  # Сохраняем словарь conversation
                        logging.info(
                            f"Conversation {conversation_id} from {chat_title} on {date} saved to file: {filename}")
                    except Exception as e:
                        logging.error(f"Error saving conversation {conversation_id} to file {filename}: {e}")
                        traceback.print_exc()

    async def sync_save_all_telegram_messages_to_files(self):
        """Saves all telegram messages to temporary files."""
        global daily_conversations

        # Create the temporary storage directory if it doesn't exist
        if not os.path.exists(TEMP_STORAGE_PATH):
            os.makedirs(TEMP_STORAGE_PATH)

        for chat_title, date_data in daily_conversations.items():
            for date, conversations in date_data.items():
                for conversation_id, conversation in conversations.items():
                    filename = self.generate_filename(chat_title, date, conversation_id)
                    filepath = os.path.join(TEMP_STORAGE_PATH, filename)
                    try:
                        json_string = json.dumps(conversation, indent=2, ensure_ascii=False)
                        # Проверка на пустую строку перед записью
                        if json_string:  # Если json_string не пустая
                            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                                await f.write(json_string)
                            logging.info(
                                f"Conversation {conversation_id} from {chat_title} on {date} saved to file: {filename}")
                        else:
                            logging.warning(f"Conversation {conversation_id} is empty.  Not saving to file.")
                    except Exception as e:
                        logging.error(f"Error saving conversation {conversation_id} to file {filename}: {e}")
                        traceback.print_exc()

    async def load_telegram_messages_from_files(self):
        """Loads telegram messages from temporary files."""
        global daily_conversations

        # Create the temporary storage directory if it doesn't exist
        if not os.path.exists(TEMP_STORAGE_PATH):
            os.makedirs(TEMP_STORAGE_PATH)

        # Iterate through files in the temporary storage directory
        for filename in os.listdir(TEMP_STORAGE_PATH):
            if filename.endswith(".json") and not filename.endswith("_commited.json"):
                filepath = os.path.join(TEMP_STORAGE_PATH, filename)
                try:
                    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                        conversation = json.loads(await f.read())

                    # Extract chat_title, date, and conversation_id from the filename
                    parts = filename.split("_")
                    chat_title = parts[1]
                    date = parts[2]
                    conversation_id = parts[3].split(".")[0]

                    # Load the conversation into the daily_conversations dictionary
                    if chat_title not in daily_conversations:
                        daily_conversations[chat_title] = {}
                    if date not in daily_conversations[chat_title]:
                        daily_conversations[chat_title][date] = {}
                    daily_conversations[chat_title][date][
                        conversation_id] = conversation  # Store the conversation directly

                    logging.info(f"Conversation {conversation_id} from file {filename} loaded into memory.")

                except Exception as e:
                    logging.error(f"Error loading conversation from file {filename}: {e}")
                    traceback.print_exc()

    async def load_offset(self):
        """Загружает offset из файла асинхронно."""
        try:
            async with aiofiles.open(OFFSET_FILE, "r") as f:
                content = await f.read()
                return int(content)
        except FileNotFoundError:
            return 0  # Начинаем с 0, если файл не существует
        except ValueError:
            print("Ошибка: Некорректный offset в файле. Начинаем с 0.")
            return 0

    async def save_offset(self, offset: int):
        """Сохраняет offset в файл асинхронно."""
        try:
            async with aiofiles.open(OFFSET_FILE, "w") as f:
                await f.write(str(offset))
        except Exception as e:
            print(f"Ошибка при сохранении offset в файл: {e}")

    async def telegram_message_collector(self):
        """Collects Telegram messages and stores them in structured format."""
        global daily_conversations
        offset = await self.load_offset()  # Загружаем offset при старте и в каждой итерации, на всякий случай
        logging.info(f"Загружен offset: {offset}")

        while self.is_running:
            try:
                updates = await self.get_telegram_updates(offset=offset)
                if updates:
                    for update in updates:
                        offset = update["update_id"] + 1
                        if "message" in update:
                            message = update["message"]
                            chat_id = message["chat"]["id"]
                            # Get chat details
                            chat_details = await self.get_chat_details(chat_id)
                            chat_title = chat_details.get("title",
                                                          f"ChatID_{chat_id}") if chat_details else f"ChatID_{chat_id}"
                            chat_description = chat_details.get("description", "") if chat_details else ""

                            # Extract deal ID from chat title or description
                            deal_id = None
                            match = re.search(r"(\d{11})", chat_title + chat_description)
                            if match:
                                deal_id = match.group(1)

                            date = datetime.fromtimestamp(message["date"]).strftime("%Y-%m-%d")

                            author_username = message["from"].get("username")
                            author_first_name = message["from"].get("first_name")

                            text = message.get("text", message.get("caption", None))
                            message_id = message.get("message_id")  # Extract the message id
                            reply_to_message = message.get("reply_to_message")  # Extract reply_to_message
                            conversation_message_start_id = message.get('message_thread_id',
                                                                        message_id)  # message_thread_id for forums, message_id otherwise
                            logging.info(
                                f"Получено сообщение из чата {chat_title} от {date}")  # Removed the message text
                            human_readable_date = datetime.fromtimestamp(message["date"]).strftime('%Y-%m-%d %H:%M:%S')

                            deal_title = None
                            company_id = None
                            partner = None

                            # Enrich message with deal and company information
                            if deal_id:
                                if deal_id in deal_cache:
                                    deal_data = deal_cache[deal_id]  # Get from cache
                                    deal_title = deal_data.get("deal_title")
                                    company_id = deal_data.get("company_id")
                                    partner = deal_data.get("company_title")
                                else:
                                    deal_data = await self.get_deal_from_hubspot(deal_id)
                                    if deal_data:
                                        deal_title = deal_data["properties"].get("dealname")
                                        company_ids = list(
                                            set([i['id'] for i in deal_data['associations']['companies']['results']]))
                                        if company_ids and len(company_ids) > 0:
                                            company_id = company_ids[0]  # Assuming one company for deal
                                            company_data = await self.get_company_from_hubspot(company_id)
                                            if company_data:
                                                partner = company_data["properties"].get("name")

                                        # Store in cache
                                        deal_cache[deal_id] = {
                                            "deal_title": deal_title,
                                            "company_id": company_id,
                                            "partner": partner
                                        }

                            if text:
                                message_data = {
                                    "message_id": message_id,
                                    "chat_id": chat_id,
                                    "date": human_readable_date,
                                    "text": text,
                                    "conversation_message_start_id": conversation_message_start_id,
                                    "author": {
                                        "username": author_username,
                                        "first_name": author_first_name
                                    },
                                }
                                if deal_id:
                                    message_data.update(
                                        {
                                            "deal_id": deal_id,
                                            "deal_title": deal_title,
                                            "company_id": company_id,
                                            "partner": partner
                                        }
                                    )

                                conversation_id = f"{chat_title}{chat_id}:{conversation_message_start_id}"

                                if chat_title not in daily_conversations:
                                    daily_conversations[chat_title] = {}
                                if date not in daily_conversations[chat_title]:
                                    daily_conversations[chat_title][date] = {}

                                if conversation_id not in daily_conversations[chat_title][date]:
                                    daily_conversations[chat_title][date][
                                        conversation_id] = message_data  # Сохраняем сообщение напрямую
                                else:
                                    # Handle replies to messages
                                    if reply_to_message:
                                        # Find the parent message and nest the reply.
                                        found_parent = False
                                        conversation = daily_conversations[chat_title][date][conversation_id]
                                        if self.find_and_nest_reply([conversation], reply_to_message,
                                                                    message_data):  # Обратите внимание: передаем conversation в списке
                                            found_parent = True

                                        if not found_parent:
                                            # Add as new if not found (rare case, but must be handled)
                                            logging.warning(
                                                f"Parent message not found for reply message_id: {message_id}, reply_to_message_id: {reply_to_message.get('message_id')}")

                    await self.save_offset(offset)  # Сохраняем offset после обработки пакета
                    logging.info(f"Сохранен offset: {offset}")
                else:
                    await asyncio.sleep(1)  # Избегаем частых запросов, если нет обновлений
            except asyncio.CancelledError:
                logging.info("Telegram message collector cancelled.")
                break
            except Exception as e:
                logging.error(f"Exception in telegram_message_collector: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)  # Wait before restarting

    def find_and_nest_reply(self, conversation: List[Dict], reply_to_message: Dict,
                            message_data: Dict) -> bool:
        """
        Recursively searches for the parent message and nests the reply.
        Returns True if parent found and nested, False otherwise.
        """
        msg = conversation[0]  # Берем единственный элемент из списка
        if msg.get("message_id") == reply_to_message.get("message_id"):
            # Found the parent message, nest the reply
            if "replies" not in msg:
                msg["replies"] = []
            msg["replies"].append(message_data)  # Nest message_data
            return True
        if "replies" in msg:
            # Оборачиваем replies в список, чтобы рекурсивно вызывать функцию
            if self.find_and_nest_reply(msg["replies"], reply_to_message, message_data):
                return True  # Parent found in nested replies
        return False

    async def periodic_save(self):
        """Периодически сохраняет данные."""
        while self.is_running:
            logging.info("Периодическое сохранение данных Telegram...")
            try:
                await self.sync_save_all_telegram_messages_to_files()
                await self.save_telegram_messages_to_chromadb()  # Сохраняем только старые сообщения
                logging.info("Периодическое сохранение Telegram успешно.")
                await asyncio.sleep(SAVE_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logging.info("Periodic save task cancelled.")
                break
            except Exception as e:
                logging.error(f"Ошибка при периодическом сохранении Telegram: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)  # Wait before restarting

    async def task_supervisor(self):
        """Supervises the collector and periodic save tasks, restarting them if they crash."""
        while self.is_running:
            try:
                if self.collector_task is None or self.collector_task.done():
                    if self.collector_task:  # Log reason for exit
                        try:
                            await self.collector_task  # Await to get exception if any
                        except Exception as e:
                            logging.error(f"Collector task exited with exception: {e}")
                            traceback.print_exc()
                    logging.info("Restarting telegram_message_collector...")
                    self.collector_task = asyncio.create_task(self.telegram_message_collector())

                if self.periodic_save_task is None or self.periodic_save_task.done():
                    if self.periodic_save_task:  # Log reason for exit
                        try:
                            await self.periodic_save_task  # Await to get exception if any
                        except Exception as e:
                            logging.error(f"Periodic save task exited with exception: {e}")
                            traceback.print_exc()
                    logging.info("Restarting periodic_save...")
                    self.periodic_save_task = asyncio.create_task(self.periodic_save())
                await asyncio.sleep(5)  # Check tasks every 5 seconds
            except asyncio.CancelledError:
                logging.info("Task supervisor cancelled.")
                break
            except Exception as e:
                logging.error(f"Exception in task_supervisor: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)

    async def stop(self):
        """Stops the telegram integration."""
        self.is_running = False

        if self.collector_task:
            self.collector_task.cancel()
        if self.periodic_save_task:
            self.periodic_save_task.cancel()
        if self.supervisor_task:
            self.supervisor_task.cancel()

        logging.info("Stopping telegram integration: saving data...")
        await self.sync_save_all_telegram_messages_to_files()
        await self.save_telegram_messages_to_chromadb()
        logging.info("Telegram data saved.")

    async def start(self):
        """Starts the telegram integration."""
        self.is_running = True

        # Сброс offset при старте
        try:
            updates = await self.get_telegram_updates()  # Получаем одно обновление без offset
            if updates:
                last_update_id = updates[-1]["update_id"]  # Берем ID последнего обновления
                await self.save_offset(last_update_id + 1)  # Сохраняем новый offset
                logging.info(f"Offset сброшен до {last_update_id + 1} при старте.")
            else:
                logging.info("Нет обновлений при старте. Оставляем offset как есть.")
        except Exception as e:
            logging.error(f"Ошибка при сбросе offset при старте: {e}")
            traceback.print_exc()

        await self.load_telegram_messages_from_files()
        self.collector_task = asyncio.create_task(self.telegram_message_collector())
        self.periodic_save_task = asyncio.create_task(self.periodic_save())  # Запускаем периодическое сохранение
        self.supervisor_task = asyncio.create_task(self.task_supervisor())  # Запускаем супервизор


telegram_integration = TelegramIntegration()
