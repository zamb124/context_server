import asyncio
import logging
import os
import re  # Импорт модуля для регулярных выражений
from typing import Optional, Dict, Any, List

import httpx

from config import config  # Убедитесь, что config.py содержит HUBSPOT_API_KEY


def clean_text(text: str) -> str:
    """Удаляет технические символы, такие как \n, \t, \r, HTML теги, а также символ >."""
    text = re.sub(r"[\n\t\r]", " ", text).strip()
    text = re.sub(r"<[^>]+>", "", text)  # Удаление HTML тегов
    text = text.replace(">", "")  # Удаление символа >
    return text


note_map = {
    'hs_note_body': 'NOTE BODY'
}
email_map = {
    'hs_email_text': 'EMAIL TEXT',
    'hs_email_subject': 'EMAIL SUBJECT',
    'hs_email_from_email': 'EMAIL FROM',
    'hs_email_from_firstname': 'EMAIL FROM FIRST NAME',
    'hs_email_from_lastname': 'EMAIL FROM LAST NAME',
    'hs_email_to_email': 'EMAIL TO',
    'hs_email_to_firstname': 'EMAIL TO FIRST NAME',
    'hs_email_to_lastname': 'EMAIL TO LAST NAME',
}
deal_map = {
    'amount': 'AMOUNT',
    'closedate': 'DEAL CLOSE DATE',
    'dealname': 'DEAL NAME',
    'pipeline': 'DEAL PIPELINE',
    'dealstage': 'DEAL STAGE',
    'hubspot_owner_id': 'DEAL OWNER ID',
}

call_map = {
    'hs_timestamp': 'CALL DATETIME',
    'hs_call_body': 'CALL BODY',
    'hs_call_direction': 'CALL DIRECTION',
    'hs_call_duration': 'CALL DURATION',
    'hs_call_recording_url': 'CALL RECORDING URL',
    'hs_call_title': 'CALL TITLE',
    'hs_activity_type': 'TYPE',
    'hs_call_from_number': 'CALL FROM NUMBER',
    'hs_call_to_number': 'CALL TO NUMBER'
}
task_map = {
    'hs_timestamp': 'TASK DATETIME',
    'hs_task_body': 'TASK BODY',
    'hs_task_subject': 'TASK SUBJECT',
    'hs_task_type': 'TYPE'
}
meeting_map = {
    'hs_timestamp': 'MEETING DATETIME',
    'hs_meeting_title': 'MEETING TITLE',
    'hs_meeting_body': 'MEETING BODY',
    'hs_meeting_location': 'MEETING LOCATION',
    'hs_meeting_start_time': 'MEETING START',
    'hs_meeting_end_time': 'MEETING END'
}


class HubSpotDataExtractor:  # Переименованный класс для ясности и цели
    BASE_URL = "https://api.hubapi.com/crm/v3/objects"

    def __init__(self, access_token: str, output_dir: str = "hubspot_data"):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30)  # Увеличенное время ожидания
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Создать выходной каталог
        self.max_retries = 3  # Максимальное количество повторных попыток для вызовов API
        self.retry_delay = 5  # Задержка в секундах перед повторной попытки

    async def _fetch_data(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Вспомогательная функция для получения данных с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logging.error(f"Ошибка HTTP {e.response.status_code} для URL {url}: {e.response.text}")
                if e.response.status_code == 429:  # Превышение лимита запросов
                    logging.warning(
                        f"Превышен лимит запросов. Повторная попытка через {self.retry_delay} секунд (попытка {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logging.error(f"Непоправимая ошибка: {e}")
                    return None  # Непоправимая ошибка
            except httpx.RequestError as e:
                logging.error(f"Ошибка запроса для URL {url}: {e}")
                logging.warning(
                    f"Повторная попытка через {self.retry_delay} секунд (попытка {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logging.error(f"Произошла непредвиденная ошибка: {e}")
                return None  # Непоправимая ошибка

        logging.error(f"Не удалось получить данные из {url} после {self.max_retries} попыток.")
        return None

    async def fetch_all(self, object_type: str, properties: Optional[List[str]] = None,
                        associations: Optional[List[str]] = None, company_id: Optional[str] = None) -> List[
        Dict[str, Any]]:
        """
        Извлекает все объекты определенного типа из HubSpot, обрабатывая пагинацию и ограничение скорости.
        Теперь также поддерживает извлечение связанных объектов.
        Добавлена возможность фильтрации по ID компании в коде.
        """
        url = f"{self.BASE_URL}/{object_type}"
        params = {"limit": 100}
        if properties:
            params["properties"] = ",".join(properties)
        if associations:
            params["associations"] = ",".join(associations)

        results = []
        after = None
        while True:
            if after:
                params["after"] = after

            data = await self._fetch_data(url, params=params)
            if data is None:  # Выйти, если получение полностью не удалось
                break

            # Если указан ID компании, фильтруем результаты
            if company_id:
                filtered_results = []
                for item in data.get("results", []):
                    # Проверяем, связана ли сущность с указанной компанией
                    if item.get("associations") and item.get("associations").get("companies") and any(
                            result.get("id") == company_id for result in
                            item.get("associations").get("companies").get("results", [])):
                        filtered_results.append(item)
                results.extend(filtered_results)
            else:
                results.extend(data.get("results", []))

            after = data.get("paging", {}).get("next", {}).get("after")
            if not after:
                break
        return results

    async def get_company_data(self, company_id: str) -> Dict[str, Any]:
        """
        Извлекает исчерпывающие данные для одной компании, включая связанные контакты и действия.
        """
        company_properties = ["name", "domain", "industry", "phone", "website", "description", "contacts",
                              "activites", "deals", "geography"]  # Добавлено описание
        # Изменяем вызов fetch_all, чтобы вернуть только одну компанию по ID
        url = f"{self.BASE_URL}/companies/{company_id}"
        params = {"properties": ",".join(company_properties),
                  "associations": "contacts,notes,emails,deals,calls,tasks,meetings"}
        data = await self._fetch_data(url, params=params)
        if not data:
            logging.warning(f"Компания с ID {company_id} не найдена")
            return {}

        company_data = {
            "id": data["id"],
            "properties": data["properties"],
            "associations": data.get("associations", {})
        }
        return company_data

    async def get_associated_contacts(self, contacts_ids: List[str]) -> List[Dict[str, Any]]:
        """Получает информацию о конкретных контактах по их ID."""
        contacts = []
        for contact_id in contacts_ids:
            try:
                url = f"{self.BASE_URL}/contacts/{contact_id}"
                params = {"properties": "firstname,lastname,email,phone,jobtitle"}
                data = await self._fetch_data(url, params=params)

                if data:
                    contacts.append({
                        "id": data["id"],
                        "properties": data["properties"]
                    })
                else:
                    logging.warning(f"Контакт с ID {contact_id} не найден.")
            except Exception as e:
                logging.error(f"Ошибка при получении контакта с ID {contact_id}: {e}")
        return contacts

    async def get_associated_activities(self, company_data: dict) -> List[Dict[str, Any]]:
        """
        Получает связанные действия (заметки, электронные письма, звонки, задачи, встречи) для компании,
        запрашивая каждый объект по ID.
        """
        associated_activities = []

        # Теперь получаем IDs связанных заметок (если они есть в company_data - adapt to your actual data!)
        note_ids = company_data.get("associations", {}).get("notes", {}).get("results", [])
        email_ids = company_data.get("associations", {}).get("emails", {}).get("results", [])
        call_ids = company_data.get("associations", {}).get("calls", {}).get("results", [])
        task_ids = company_data.get("associations", {}).get("tasks", {}).get("results", [])
        meeting_ids = company_data.get("associations", {}).get("meetings", {}).get("results", [])
        deals_ids = company_data.get("associations", {}).get("deals", {}).get("results", [])

        # Функция для получения информации об одном объекте по ID
        async def get_activity_by_id(object_type: str, object_id: str, params: dict) -> Optional[Dict[str, Any]]:
            url = f"{self.BASE_URL}/{object_type}/{object_id}"
            data = await self._fetch_data(url, params=params)
            return data

        # Получаем информацию о каждой заметке
        for note_id in note_ids:
            note = await get_activity_by_id(
                "notes", note_id.get("id"),
                params={"properties": "hs_note_body"}
            )
            if note:
                associated_activities.append({
                    "type": "note",
                    "id": note["id"],
                    "properties": note["properties"]
                })

        # Получаем информацию о каждом email
        for email_id in email_ids:
            email = await get_activity_by_id(
                "emails", email_id.get("id"),
                params={
                    "properties": ",".join([i for i in email_map.keys()])
                }
            )
            if email:
                associated_activities.append({
                    "type": "email",
                    "id": email["id"],
                    "properties": email["properties"]
                })
        # Получаем информацию о каждом deal
        for deal_id in deals_ids:
            deal = await get_activity_by_id(
                "deals", deal_id.get("id"),
                params={
                    "properties": ",".join([i for i in deal_map.keys()])
                }
            )
            if deal:
                associated_activities.append({
                    "type": "deal",
                    "id": deal["id"],
                    "properties": deal["properties"]
                })

        # Получаем информацию о каждом звонке
        for call_id in call_ids:

            call = await get_activity_by_id(
                "calls", call_id.get("id"),
                params={
                    "properties": ",".join([i for i in call_map.keys()])
                }
            )
            if call:
                associated_activities.append({
                    "type": "call",
                    "id": call["id"],
                    "properties": call["properties"]
                })

        # Получаем информацию о каждой задаче
        for task_id in task_ids:

            task = await get_activity_by_id(
                "tasks", task_id.get("id"),
                params={
                    "properties": ",".join([i for i in task_map.keys()])
                }
            )
            if task:
                associated_activities.append({
                    "type": "task",
                    "id": task["id"],
                    "properties": task["properties"]
                })

        # Получаем информацию о каждой встрече
        for meeting_id in meeting_ids:

            meeting = await get_activity_by_id(
                "meetings", meeting_id.get("id"),
                params={
                    "properties": ",".join([i for i in meeting_map.keys()])
                }
            )
            if meeting:
                associated_activities.append({
                    "type": "meeting",
                    "id": meeting["id"],
                    "properties": meeting["properties"]
                })

        return associated_activities

    async def process_attachments(self, company_data: Dict[str, Any]) -> List[str]:
        """
        Извлекает и обрабатывает вложения (документы), связанные с компанией,
        подразумевая, что они хранятся как вложения в email.
        Адаптируйте этот код к вашей структуре данных!
        """
        document_texts = []
        email_ids = company_data.get("associations", {}).get("emails", {}).get("results", [])

        # Функция для получения информации об email по ID
        async def get_email_with_attachments(email_id: str) -> Optional[Dict[str, Any]]:
            url = f"{self.BASE_URL}/emails/{email_id}"
            params = {"properties": "hs_attachment_names,hs_attachment_urls"}  # Пример. Adapt!
            data = await self._fetch_data(url, params=params)
            return data

        for email_id in email_ids:
            email = await get_email_with_attachments(email_id.get("id"))
            if email:
                attachment_names = email["properties"].get("hs_attachment_names", "").split(";")  # Пример!
                attachment_urls = email["properties"].get("hs_attachment_urls", "").split(";")  # Пример!

                for i, attachment_url in enumerate(attachment_urls):
                    attachment_name = attachment_names[i] if i < len(attachment_names) else "Unknown"
                    try:
                        # Загрузка и обработка документа
                        if attachment_url:  # Проверяем, что URL не пустой
                            async with self.client.get(attachment_url) as response:
                                response.raise_for_status()
                                file_content = await response.read()
                                file_path = os.path.join(self.output_dir, attachment_name)
                                with open(file_path, "wb") as f:
                                    f.write(file_content)
                                # elements = partition(filename=file_path) # убрал вызов несуществующей функции
                                # text = "\n".join([element.text for element in elements])
                                # document_texts.append(f"Документ: {attachment_name}\n{text}")
                                # При желании удалите временный файл
                                os.remove(file_path)
                    except Exception as e:
                        logging.error(
                            f"Ошибка обработки документа {attachment_name} из {attachment_url}: {e}")
        return document_texts



    def build_company_text(self, company_data: Dict[str, Any], contacts: List[Dict[str, Any]],
                           activities: List[Dict[str, Any]]) -> str:
        """
        Создает единый текстовый блок из данных компании, контактов и действий,
        используя разделители для логической структуры.
        """

        text = f"COMPANY: {company_data['properties'].get('name', 'Без названия')}\n"  # Основная информация о компании
        text += f"ID: {company_data['id']}\n"
        text += f"DOMAIN: {company_data['properties'].get('domain', 'Нет домена')}\n"
        text += clean_text(f"DESCRIPTION: {company_data['properties'].get('description', 'Нет описания')}\n")
        text += "    \n\n"  # Разделитель между компанией и контактами

        text += "CONTACTS:\n"
        for contact in contacts:
            text += f"  CONTACT: {contact['properties'].get('firstname', 'Нет имени')} {contact['properties'].get('lastname', 'Нет фамилии')}\n"
            text += f"  EMAIL: {contact['properties'].get('jobtitle', 'Нет email')}\n"
            text += f"  EMAIL: {contact['properties'].get('email', 'Нет email')}\n"
            text += f"  PHONE: {contact['properties'].get('phone', 'Нет телефона')}\n"
            text += "   \n"  # Разделитель между контактами
        text += "    \n\n"  # Разделитель между контактами и активностями

        text += "ACTIVITIES:\n"
        for activity in activities:
            text += f"  TYPE: {activity['type']}\n"
            text += f"  ID: {activity['id']}\n"
            # Добавляем свойства активности, адаптируясь к их типу
            for k, v in eval(f"{activity['type']}_map").items():
                text += "   "  # разделитель между полями активности
                text += clean_text(f"  {v}: {activity['properties'].get(k, 'Нет данных')}")
                text += "   \n"  # разделитель между полями активности
            text += "   \n"  # разделитель между активностями
        text += "    \n"  # Разделитель между компанией и другими компаниями
        return text

    async def process_all_companies(self, company_id: Optional[str] = None):
        """Извлекает все компании и создает отдельные текстовые файлы для каждой компании."""

        if company_id:
            companies = [{"id": company_id}]  # Создаем список только с одним ID компании
        else:
            companies = await self.fetch_all("companies", properties=["name"])
            if not companies:
                logging.warning("Компании в HubSpot не найдены.")
                return

        logging.info(f"Найдено {len(companies)} компаний. Начинаем обработку...")

        for company in companies:
            company_id = company["id"]
            company_data = await self.get_company_data(company_id)

            if not company_data:
                logging.warning(f"Данные для компании с ID {company_id} не найдены. Пропускаем.")
                continue

            # Извлекаем контакты и действия
            contacts_ids = [i['id'] for i in company_data.get("associations", {}).get("contacts", {}).get("results", [])]
            contacts = await self.get_associated_contacts(contacts_ids)
            activities = await self.get_associated_activities(company_data)

            # Создаем текстовый блок для компании
            company_text = self.build_company_text(company_data, contacts, activities)

            # Сохраняем данные в отдельный текстовый файл для каждой компании
            company_name = company_data['properties'].get('name', 'Без названия').replace('/', '_').replace('\\', '_')
            output_file = os.path.join(self.output_dir, f"{company_name}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(company_text)

            logging.info(f"Данные компании {company_name} ({company_id}) успешно сохранены в файл: {output_file}")

        logging.info("Обработка всех компаний завершена.")

    async def close(self):
        await self.client.aclose()


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        access_token = config.HUBSPOT_API_KEY  # Замените своим фактическим токеном доступа.
        output_directory = "hubspot_company_data"  # Каталог для хранения файлов компаний
        extractor = HubSpotDataExtractor(access_token=access_token, output_dir=output_directory)
        try:
            # Раскомментируйте и укажите ID компании, чтобы обработать только ее
            company_id_to_process = '18983060297'
            await extractor.process_all_companies(company_id_to_process)
        except Exception as e:
            logging.error(f"Во время обработки произошла ошибка: {e}")
        finally:
            await extractor.close()


    asyncio.run(main())
