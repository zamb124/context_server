import asyncio
import logging
import os
import re  # Импорт модуля для регулярных выражений
import traceback
from typing import Optional, Dict, Any, List

import httpx

from config import config  # Убедитесь, что config.py содержит HUBSPOT_API_KEY
import json


def clean_text(text: str) -> str:
    """Удаляет технические символы, такие как \n, \t, \r, HTML теги, а также символ >."""
    if text is None:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t\r]", " ", text).strip()
    if not text:
        text = ''
    text = text.replace(">", "")
    return text


note_map = {
    'hs_timestamp': 'create_date',
    'id': 'id',
    'hs_note_body': 'text',
}
email_map = {
    'hs_timestamp': 'create_date',
    'id': 'id',
    'hs_email_text': 'text',
    'hs_email_subject': 'subject',
    'hs_email_from_email': 'from',
    'hs_email_from_firstname': 'from_firstname',
    'hs_email_from_lastname': 'from_lastname',
    'hs_email_to_email': 'to',
    'hs_email_to_firstname': 'to_firstname',
    'hs_email_to_lastname': 'to_lastname',
}
deal_map = {
    'hs_timestamp': 'create_date_hs',
    'id': 'id',
    'amount': 'amount',
    'closedate': 'close_date',
    'dealname': 'deal_name',
    'pipeline': 'pipeline',
    'dealstage': 'deal_stage',
    'geography': 'geography',
    'createdate': 'create_date',
    'telegram_group': 'telegram_invite_link'
}

call_map = {
    'hs_timestamp': 'create_date',
    'id': 'id',
    'hs_call_body': 'body',
    'hs_call_direction': 'call_direction',
    'hs_call_duration': 'call_duration',
    'hs_call_recording_url': 'call_recording_url',
    'hs_call_title': 'call_title',
    'hs_call_from_number': 'from_number',
    'hs_call_to_number': 'to_number'
}
task_map = {
    'hs_timestamp': 'create_date',
    'id': 'id',
    'hs_task_body': 'text',
    'hs_task_subject': 'subject',
    'hs_task_type': 'type'
}
meeting_map = {
    'hs_timestamp': 'create_date',
    'id': 'id',
    'hs_meeting_title': 'title',
    'hs_meeting_body': 'body',
    'hs_meeting_location': 'location',
    'hs_meeting_start_time': 'start_time',
    'hs_meeting_end_time': 'end_time'
}
activities_maps = {
    'note': note_map,
    'email': email_map,
    'deal': deal_map,
    'call': call_map,
    'task': task_map,
    'meeting': meeting_map
}


class HubSpotDataExtractor:
    BASE_URL = "https://api.hubapi.com/crm/v3/objects"

    def __init__(self, access_token: str, output_dir: str = "hubspot_data"):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=httpx.Timeout(30.0, connect=10.0))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_retries = 3
        self.retry_delay = 5

    async def _fetch_data(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logging.error(f"Ошибка HTTP {e.response.status_code} для URL {url} с параметрами {params}: {e.response.text}")
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", self.retry_delay))
                    logging.warning(
                        f"Превышен лимит запросов. Повторная попытка через {retry_after} секунд (попытка {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(retry_after)
                elif 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    logging.error(f"Непоправимая ошибка клиента: {e}") # Не повторяем для большинства ошибок 4xx
                    return None
                else: # Серверные ошибки 5xx или другие HTTPStatusError, которые можно повторить
                    logging.warning(
                        f"Ошибка сервера или сети ({e.response.status_code}). Повторная попытка через {self.retry_delay} секунд (попытка {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
            except httpx.RequestError as e: # Ошибки сети, таймауты и т.д.
                logging.error(f"Ошибка запроса для URL {url} с параметрами {params}: {type(e).__name__} - {e}")
                if attempt < self.max_retries - 1:
                    logging.warning(
                        f"Повторная попытка через {self.retry_delay} секунд (попытка {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logging.error(f"Не удалось выполнить запрос к {url} после {self.max_retries} попыток.")
                    return None
            except json.JSONDecodeError as e: # Ошибка парсинга JSON
                logging.error(f"Ошибка декодирования JSON ответа с URL {url}: {e}")
                # Логируем часть ответа для диагностики, если он есть
                raw_response_text = response.text if 'response' in locals() and hasattr(response, 'text') else 'Ответ не получен или недоступен'
                logging.error(f"Текст ответа (первые 500 символов): {raw_response_text[:500]}")
                return None # Непоправимая ошибка
            except Exception as e: # Другие непредвиденные ошибки
                logging.error(f"Произошла непредвиденная ошибка при запросе к {url}: {e}")
                logging.error(traceback.format_exc())
                return None
        logging.error(f"Не удалось получить данные из {url} после {self.max_retries} попыток (все попытки исчерпаны).")
        return None

    async def fetch_all(self, object_type: str, properties: Optional[List[str]] = None,
                        associations: Optional[List[str]] = None, object_id: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{object_type}"
        if object_id:
            url = f"{url}/{object_id}"
        params = {}
        if properties:
            params["properties"] = ",".join(list(set(properties)))
        if associations:
            params["associations"] = ",".join(list(set(associations)))
        results = []
        if object_id:
            data = await self._fetch_data(url, params=params)
            if data:
                results.append(data)
            return results
        params["limit"] = 100
        after = None
        while True:
            current_params = params.copy()
            if after:
                current_params["after"] = after
            data = await self._fetch_data(url, params=current_params)
            if data is None:
                break
            results.extend(data.get("results", []))
            paging_next = data.get("paging", {}).get("next")
            if paging_next and paging_next.get("after"):
                after = paging_next["after"]
            else:
                break
        return results

    async def get_company_data(self, company_id: str) -> Dict[str, Any]:
        company_properties = ["name", "domain", "industry", "phone", "website", "description",
                              "geography", "country", "city", "hs_timestamp", "createdate"]
        unique_company_properties = list(set(company_properties))
        url = f"{self.BASE_URL}/companies/{company_id}"
        params = {"properties": ",".join(unique_company_properties),
                  "associations": "contacts,notes,emails,deals,calls,tasks,meetings"}
        data = await self._fetch_data(url, params=params)
        if not data:
            logging.warning(f"Компания с ID {company_id} не найдена или не удалось получить данные.")
            return {}
        return {
            "id": data.get("id"),
            "properties": data.get("properties", {}),
            "associations": data.get("associations", {})
        }

    async def get_associated_contacts(self, contacts_ids: List[str]) -> List[Dict[str, Any]]:
        contacts = []
        contact_properties = ["firstname", "lastname", "email", "phone", "jobtitle",
                              "hs_timestamp", "createdate"]
        unique_contact_properties = list(set(contact_properties))
        # Уникализация ID контактов перед запросом
        unique_contacts_ids_to_fetch = list(set(cid for cid in contacts_ids if cid))

        for contact_id in unique_contacts_ids_to_fetch:
            # if not contact_id: # Эта проверка уже сделана при создании unique_contacts_ids_to_fetch
            #     logging.warning("Обнаружен пустой ID контакта, пропуск.")
            #     continue
            try:
                url = f"{self.BASE_URL}/contacts/{contact_id}"
                params = {"properties": ",".join(unique_contact_properties)}
                data = await self._fetch_data(url, params=params)
                if data:
                    contacts.append({
                        "id": data.get("id"),
                        "properties": data.get("properties", {})
                    })
                else:
                    logging.warning(f"Контакт с ID {contact_id} не найден или не удалось получить данные.")
            except Exception as e:
                logging.error(f"Ошибка при получении контакта с ID {contact_id}: {e}")
                logging.error(traceback.format_exc())
        return contacts

    async def get_associated_activities(self, company_data: dict) -> List[Dict[str, Any]]:
        associated_activities = []
        associations_data = company_data.get("associations", {})
        if not associations_data:
            logging.debug(f"Нет ассоциаций для компании ID {company_data.get('id')}")
            return []

        activity_types_map_config = {
            "notes": ("notes", list(set(note_map.keys()))),
            "emails": ("emails", list(set(email_map.keys()))),
            "deals": ("deals", list(set(deal_map.keys()))),
            "calls": ("calls", list(set(call_map.keys()))),
            "tasks": ("tasks", list(set(task_map.keys()))),
            "meetings": ("meetings", list(set(meeting_map.keys()))),
        }

        async def get_activity_by_id(object_type_plural: str, object_id: str, props_to_fetch: List[str]) -> Optional[Dict[str, Any]]:
            url = f"{self.BASE_URL}/{object_type_plural}/{object_id}"
            props_to_fetch_cleaned = [p for p in props_to_fetch if p.lower() != 'id']
            params = {"properties": ",".join(list(set(props_to_fetch_cleaned)))}
            data = await self._fetch_data(url, params=params)
            return data

        for assoc_key, (activity_api_name, properties_list) in activity_types_map_config.items():
            activity_refs = associations_data.get(assoc_key, {}).get("results", [])

            # <--- НАЧАЛО ИЗМЕНЕНИЯ: Уникализация ID активностей перед запросом деталей
            unique_activity_ids = list(set(
                item.get("id") for item in activity_refs if item and item.get("id")
            ))
            # <--- КОНЕЦ ИЗМЕНЕНИЯ

            for activity_id in unique_activity_ids: # Итерация по уникальным ID
                activity_data = await get_activity_by_id(activity_api_name, activity_id, properties_list)
                if activity_data:
                    singular_type = activity_api_name.rstrip('s')
                    associated_activities.append({
                        "type": singular_type,
                        "id": activity_data.get("id"),
                        "properties": activity_data.get("properties", {})
                    })
        return associated_activities

    async def process_attachments(self, company_data: Dict[str, Any]) -> List[str]:
        logging.info("Функция process_attachments требует детальной реализации согласно HubSpot API v3 для работы с файлами.")
        return []

    def build_company_json(self, company_data: Dict[str, Any], contacts: List[Dict[str, Any]],
                           activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        company_props = company_data.get("properties", {})
        current_company_id = company_data.get("id")
        company_json = {
            "city": company_props.get("city", ""),
            "create_date": company_props.get('createdate', company_props.get('hs_timestamp', "")),
            "country": company_props.get("country", ""),
            "name": company_props.get("name", "No Name Provided"),
            "industry": company_props.get("industry", ""),
            "id": current_company_id,
            "domain": company_props.get("domain", ""),
            "description": clean_text(company_props.get("description", "")),
            "type": "company",
            "contacts": [], "notes": [], "emails": [], "calls": [], "tasks": [], "meetings": [], "deals": []
        }
        for contact in contacts:
            contact_props = contact.get("properties", {})
            contact_data = {
                "name": f"{contact_props.get('firstname', '')} {contact_props.get('lastname', '')}".strip(),
                "create_date": contact_props.get('createdate', contact_props.get('hs_timestamp', "")),
                "job_title": contact_props.get('jobtitle', ''), "email": contact_props.get('email', ''),
                "phone": contact_props.get('phone', ''), "type": "contact", "id": contact.get('id')
            }
            company_json["contacts"].append(contact_data)

        # Поскольку get_associated_activities теперь возвращает уникальные активности,
        # дополнительная проверка на уникальность здесь не требуется.
        for activity in activities:
            activity_type = activity.get("type")
            if not activity_type:
                logging.warning(f"Активность без типа пропущена: {activity}")
                continue
            properties = activity.get("properties", {})
            activities_map_for_type = activities_maps.get(activity_type)
            if not activities_map_for_type:
                logging.warning(f"Не найдена карта свойств для типа активности: {activity_type}")
                continue
            new_activity_obj = {}
            for hubspot_prop, json_key in activities_map_for_type.items():
                raw_value = properties.get(hubspot_prop)
                if json_key == 'telegram_invite_link' or hubspot_prop == 'id' or 'url' in json_key.lower():
                    new_activity_obj[json_key] = str(raw_value) if raw_value is not None else ""
                else:
                    new_activity_obj[json_key] = clean_text(str(raw_value) if raw_value is not None else "")
            new_activity_obj["type"] = activity_type
            if 'id' not in new_activity_obj or not new_activity_obj['id']:
                new_activity_obj['id'] = activity.get('id')
            if activity_type == 'deal':
                new_activity_obj['associated_company_id'] = current_company_id
            target_key = activity_type + "s"
            if target_key not in company_json:
                if activity_type in company_json: target_key = activity_type
                else:
                    logging.warning(f"Не удалось найти подходящий ключ в company_json для {target_key} или {activity_type}. Пропускаем.")
                    continue
            company_json[target_key].append(new_activity_obj)
        return company_json

    async def process_companies_via_deals(self, specific_deal_id: Optional[str] = None):
        logging.info("Начинаем процесс извлечения компаний через сделки (только со сделками, имеющими telegram_group).")
        telegram_group_hubspot_field_name = 'telegram_group'
        deal_properties_to_fetch = [prop_key for prop_key in deal_map.keys() if prop_key.lower() != 'id']
        if telegram_group_hubspot_field_name not in deal_properties_to_fetch and telegram_group_hubspot_field_name in deal_map:
            deal_properties_to_fetch.append(telegram_group_hubspot_field_name)
        unique_deal_properties = list(set(deal_properties_to_fetch))
        initial_deals_data = []
        if specific_deal_id:
            logging.info(f"Запрос данных для конкретной сделки с ID: {specific_deal_id}")
            initial_deals_data = await self.fetch_all(
                object_type="deals", properties=unique_deal_properties,
                associations=["companies"], object_id=specific_deal_id)
            if not initial_deals_data:
                logging.warning(f"Сделка с ID {specific_deal_id} не найдена или не удалось получить данные.")
                return
            deal_props = initial_deals_data[0].get("properties", {})
            if not deal_props.get(telegram_group_hubspot_field_name):
                logging.info(f"Сделка {specific_deal_id} не имеет '{telegram_group_hubspot_field_name}'. Обработка прекращена.")
                return
            deals_to_process = initial_deals_data
        else:
            logging.info("Запрос всех сделок для фильтрации по telegram_group...")
            initial_deals_data = await self.fetch_all(
                object_type="deals", properties=unique_deal_properties, associations=["companies"])
            if not initial_deals_data:
                logging.warning("Сделки в HubSpot не найдены.")
                return
            deals_to_process = [
                deal_item for deal_item in initial_deals_data
                if deal_item.get("properties", {}).get(telegram_group_hubspot_field_name)
            ]
            if not deals_to_process:
                logging.info(f"Не найдено сделок с '{telegram_group_hubspot_field_name}'.")
                return

        logging.info(f"Найдено {len(deals_to_process)} сделок с '{telegram_group_hubspot_field_name}'. Обработка компаний...")
        processed_company_ids = set()
        for deal_item in deals_to_process:
            deal_id_for_log = deal_item.get("id", "Неизвестный ID сделки")
            company_id = None
            deal_associations = deal_item.get("associations")
            if deal_associations and deal_associations.get("companies") and deal_associations["companies"].get("results"):
                company_assoc_results = deal_associations["companies"]["results"]
                if company_assoc_results: company_id = company_assoc_results[0].get("id")
            if not company_id:
                logging.warning(f"Сделка {deal_id_for_log} (с telegram_group) не имеет связанной компании. Пропускаем.")
                continue
            if company_id in processed_company_ids:
                logging.info(f"Компания ID {company_id} (сделка {deal_id_for_log}) уже обработана. Пропускаем.")
                continue
            logging.info(f"Обработка компании ID {company_id} (сделка {deal_id_for_log}).")
            company_data = await self.get_company_data(company_id)
            if not company_data or not company_data.get("id"):
                logging.warning(f"Данные для компании ID {company_id} (сделка {deal_id_for_log}) не найдены. Пропускаем.")
                continue
            company_name_from_props = company_data.get('properties', {}).get('name', f'Без_названия_{company_id}')
            company_name_safe = re.sub(r'[<>:"/\\|?*]', '_', company_name_from_props)
            output_file = os.path.join(self.output_dir, f"{company_name_safe}.json")
            if os.path.exists(output_file):
                logging.info(f"Файл для '{company_name_safe}' (ID: {company_id}) уже существует. Пропускаем.")
                processed_company_ids.add(company_id)
                continue
            contact_associations = company_data.get("associations", {}).get("contacts", {}).get("results", [])
            contacts_ids = [item.get('id') for item in contact_associations if item and item.get('id')]
            contacts = await self.get_associated_contacts(contacts_ids)
            activities = await self.get_associated_activities(company_data) # Теперь возвращает уникальные активности
            company_json_output = self.build_company_json(company_data, contacts, activities)
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(company_json_output, f, indent=4, ensure_ascii=False)
                logging.info(f"Данные '{company_name_safe}' (ID: {company_id}) сохранены: {output_file}")
            except IOError as e:
                logging.error(f"Ошибка записи {output_file} для '{company_name_safe}': {e}")
            except Exception as e:
                logging.error(f"Ошибка сохранения файла для '{company_name_safe}': {e}")
                logging.error(traceback.format_exc())
            processed_company_ids.add(company_id)
        logging.info("Обработка компаний (через сделки с telegram_group) завершена.")

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()

if __name__ == "__main__":
    async def main():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        access_token = None
        if hasattr(config, 'HUBSPOT_API_KEY') and config.HUBSPOT_API_KEY:
            access_token = config.HUBSPOT_API_KEY
        else:
            logging.error("HUBSPOT_API_KEY не найден в config.py или не задан.")
            return
        output_directory = "hubspot_data" # Изменил версию
        extractor = HubSpotDataExtractor(access_token=access_token, output_dir=output_directory)
        try:
            # deal_id_to_process = 'ID_ВАШЕЙ_СДЕЛКИ_ДЛЯ_ТЕСТА'
            # await extractor.process_companies_via_deals(deal_id_to_process)
            await extractor.process_companies_via_deals()
        except Exception as e:
            logging.error(f"Во время основной обработки произошла критическая ошибка: {e}")
            logging.error(traceback.format_exc())
        finally:
            await extractor.close()
            logging.info("Завершение работы скрипта.")
    asyncio.run(main())