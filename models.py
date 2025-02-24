from enum import Enum
from typing import Optional, Any, Dict, List

import pycountry
from pydantic import BaseModel, Field, field_validator


class Category(Enum):
    SALES = 'sales'
    OPS = 'ops'
    PRODUCT = 'product'


class DocumentType(Enum):
    TELEGRAMM_MESSAGE = 'telegramm_message'
    OPS_DOCUMENT = 'ops_document'
    OPENAPI_SPEC = 'openapi_spec'
    COMPANY = 'company'
    EMAIL = 'email'
    OTHER = 'other'
    CALL = 'call'
    MEETING = 'meeting'
    TASK = 'task'
    NOTE = 'note'
    WIKI = 'wiki'
    STARTREK_TICKET = 'startrek_ticket'
    CONTACT = 'contact'


class BaseMetadata(BaseModel):
    type: DocumentType
    author: str
    partner: str = ''  # default = False
    chunk: bool = True  # default = True
    category: Category = Field(description="Категория к какому разделу относится например это данные sales команды ")
    country: str = ''  # default = False

    @field_validator('country')
    def validate_country(cls, v):
        if not v:
            return v
        try:
            country = pycountry.countries.lookup(v)
            return country.alpha_2  # Возвращаем ISO 3166-1 alpha-2 код страны
        except LookupError:
            return v

    class Config:
        extra = 'allow'


class ValidatedTelegramMetadata(BaseMetadata):
    chat: str
    chat_id: str
    origin_conversation_id: str
    date: str
    deal_id: Optional[str]
    deal_title: Optional[str]
    company_id: Optional[str]
    type: DocumentType = DocumentType.TELEGRAMM_MESSAGE
    category: Category = Category.SALES


class ValidLabels(str, Enum):
    hubspot = "hubspot"
    telegram = "telegram_sales"
    wiki = "wiki"
    startrek = "startrek"


# --- Модели данных ---
class DocumentBase(BaseMetadata):
    id: str = Field(..., description="Уникальный идентификатор документа")


class Query(BaseModel):
    text: str = Field(..., description="Вопрос или запрос данных", min_length=1)
    labels: List[ValidLabels] = Field(
        description="Указать доступные коллекции где производить поиск")  # Обязательный атрибут labels
    n_results: int = Field(default=5, ge=1, le=300, description="Количество результатов для возврата")
    where: Optional[Dict[str, Any]] = Field(default=None, description="Фильтры по метаданным")


class ContextResponse(BaseModel):
    results: List[Dict]


class ForceSaveResponse(BaseModel):
    message: str
