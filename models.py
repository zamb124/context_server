from enum import Enum
from typing import Optional, Any, Dict, List

import pycountry
from pydantic import BaseModel, Field, field_validator


# =============================================================================
#                         Модели данных
# =============================================================================

class Category(Enum):
    SALES = 'sales'
    OPS = 'ops'
    PRODUCT = 'product'


class DocumentType(Enum):
    TELEGRAMM_MESSAGE = 'telegramm_message'
    OPS_DOCUMENT = 'ops_document'
    USER_MANUAL = 'user_manual'
    OPENAPI_SPEC = 'openapi_spec'
    COMPANY = 'company'
    EMAIL = 'email'
    OTHER = 'other'
    CALL = 'call'
    MEETING = 'meeting'
    TASK = 'task'
    DEAL = 'deal'
    NOTE = 'note'
    WIKI = 'wiki'
    STARTREK_TICKET = 'startrek_ticket'
    CONTACT = 'contact'


class BaseMetadata(BaseModel):
    type: DocumentType
    author: str
    partner: str = ''
    create_date: float
    partner_search: str = ''
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
    create_date: int
    deal_id: Optional[str] = ''
    deal_title: Optional[str] = ''
    company_id: Optional[str]
    type: DocumentType = DocumentType.TELEGRAMM_MESSAGE
    category: Category = Category.SALES


class ValidLabels(str, Enum):
    hubspot = "hubspot"
    telegram = "telegram_sales"
    telegram_ops = "telegram_ops"
    telegram_product = "telegram_product"
    wiki = "wiki"
    startrek = "startrek"
    manual = "manual"


# --- Модели данных ---
class DocumentBase(BaseMetadata):
    id: str = Field(..., description="Уникальный идентификатор документа")


class Query(BaseModel):
    text: str = Field(..., description="Вопрос или запрос данных", min_length=1)
    summarize: bool = Field(default=False, description="Флаг суммаризации, если нужно ")
    labels: List[ValidLabels] = Field(
        description="Указать доступные коллекции где производить поиск")  # Обязательный атрибут labels
    n_results: int = Field(default=5, ge=1, le=1000, description="Количество результатов для возврата")
    where: Optional[Dict[str, Any]] = Field(default=None, description="Фильтры по метаданным")


class ContextResponse(BaseModel):
    results: List[str]


class ForceSaveResponse(BaseModel):
    message: str


class AddDocumentRequest(BaseModel):
    """
    Модель запроса для добавления документа в векторное хранилище.
    """
    text: str
    label: ValidLabels
    document_id: Optional[str] = None
    metadata: DocumentBase
    chunk: Optional[bool] = True
    author: Optional[str] = None


class CompressContextRequest(BaseModel):
    """
    Модель запроса для сжатия (summarization) контекста.
    """
    question: str
    contexts: list
