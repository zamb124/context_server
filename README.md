# Документация сервиса для обработки документов и суммаризации

Этот сервис предназначен для:
- Индексации и хранения документов (текстов) в векторном хранилище (ChromaDB).
- Поиска по документам с фильтрацией по метаданным.
- Суммаризации (генерации краткого описания) текста с использованием NLP-моделей.
- Интеграции с внешними системами, такими как Telegram и HubSpot.

Сервис построен на базе [FastAPI](https://fastapi.tiangolo.com/) и использует [uvicorn](https://www.uvicorn.org/) для запуска сервера. Для работы с NLP-моделями применяется библиотека [transformers](https://huggingface.co/transformers/), а для хранения векторов документов — ChromaDB. Для асинхронной обработки запросов используется стандартная библиотека Python `asyncio`.

---

## Структура проекта

Проект состоит из следующих основных файлов:

- **main.py**  
  Основной модуль, в котором:
  - Инициализируется приложение FastAPI.
  - Определяются API-эндпоинты.
  - Настраиваются зависимости (например, проверка токена для доступа к API).
  - Запускаются фоновые процессы (например, процессы для обработки запросов к модели).
  - Реализуются интеграции с Telegram и HubSpot.

- **model_process.py**  
  Модуль для работы с NLP-моделью в отдельном процессе. Отвечает за:
  - Загрузку модели и токенизатора.
  - Ожидание сигнала `MODEL_READY` для уведомления основного процесса о готовности.
  - Обработку входящих запросов (через стандартные потоки stdin/stdout) и генерацию ответов.

- **chromadb_utils.py**  
  Утилиты для работы с векторным хранилищем ChromaDB:
  - Функция для получения или создания коллекций документов по метке.
  - Функция для вставки (upsert) и удаления документов в коллекциях.

- **text_utils.py**  
  Утилиты для обработки текста, например, функция семантического разбиения текста на логически завершённые чанки. Это полезно для подготовки длинных текстов к суммаризации.

---

## 1. Файл `main.py`

### 1.1. Импорты и конфигурация

В начале файла импортируются стандартные библиотеки (для работы с асинхронностью, логированием, обработкой сигналов и т.д.), сторонние библиотеки (FastAPI, uvicorn, torch, nltk) и локальные модули (например, для работы с ChromaDB, Telegram и текстовыми утилитами):

```python
import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any

import nltk
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header, status, Body, Depends
from pydantic import BaseModel
from starlette.requests import Request

from chromadb_utils import get_collection, upsert_to_collection
from config import config
from models import ValidLabels, DocumentBase, Query, ContextResponse, ForceSaveResponse
from telegram_integration import telegram_integration
from text_utils import split_text_semantically