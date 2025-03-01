# config.py
import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.CHAT_TOKEN = os.getenv("CHAT_TOKEN")
        self.HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
        self.TELEGRAM_BOT_TOKEN= os.getenv("TELEGRAM_BOT_TOKEN")
        self.FIREFLIES_API_KEY = os.getenv("FIREFLIES_API_KEY")


config = Config()
sessions = {}