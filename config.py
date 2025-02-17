# config.py
import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.CHAT_TOKEN = os.getenv("CHAT_TOKEN")

config = Config()
sessions = {}