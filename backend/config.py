# backend/config.py
import os
from dotenv import load_dotenv

# Корневая папка проекта (на уровень выше backend)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(ROOT_DIR, '.env')
load_dotenv(DOTENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
MODELS_DIR = r"C:\models"
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, 'paraphrase-multilingual-MiniLM-L12-v2')
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'rubert-base-cased-sentiment')

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')

LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gemma2:9b')

LLM_API_KEY = os.getenv('LLM_API_KEY', '')

LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:11434')
