# backend/database/connection.py
from supabase import create_client, Client
from backend.config import SUPABASE_URL, SUPABASE_KEY

_client: Client = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client