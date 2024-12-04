from supabase import create_client

# URL y API Key de Supabase
URL = 'https://beryfdwrzvykxrnnshxa.supabase.co'
KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJlcnlmZHdyenZ5a3hybm5zaHhhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI2Nzc3MTUsImV4cCI6MjA0ODI1MzcxNX0.4gCYexJCUYQHEWYj2J5CceKSNvBXqC3SxwNT8fBE9cU'

def get_supabase_client():
    """Crea y devuelve una instancia del cliente Supabase."""
    return create_client(URL, KEY)