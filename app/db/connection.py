import sqlite3
import os
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """Context manager for safe, leak-free SQLite connection handling."""
    # Read dynamically so it respects os.environ changes during pytest runs
    db_path = os.getenv("DB_PATH", "career_coach.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initializes the database schema."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                profile_data TEXT
            )
        ''')
        conn.commit()