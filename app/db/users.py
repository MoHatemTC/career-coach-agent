import json
import os
from app.db.connection import get_db_connection

def save_user_profile(username: str, profile_data: dict):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, profile_data) 
            VALUES (?, ?) 
            ON CONFLICT(username) DO UPDATE SET profile_data=excluded.profile_data
        ''', (username, json.dumps(profile_data)))
        conn.commit()

def get_user_profile(username: str) -> dict:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT profile_data FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        
    if row:
        return json.loads(row['profile_data'])
    return {}

def get_all_users() -> list:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users")
        rows = cursor.fetchall()
    return [row['username'] for row in rows]

def seed_sample_user():
    if not get_user_profile("student_intern"):
        json_path = os.path.join(os.getcwd(), "data", "sample_user.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                sample_data = json.load(f)
                save_user_profile(sample_data["username"], sample_data["profile_data"])