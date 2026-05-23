import sqlite3
from datetime import datetime

DB_PATH = "deepfake_results.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            probability REAL,
            threshold REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_result(filename, prediction, probability, threshold):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions (filename, prediction, probability, threshold, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename,
        prediction,
        probability,
        threshold,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def fetch_all_results():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cur.fetchall()

    conn.close()
    return rows

def delete_all_results():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
