import sqlite3
import json
import datetime

class SQLiteLogger:
    def __init__(self, db_path="logs.db"):
        self.db_path = db_path
        self.init_sqlite()

    def init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                user_input TEXT,
                original_response TEXT,
                final_response TEXT,
                action_taken TEXT,
                reason TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_to_db(self, req_id, input_text, original_resp, final_resp, passed, details):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log (request_id, user_input, original_response, final_response, action_taken, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            req_id,
            input_text,
            original_resp,
            final_resp,
            "allowed" if passed else "blocked",
            json.dumps(details["failure_reasons"]),
            str(datetime.datetime.utcnow())
        ))
        conn.commit()
        conn.close()