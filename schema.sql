CREATE TABLE IF NOT EXISTS audit_log (Add commentMore actions
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    user_input TEXT,
    original_response TEXT,
    final_response TEXT,
    action_taken TEXT,
    reason TEXT,
    timestamp TEXT
);