import sqlite3
import os

DB_PATH = "saas_data.db"

if not os.path.exists(DB_PATH):
    print(f"[INFO] No existe '{DB_PATH}'.")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bot_id INTEGER NOT NULL,
        userbot_identifier TEXT NOT NULL,
        phone TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT,
        raw_payload TEXT,
        timestamp_ms INTEGER NOT NULL,
        FOREIGN KEY(bot_id) REFERENCES bots(id)
    );
    """)
    print("[OK] Tabla 'conversation_messages' creada o ya existía.")
    
    # Create indexes if they don't exist
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_bot_phone ON conversation_messages (bot_id, phone);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_userbot_phone ON conversation_messages (userbot_identifier, phone);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversation_messages (timestamp_ms);")
    
    conn.commit()
    print("[OK] Índices creados o ya existían.")
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    conn.close()
