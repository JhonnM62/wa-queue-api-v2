import sqlite3
import os

DB_PATH = "saas_data.db"

if not os.path.exists(DB_PATH):
    print(f"[INFO] No existe '{DB_PATH}'.")
    exit(0)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get existing columns
cursor.execute("PRAGMA table_info(bots)")
existing_cols = {row[1] for row in cursor.fetchall()}
print(f"[INFO] Columnas existentes en 'bots': {existing_cols}")

NEW_COLUMNS = [
    ("numerodemensajes", "INTEGER DEFAULT 10"),
    ("temperature", "REAL DEFAULT 0.5"),
    ("topP", "REAL DEFAULT 0.95"),
    ("maxOutputTokens", "INTEGER DEFAULT 4096")
]

added = []
for col_name, col_def in NEW_COLUMNS:
    if col_name not in existing_cols:
        try:
            cursor.execute(f"ALTER TABLE bots ADD COLUMN {col_name} {col_def}")
            added.append(col_name)
            print(f"[OK] Columna agregada: {col_name}")
        except Exception as e:
            print(f"[WARN] No se pudo agregar '{col_name}': {e}")
    else:
        print(f"[SKIP] Ya existe: {col_name}")

conn.commit()
conn.close()

print(f"\n[OK] Migración completada. {len(added)} columna(s) nueva(s) agregada(s).")
