"""
Migración de la tabla 'bots' para agregar los nuevos campos.
Ejecutar con: python migrate_db.py

Es seguro correrlo múltiples veces: omite columnas que ya existan.
"""
import sqlite3
import os

DB_PATH = "saas_data.db"

if not os.path.exists(DB_PATH):
    print(f"[INFO] No existe '{DB_PATH}'. Se creará automáticamente al iniciar main.py.")
    exit(0)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Obtener columnas actuales
cursor.execute("PRAGMA table_info(bots)")
existing_cols = {row[1] for row in cursor.fetchall()}
print(f"[INFO] Columnas existentes en 'bots': {existing_cols}")

# Definir columnas a agregar: (nombre, definicion SQL)
NEW_COLUMNS = [
    ("apikey",                   "TEXT DEFAULT ''"),
    ("ai_model",                 "TEXT DEFAULT 'gemini-2.5-flash'"),
    ("thinking_budget",          "INTEGER DEFAULT -1"),
    ("thinking_level",           "TEXT DEFAULT 'HIGH'"),
    ("use_google_search",        "INTEGER DEFAULT 0"),
    ("use_google_maps",          "INTEGER DEFAULT 0"),
    ("pais",                     "TEXT DEFAULT 'colombia'"),
    ("idioma",                   "TEXT DEFAULT 'es'"),
    ("delay_seconds",            "REAL DEFAULT 0.0"),
    ("pause_timeout_minutes",    "INTEGER DEFAULT 30"),
    ("activarnotificacion",      "INTEGER DEFAULT 0"),
    ("estado_notificacion",      "TEXT DEFAULT ''"),
    ("lineaogruponotificacion",  "TEXT DEFAULT ''"),
    ("es_grupo_notificacion",    "INTEGER DEFAULT 0"),
    ("activaruserbotopcional",   "INTEGER DEFAULT 0"),
    ("userbotopcional",          "TEXT DEFAULT ''"),
    ("tools_config",             "TEXT DEFAULT '{\"buscar_catalogo\": true, \"enviar_notificacion\": true, \"obtener_hora\": true}'"),
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

if added:
    print(f"\n[OK] Migración completada. {len(added)} columna(s) nueva(s) agregada(s).")
else:
    print("\n[OK] La base de datos ya está actualizada. No se hicieron cambios.")
