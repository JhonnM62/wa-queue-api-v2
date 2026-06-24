import sqlite3
conn = sqlite3.connect('./saas_data.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in saas_data.db:")
for row in cursor.fetchall():
    print(f"- {row[0]}")
conn.close()
