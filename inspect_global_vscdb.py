import sqlite3, json

db_path = r'C:\Users\Administrador\AppData\Roaming\Antigravity IDE\User\globalStorage\state.vscdb'

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%cursor%' OR key LIKE '%history%'")
    rows = cur.fetchall()
    
    print(f'Total keys found: {len(rows)}')
    for k, v in rows:
        print(f'\nKey: {k}')
        if len(v) > 300:
            print(f'Value (recortado): {v[:300]}...')
        else:
            print(f'Value: {v}')
            
    conn.close()
except Exception as e:
    print(e)
