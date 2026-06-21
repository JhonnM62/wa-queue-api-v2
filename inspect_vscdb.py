import sqlite3, json

db_path = r'C:\Users\Administrador\AppData\Roaming\Antigravity IDE\User\workspaceStorage\988270325fbc87dd94f236eea0cf2928\state.vscdb'

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%cursor%'")
    rows = cur.fetchall()
    
    print(f'Total keys found: {len(rows)}')
    for k, v in rows:
        print(f'\nKey: {k}')
        # Si es un string muy largo, lo recortamos
        if len(v) > 500:
            print(f'Value (recortado): {v[:500]}...')
        else:
            print(f'Value: {v}')
            
    conn.close()
except Exception as e:
    print(e)
