import json, os, sqlite3

sys_path = r'C:\Users\Administrador\AppData\Roaming\Antigravity IDE\User\workspaceStorage'

for d in os.listdir(sys_path):
    wpath = os.path.join(sys_path, d, 'workspace.json')
    if os.path.exists(wpath):
        with open(wpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            folder = data.get('folder', 'Unknown')
            print(f'Workspace ID: {d}')
            print(f'Path: {folder}')
            
            db_path = os.path.join(sys_path, d, 'state.vscdb')
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute("SELECT key FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%cursor%' OR key LIKE '%ai%'")
                    keys = cur.fetchall()
                    if keys:
                        print(f'  [+] Found DB keys: {[k[0] for k in keys[:5]]}...')
                    conn.close()
                except Exception as e:
                    print(f'  [-] Error reading DB: {e}')
            print('-' * 40)
