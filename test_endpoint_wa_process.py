import json
import os
import sys
import urllib.request
import urllib.error

def run_endpoint_test():
    try:
        # 1. Leer los datos de prueba
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "test_request.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("=== Iniciando Simulación de Petición al Endpoint /wa/process ===")
        print(f"Userbot: {data.get('userbot')}")
        print(f"Teléfono: {data.get('lineaWA')}")
        print(f"Mensaje enviado: {data.get('mensaje_reciente')}")
        
        # 2. Configurar la petición HTTP
        url = "http://127.0.0.1:8000/wa/process"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Convertir datos a bytes
        json_data = json.dumps(data).encode('utf-8')
        
        # 3. Hacer la petición POST
        print("\n--- Enviando petición POST a http://127.0.0.1:8000/wa/process ...")
        req = urllib.request.Request(url, data=json_data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req) as response:
            status_code = response.getcode()
            response_body = response.read().decode('utf-8')
            
            print(f"\n[OK] RESPUESTA DEL SERVIDOR (Código {status_code}):")
            print(response_body)
            
            # Intentar parsear el JSON de respuesta para mostrarlo bonito
            try:
                parsed_resp = json.loads(response_body)
                print("\nEstructura recibida:")
                print(json.dumps(parsed_resp, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                pass

    except urllib.error.HTTPError as e:
        print(f"\n[ERROR] HTTP {e.code}: {e.reason}")
        print(e.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"\n[ERROR] DE CONEXIÓN: {e.reason}")
        print("¿Está corriendo el servidor (python main.py)?")
    except Exception as e:
        print("\n[ERROR] DURANTE LA PRUEBA:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_endpoint_test()
