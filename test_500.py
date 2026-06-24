import requests

url = "http://100.42.185.2:8014/api/orders/573181359070/status"
data = {"status": "En proceso"}
try:
    response = requests.put(url, json=data)
    print(response.status_code)
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
