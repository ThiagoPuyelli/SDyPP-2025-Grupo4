import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

API_URL = "http://server:8004/health"

while True:
    start = time.time()
    try:
        response = requests.get(API_URL)
        latency = (time.time() - start) * 1000  # en milisegundos
        logging.info(f"Status: {response.status_code}, Latencia: {latency:.2f} ms")
    except Exception as e:
        logging.error(f"Error al conectar: {e}")
    time.sleep(5)