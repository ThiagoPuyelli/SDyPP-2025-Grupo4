import json
import time
import websocket
import threading
from log_config import logger
import state
import config

ws_connected_event = threading.Event()

def ws_listener():
    """
    Mantiene un WebSocket permanente contra el pool.
    Se reconecta automáticamente si se cae.
    """
    while True:
        try:
            logger.info("Intentando conectar WebSocket con el pool...")

            ws = websocket.WebSocket()

            # Crear conexión WS
            ws.connect(f"ws://{config.URI.replace('http://', '')}/login")

            # Enviar datos iniciales
            init_message = {
                "id": config.MINER_ID,
                "processing_tier": config.PROCESSING_TIER
            }
            ws.send(json.dumps(init_message))

            logger.info("WS conectado y minero registrado via WebSocket.")
            ws_connected_event.set()

            # Loop principal del WS
            while True:
                msg = ws.recv()  # Espera comandos opcionales del pool
                logger.info(f"Mensaje recibido por WS: {msg}")

                # Si el pool manda STOP_MINING lo manejás así:
                if msg == "STOP_MINING":
                    state.finalizar_mineria_por_pool = True

        except Exception as e:
            ws_connected_event.clear()
            logger.warning(f"WS desconectado: {e}. Reintentando en 3 segundos...")
            time.sleep(3)
