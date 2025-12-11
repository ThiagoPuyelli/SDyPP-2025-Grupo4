from cumplirTareas import minar
from utils import get_current_phase, sync_con_coordinador
from state import CoordinatorState
import state
from log_config import logger
import time
import requests
import threading
import config
import minero_websocket

stop_mining_event = threading.Event()

def iniciar ():
    global ws_connected_event

    hilo = None
    mining = False
    results_delivered = False

    # obtengo la configuracion de la blockchain
    while True:
        try:
            response = requests.get(config.URI + '/block', params={"hash": 0}, timeout=5)
            if response.ok:
                data = response.json()
                config.INTERVAL_DURATION = data["blockchain_config"]["interval_duration"]
                config.AWAIT_RESPONSE_DURATION = data["blockchain_config"]["await_response_duration"]
                config.MAX_MINING_ATTEMPTS = data["blockchain_config"]["max_mining_attempts"]
                config.ACCEPTED_ALGORITHM = data["blockchain_config"]["accepted_algorithm"]
                
                state.pool_id = data.get("pool_id", None)
                break
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)

    # Crear WS solo si es pool
    if state.pool_id:
        minero_websocket.ws_connected_event.clear()

        ws_thread = threading.Thread(target=minero_websocket.ws_listener, daemon=True)
        ws_thread.start()

        while not minero_websocket.ws_connected_event.is_set():
            logger.info("Esperando conexión WS con pool...")
            time.sleep(3)
        logger.info("WS conectado correctamente...")

    # sincronizo el reloj con el coordinador
    sync_con_coordinador()

    # ciclo principal
    while True:
        if state.pool_id and not minero_websocket.ws_connected_event.is_set():
            if mining:
                logger.warning("WS perdido. Deteniendo minería.")
                detener_mineria()
                hilo.join()
                mining = False
                results_delivered = True

        if state.finalizar_mineria_por_pool:
            if mining:
                detener_mineria()
                hilo.join()
                logger.info("Mineria finalizada por notificación del pool")
                mining = False
                results_delivered = True # no son realmente enviados (pero evito enviar si justo coincide que cambia el estado a OPEN_TO_RESULTS)
                state.finalizar_mineria_por_pool = False 
        else:
            nuevo_estado = get_current_phase(state.mono_time.get_hora_actual())
            if nuevo_estado == CoordinatorState.GIVING_TASKS:
                if not mining:
                    hilo = iniciar_minero()
                    results_delivered = False
                    mining = True
                    # vuelvo a coordinar el tiempo
                    sync_con_coordinador()
                else:
                    if state.cant_transacciones_a_minar > 0 and len(state.mined_blocks.blocks) == state.cant_transacciones_a_minar:
                        # si ya se han minado todas las transacciones, detengo el hilo del min
                        mining = False
                        enviar_resultados()
                        results_delivered = True
                        state.cant_transacciones_a_minar = 0
                    
            else:
                mining = False
                if not results_delivered:
                    if hilo:
                        detener_mineria()
                        hilo.join()
                        logger.info("Mineria finalizada, enviando resultados")
                        if len(state.mined_blocks.blocks) > 0:
                        # enviar resultados
                            enviar_resultados()
                    results_delivered = True
                    state.cant_transacciones_a_minar = 0

        time.sleep(1)

def enviar_resultados():
    try:
        while True:
            logger.info(state.mined_blocks.model_dump())
            res = requests.post(
                config.URI + "/results", 
                json=state.mined_blocks.model_dump(), 
                params={"miner_id": config.MINER_ID},
                timeout=5
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "received":
                    logger.info("Resultados recibidos por el coordinador")
                    break
            time.sleep(3)
    except requests.RequestException as e:
        logger.error(f"Error al enviar resultados al coordinador: {e}")

def iniciar_minero():
    # obtengo tareas
    data = None
    try:
        while True:
            try:
                response = requests.get(config.URI + '/tasks', timeout=5)
                if response.ok:
                    break
                else:
                    logger.info(f"Respuesta inválida del coordinador: {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Error al conectar con el coordinador: {e}")
            
            logger.info("Reintento de conexión con el coordinador en 3 segundos...")
            time.sleep(3)

    except Exception as e:
        logger.error(f"Fallo crítico al obtener tareas: {e}")
    # comienzo la mineria
    data = response.json()
    logger.info(f"Tareas recibidas: {data}")
    state.mined_blocks.blocks.clear()

    stop_mining_event.clear()
    
    hilo = threading.Thread(target=minar, args=(data, stop_mining_event,))
    hilo.start()
    
    return hilo

def detener_mineria():
    stop_mining_event.set()


iniciar()