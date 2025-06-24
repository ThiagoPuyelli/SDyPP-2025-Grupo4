from datetime import datetime
from cumplirTareas import minar
from utils import get_current_phase, sync_con_coordinador
from state import CoordinatorState
import state
from log_config import logger
import time
import requests
import threading
import config

stop_mining_event = threading.Event()

def iniciar ():
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
                break
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)

    # sincronizo el reloj con el coordinador
    sync_con_coordinador()

    # ciclo principal
    while True:
        nuevo_estado = get_current_phase(state.mono_time.get_hora_actual())
        if nuevo_estado == CoordinatorState.GIVING_TASKS:
            results_delivered = False
            if not mining:
                hilo = iniciar_minero()
                mining = True
                # vuelvo a coordinar el tiempo
                sync_con_coordinador()
                
        else:
            mining = False
            if not results_delivered:
                if hilo:
                    detener_mineria()
                    hilo.join()
                    logger.info("Mineria finalizada, enviando resultados")
                    # enviar resultados
                    if len(state.mined_blocks.blocks) > 0:
                        try:
                            while True:
                                logger.info(state.mined_blocks.model_dump())
                                res = requests.post(config.URI + "/results", json=state.mined_blocks.model_dump(), timeout=5)
                                if res.status_code == 200:
                                    data = res.json()
                                    if data.get("status") == "received":
                                        logger.info("Resultados recibidos por el coordinador")
                                        break
                                time.sleep(3)
                        except requests.RequestException as e:
                            logger.error(f"Error al enviar resultados al coordinador: {e}")
                results_delivered = True

        time.sleep(1)

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