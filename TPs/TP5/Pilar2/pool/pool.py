from utils import get_current_phase, sync_con_coordinador, get_tareas, publish_seguro, conectar_rabbit
from state import CoordinatorState
import state
from log_config import logger
import time
import requests
import config

def iniciar ():
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
                config.BLOCKCHAIN_PRIZE_AMOUNT = data["blockchain_config"]["prize_amount"]
                break
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)

    # sincronizo el reloj con el coordinador
    sync_con_coordinador()

    # creo canal para el exchange
    state.rabbit_connection, state.queue_channel = conectar_rabbit()

    # ciclo principal
    while True:
        nuevo_estado = get_current_phase(state.mono_time.get_hora_actual())
        if nuevo_estado == CoordinatorState.GIVING_TASKS:
            if not mining and not results_delivered:

                get_tareas()

                event = {
                    "type": "NEW_TX"
                }
                publish_seguro(event)
                logger.info("Notificando mineros")

                mining = True
                state.nonce_start = 0
                sync_con_coordinador()
            elif state.cant_transacciones_a_minar > 0 and len(state.mined_blocks.blocks) == state.cant_transacciones_a_minar:
                ## si termino enviar tareas
                enviar_resultados()
                mining = False
                results_delivered = True
                state.cant_transacciones_a_minar = 0
        else:
            ## si hay que entregar resultados hacerlo
            mining = False
            results_delivered = False
            if state.cant_transacciones_a_minar > 0:
                if len(state.mined_blocks.blocks) > 0:
                    enviar_resultados()
                state.cant_transacciones_a_minar = 0

        time.sleep(1)

def enviar_resultados():
    try:
        while True:
            logger.info(state.mined_blocks.model_dump())
            res = requests.post(config.URI + "/results", json=state.mined_blocks.model_dump(), timeout=5)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "received":
                    logger.info("Resultados recibidos por el coordinador")
                    break
            elif res.status_code == 400:
                logger.info("Resultados rechazados por el coordinador")
                break
            time.sleep(3)
    except requests.RequestException as e:
        logger.error(f"Error al enviar resultados al coordinador: {e}")
