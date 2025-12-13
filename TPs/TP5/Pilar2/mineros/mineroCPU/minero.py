from cumplirTareas import minar
from utils import get_current_phase, sync_con_coordinador
from state import CoordinatorState
import state
from log_config import logger
import time
import requests
import threading
import config
import pika

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

    # crear cola efimera solo si es con pool
    if state.pool_id:

        credentials = pika.PlainCredentials(
            username=config.RABBIT_USER,
            password=config.RABBIT_PASS
        )

        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=config.RABBIT_HOST, 
            credentials=credentials)
            )
        channel = connection.channel()
        
        result = channel.queue_declare(
            queue="",
            exclusive=True,
            auto_delete=True
        )

        queue_name = result.method.queue

        channel.queue_bind(
            exchange="blockchain.exchange",
            queue=queue_name
        )

        channel.basic_consume(
            queue=queue_name,
            on_message_callback=frenar_minado_pool,
            auto_ack=True
        )

        channel.start_consuming()

    # sincronizo el reloj con el coordinador
    sync_con_coordinador()

    # ciclo principal (sin pool)
    while True:
        if not state.pool_id:
            nuevo_estado = get_current_phase(state.mono_time.get_hora_actual())
            if nuevo_estado == CoordinatorState.GIVING_TASKS:
                if not mining and not results_delivered:
                    hilo = iniciar_minero()
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
                results_delivered = False
                if state.cant_transacciones_a_minar > 0 and:
                    if hilo:
                        detener_mineria()
                        hilo.join()
                        logger.info("Mineria finalizada, enviando resultados")
                        if len(state.mined_blocks.blocks) > 0:
                        # enviar resultados
                            enviar_resultados()
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

def obtener_tareas():
        # obtengo tareas
    data = None
    try:
        while True:
            try:
                response = requests.get(config.URI + '/tasks', timeout=5)
                if response.ok:
                    data = response.json()
                    break
                else:
                    logger.info(f"Respuesta inválida del coordinador: {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Error al conectar con el coordinador: {e}")
            
            logger.info("Reintento de conexión con el coordinador en 3 segundos...")
            time.sleep(3)

    except Exception as e:
        logger.error(f"Fallo crítico al obtener tareas: {e}")

    return data

def iniciar_minero():
    
    data = obtener_tareas()

    logger.info(f"Tareas recibidas: {data}")
    state.mined_blocks.blocks.clear()

    stop_mining_event.clear()
    
    hilo = threading.Thread(target=minar, args=(data, stop_mining_event,))
    hilo.start()
    
    return hilo

def detener_mineria():
    stop_mining_event.set()

def frenar_minado_pool(ch, method, properties, body):
    threading.Thread(
        target=handle_frenar_minado
    ).start()
    
def handle_frenar_minado():
    print("Evento recibido desde pool - interrumpiendo minado")
    
    detener_mineria()
    
    data = obtener_tareas()
    logger.info(f"Tareas recibidas: {data}")
    state.mined_blocks.blocks.clear()
    stop_mining_event.clear()
    minar(data, stop_mining_event)

    if len(state.mined_blocks.blocks) > 0:
        enviar_resultados()


iniciar()