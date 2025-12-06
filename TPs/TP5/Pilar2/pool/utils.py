from datetime import datetime, timezone
import time
import requests
from monotonic import MonotonicTime
from log_config import setup_logger_con_monotonic
import state
from state import CoordinatorState
import config
from models import ActiveTransaction

def get_current_phase(now) -> CoordinatorState:
    if now == None:
        now = datetime.now(timezone.utc)
    intervalo = get_last_interval_start(now)
    segundos = (now - intervalo).total_seconds()
    if segundos < config.INTERVAL_DURATION - config.AWAIT_RESPONSE_DURATION:
        return CoordinatorState.GIVING_TASKS
    else: 
        return CoordinatorState.OPEN_TO_RESULTS

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // config.INTERVAL_DURATION) * config.INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)

def sync_con_coordinador():
    while True:
        try:
            response = requests.get(
                config.URI + '/state',
                params={
                    "miner_id": config.MINER_ID,
                    "parametro_procesamiento": config.PROCESSING_TIER
                },
                timeout=5
            )
            if response.ok:
                data = response.json()
                state.mono_time = MonotonicTime(datetime.fromisoformat(data["server-date-time"]))
                break
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)

    # sincronizo el reloj del logger
    logger = setup_logger_con_monotonic(state.mono_time.hora_inicio, state.mono_time.start_monotonic)

def get_tareas():
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

    data = response.json()
    logger.info(f"Tareas recibidas: {data}")
    
    state.mined_blocks.blocks.clear()
    state.tareas_disponibles = []
    
    for tx in data["transaction"]:
        tr = ActiveTransaction(transaction=tx, mined=False)
        state.tareas_disponibles.append(tr)
    state.cant_transacciones_a_minar = len(state.tareas_disponibles)
    state.previous_hash = data["previous_hash"]
    state.prefix = data["target_prefix"]