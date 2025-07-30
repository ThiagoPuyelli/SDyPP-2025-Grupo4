from datetime import datetime, timezone
import time
import requests
from monotonic import MonotonicTime
from log_config import setup_logger_con_monotonic
import state
from state import CoordinatorState
import config

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
