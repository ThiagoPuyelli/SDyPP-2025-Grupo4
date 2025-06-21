from datetime import datetime, timezone
from state import CoordinatorState
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION

def get_current_phase() -> CoordinatorState:
    now = datetime.now(timezone.utc)
    intervalo = get_last_interval_start()
    segundos = (now - intervalo).total_seconds()
    if segundos < INTERVAL_DURATION - AWAIT_RESPONSE_DURATION:
        return CoordinatorState.GIVING_TASKS
    else: 
        return CoordinatorState.OPEN_TO_RESULTS

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // INTERVAL_DURATION) * INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)