import json
import hashlib
from datetime import datetime, timezone
from config import BLOCK_TARGET_TIME, ACCEPTED_ALGORITHM, INTERVAL_DURATION
import state

def compute_hash(block_data: dict) -> str:
    block_string = json.dumps(block_data, sort_keys=True).encode()
    return hashlib.new(ACCEPTED_ALGORITHM, block_string).hexdigest()

def is_valid_hash(h: str) -> bool:
    return h.startswith(state.current_target_prefix)

def adjust_difficulty():
    if len(state.blockchain) < 2:
        return
    t1 = datetime.fromisoformat(state.blockchain[-1]["timestamp"])
    t0 = datetime.fromisoformat(state.blockchain[-2]["timestamp"])
    delta = (t1 - t0).total_seconds()
    if delta < BLOCK_TARGET_TIME / 2:
        state.current_target_prefix = "00000"
    elif delta > BLOCK_TARGET_TIME * 2:
        state.current_target_prefix = "000"

def seconds_until_next_interval(interval_minutes: int = INTERVAL_DURATION // 60) -> float:
    now = datetime.now(timezone.utc)
    minutes = now.hour * 60 + now.minute
    next_minutes = ((minutes // interval_minutes) + 1) * interval_minutes
    delta_minutes = next_minutes - minutes
    delta_seconds = delta_minutes * 60 - now.second - now.microsecond / 1_000_000
    return delta_seconds

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // INTERVAL_DURATION) * INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)