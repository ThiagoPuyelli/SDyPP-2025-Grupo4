import json
import hashlib
from datetime import datetime, timedelta
from config import BLOCK_TARGET_TIME, ACCEPTED_ALGORITHM
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

def seconds_until_next_interval(interval_minutes: int) -> float:
    now = datetime.utcnow()
    next_minute = ((now.minute // interval_minutes) + 1) * interval_minutes
    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)
    if next_time <= now:
        next_time += timedelta(minutes=interval_minutes)
    return (next_time - now).total_seconds()
