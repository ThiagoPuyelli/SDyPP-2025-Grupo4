import json
import hashlib
from datetime import datetime, timedelta, timezone
from config import AWAIT_RESPONSE_DURATION, BLOCK_TARGET_TIME, ACCEPTED_ALGORITHM, INTERVAL_DURATION, CoordinatorState
import state
from typing import Optional, Tuple

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

def seconds_until_next_phase(
        current_state: CoordinatorState,
        phase_started_at: Optional[datetime],
    ) -> Tuple[Optional[float], Optional[datetime]]:
    if not phase_started_at:
        return None, None

    if current_state == CoordinatorState.GIVING_TASKS:
        phase_duration = INTERVAL_DURATION - AWAIT_RESPONSE_DURATION
    elif current_state == CoordinatorState.OPEN_TO_RESULTS:
        phase_duration = AWAIT_RESPONSE_DURATION
    else:
        # Para SELECTING_WINNER o UNSET, asumimos que no hay duración fija
        return None, None

    now = datetime.now(timezone.utc)
    elapsed = (now - phase_started_at).total_seconds()
    remaining = max(0, phase_duration - elapsed)
    next_phase_time = phase_started_at + timedelta(seconds=phase_duration)

    return remaining, next_phase_time