from fastapi import APIRouter
import state
from datetime import datetime, timedelta, timezone
from utils import seconds_until_next_interval, seconds_until_next_phase


router = APIRouter()

@router.get("/chain")
def get_chain():
    return state.blockchain

@router.get("/state")
def get_state():
    seconds_left = seconds_until_next_interval()
    next_cycle_time = datetime.now(timezone.utc) + timedelta(seconds=seconds_left)

    seconds_until_phase, next_phase_utc = seconds_until_next_phase(state.cicle_state, state.phase_started_at)

    return {
        "state": state.cicle_state.name,
        "description": state.cicle_state.value,
        "seconds_until_next_cycle": round(seconds_left),
        "next_cycle_utc": next_cycle_time.isoformat() + "Z",
        "seconds_until_next_phase": round(seconds_until_phase) if seconds_until_phase is not None else None,
        "next_phase_utc": next_phase_utc.isoformat() + "Z" if next_phase_utc else None,
    }
