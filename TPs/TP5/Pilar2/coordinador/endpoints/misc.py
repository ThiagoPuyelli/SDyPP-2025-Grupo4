from fastapi import APIRouter, Query
import state
from datetime import datetime, timedelta, timezone
from utils import seconds_until_next_interval


router = APIRouter()

@router.get("/chain")
def get_chain():
    return state.blockchain.get_chain()

@router.get("/state")
def get_state():
    seconds_left = seconds_until_next_interval()
    next_cycle_time = datetime.now(timezone.utc) + timedelta(seconds=seconds_left)

    return {
        "state": state.cicle_state.name,
        "description": state.cicle_state.value,
        "seconds_until_next_cycle": round(seconds_left),
        "next_cycle_utc": next_cycle_time.isoformat() + "Z",
    }

@router.get("/block")
def get_block(hash: str = Query(..., description="Hash del bloque a buscar")):
    block = state.blockchain.get_block_by_hash(hash)
    if block:
        return block
    return {"error": "Bloque no encontrado"}, 404