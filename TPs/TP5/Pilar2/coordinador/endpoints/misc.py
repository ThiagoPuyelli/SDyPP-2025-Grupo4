from fastapi import APIRouter, Query, HTTPException
import state
from monotonic import mono_time


router = APIRouter()

@router.get("/chain")
def get_chain():
    return state.blockchain.get_chain()

@router.get("/state")
def get_state():
    return {
        "state": state.cicle_state.name,
        "description": state.cicle_state.value,
        "server-date-time": mono_time.get_hora_actual(),
        "target-prefix": state.current_target_prefix,
    }

@router.get("/block")
def get_block(hash: str = Query(..., description="Hash del bloque a buscar")):
    block = state.blockchain.get_block(hash)
    if block:
        return block
    
    raise HTTPException(
        status_code=404,
        detail="Bloque no encontrado"
    )

@router.get("/")
def root():
    return {"status": "ok"}

@router.put("/prefix")
async def submit_result(prefijo: str = Query(...)):
    state.next_target_prefix = prefijo
    return {"status": "received"}