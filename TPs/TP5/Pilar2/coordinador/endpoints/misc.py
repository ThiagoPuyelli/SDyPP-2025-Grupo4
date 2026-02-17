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
        "target-prefix": state.persistent_state.get_prefix(),
    }

@router.get("/cycle/summary")
def get_cycle_summary():
    active_transactions = state.active_transactions.get_all_transactions_with_ttl()
    pending_transactions = state.pending_transactions.get_all_transactions_with_ttl()

    return {
        "last_cycle": state.persistent_state.get_last_cycle_summary(),
        "current": {
            "state": state.cicle_state.name,
            "counts": {
                "active_transactions": len(active_transactions),
                "pending_transactions": len(pending_transactions),
            },
            "active_transactions": [tx.model_dump() for tx in active_transactions],
            "pending_transactions": [tx.model_dump() for tx in pending_transactions],
        },
    }

@router.get("/block")
def get_block(hash: str = Query(..., description="Hash del bloque a buscar")):
    if (hash == '0'):
        block = state.blockchain.get_genesis()
    else:
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
