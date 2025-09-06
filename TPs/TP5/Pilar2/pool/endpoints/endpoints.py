from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

## registrar un minero y dar trabajo
@router.get("/tasks")
async def submit_transaction():
    return
    active_tx = ActiveTransaction(transaction=tx)
    state.pending_transactions.put(active_tx)
    return {"status": "ok"}

## recibir cadenas minadas, evaluarlas?, si esta bien cortar el minado de los dema?s ? 
@router.post("/results")
async def submit_result(chain: MinedChain):
    return

## pasamanos?
@router.get("/state")
async def get_task():
    return
    if state.cicle_state != CoordinatorState.GIVING_TASKS:
        raise HTTPException(
            status_code=403,
            detail=f"Endpoint not available, wait for {CoordinatorState.GIVING_TASKS.name} state"
        )

    return {
        "previous_hash": state.blockchain.get_last_block().hash,
        "transaction": state.active_transactions.peek_all(),
        "target_prefix": state.current_target_prefix,
    }

## pasamanos?
@router.get("/block")
def get_block(hash: str = Query(..., description="Hash del bloque a buscar")):
    return