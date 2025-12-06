from fastapi import APIRouter, HTTPException, Query
import state

router = APIRouter()

# a los mineros dar tareas bloque a bloque, solo una. cuando uno termina de minar, el pool deberia notificarles a los demas y dar mas tareas y el bloque minado para que puedan encadenar. 
# hacer registro de los mineros en el pool

# dar tarea de minado (una sola transaccion)
@router.get("/tasks")
async def get_transaction():
    tarea = next(
        (t.transaction for t in state.tareas_disponibles if not t.mined),
        None
    )
    if tarea is None:
        raise HTTPException(
            status_code=404,
            detail="No hay tareas disponibles para minar"
        )
    return {
        "previous_hash": state.previous_hash,
        "transaction": tarea,
        "target_prefix": state.prefix,
    }

## recibir cadenas minadas, evaluarlas, si esta bien cortar el minado de los demas
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

## registro de mineros
@router.post("/login")
def get_block(hash: str = Query(..., description="Hash del bloque a buscar")):
    return