from fastapi import APIRouter, HTTPException, Query
import state
from models import MinedChain
from utils import is_valid_hash, notify_miners_new_block

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
    if not chain.blocks:
        raise HTTPException(
            status_code=400,
            detail="Empty chain"
        )
    
    block = chain.blocks[0]
    
    if block.previous_hash != state.previous_hash:
        raise HTTPException(
            status_code=400,
            detail="Block does not chain"
        )

    if not is_valid_hash(block, state.prefix):
        raise HTTPException(
            status_code=400,
            detail=f"Block is invalid"
        )

    state.mined_blocks.blocks.append(block)
    for t in state.tareas_disponibles:
        if t.transaction == block.transaction:
            t.mined = True
            break
    state.previous_hash = block.hash
    
    notify_miners_new_block()

    logger.info(f"Workload recibida: {block}")
    return {"status": "received"}

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