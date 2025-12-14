from fastapi import APIRouter, HTTPException, Query
import state
from models import MinedChain
from utils import is_valid_hash, publish_seguro
import requests
import config
import time
from log_config import logger

router = APIRouter()

## dar tarea de minado (una sola transaccion)
@router.get("/tasks")
async def get_transaction():
    tarea = next(
        (t.transaction for t in state.tareas_disponibles if not t.mined),
        None
    )
    if tarea is None:
        raise HTTPException(
            status_code=204,
            detail="No hay tareas disponibles para minar"
        )
    
    state.nonce_start += 1000000000
    
    return {
        "previous_hash": state.previous_hash,
        "transaction": [tarea],
        "target_prefix": state.prefix,
        "nonce_start": state.nonce_start - 1000000000,
        "nonce_end": state.nonce_start
    }

## recibir cadenas minadas, evaluarlas, si esta bien cortar el minado de los demas
@router.post("/results")
async def submit_result(chain: MinedChain, miner_id: str = Query(..., description="Miner PK")):
    # if not state.mineros_activos.validar_minero(miner_id):
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Miner not registered, login with /login first"
    #     )
    
    # if miner_id not in state.conexiones_ws:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Miner not connected via WebSocket"
    #     )

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
    
    event = {
        "type": "NEW_TX"
    }

    if not state.queue_channel:
        raise HTTPException(
            status_code=400,
            detail="Pool not yet initialized"
        )

    publish_seguro(event)
    
    logger.info("Notificando mineros")

    logger.info(f"Workload recibida: {block}")
    return {"status": "received"}

# ## login de minero via websocket
# @router.websocket("/login")
# async def miner_ws(ws: WebSocket):
#     await ws.accept()

#     # Primer mensaje contiene ID + tier
#     init_msg = await ws.receive_json()
#     miner_id = init_msg["id"]
#     processing_tier = init_msg["processing_tier"]

#     # Registrar minero
#     miner = Miner(id=miner_id, processing_tier=processing_tier)
#     state.mineros_activos.agregar_minero(miner)
#     state.conexiones_ws[miner_id] = ws

#     logger.info(f"Miner WS connected: {miner_id}")

#     try:
#         while True:
#             await ws.receive_text()
#     except WebSocketDisconnect:
#         logger.info(f"Miner disconnected: {miner_id}")
#         state.conexiones_ws.pop(miner_id, None)
#         state.mineros_activos.eliminar_minero(miner_id)

## pasamanos
@router.get("/state")
async def get_task():
    while True:
        try:
            response = requests.get(
                config.URI + '/state',
                timeout=5
            )
            if response.ok:
                return response.json()
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)

## pasamanos
@router.get("/block")
def get_block(hash: str = Query(..., description="Hash del bloque a buscar")):
    while True:
        try:
            response = requests.get(config.URI + '/block', params={"hash": hash}, timeout=5)
            if response.ok:
                r_mod = response.json()
                r_mod["pool_id"] = config.POOL_ID
                return r_mod
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)