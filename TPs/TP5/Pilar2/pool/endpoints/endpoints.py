from fastapi import APIRouter, HTTPException, Query
import state
from models import MinedChain
from utils import is_valid_hash, publish_seguro, tx_signature, verify_tx_signature
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
async def submit_result(chain: MinedChain, miner_pk: str = Query(..., description="Public key del minero")):
    global is_share
    is_share = False

    if not chain.blocks:
        logger.info("HTTP 400 - Empty chain")
        raise HTTPException(
            status_code=400,
            detail="Empty chain"
        )
    
    block = chain.blocks[0]

    if block.miner_id != config.POOL_ID:
        logger.info("HTTP 400 - Miner ID does not match this pool's ID")
        raise HTTPException(
            status_code=400,
            detail="Miner ID does not match this pool's ID"
        )
    
    if block.previous_hash != state.previous_hash:
        logger.info("HTTP 400 - Block does not chain")
        raise HTTPException(
            status_code=400,
            detail="Block does not chain"
        )

    # valido hash completo
    if not is_valid_hash(block, state.prefix):
        # valido si es un share
        if len(state.prefix) <= 1 or not is_valid_hash(block, state.prefix[:-1]):
            logger.info("HTTP 400 - Block is invalid")
            raise HTTPException(
                status_code=400,
                detail=f"Block is invalid"
            )
        else: # (es un share)
            is_share = True

    if not state.queue_channel:
        logger.info("HTTP 400 - Pool not yet initialized")
        raise HTTPException(
            status_code=400,
            detail="Pool not yet initialized"
        )

    if not verify_tx_signature(block.transaction):
        logger.info("HTTP 400 - Invalid transaction signature")
        raise HTTPException(
            status_code=400,
            detail="Invalid transaction signature"
        )

    sig = tx_signature(block.transaction)

    for t in state.tareas_disponibles:
        if tx_signature(t.transaction) == sig:
            
            if t.mined:
                logger.info("HTTP 400 - Transaction already mined")
                raise HTTPException(
                    status_code=400,
                    detail="Transaction already mined"
                )

            state.mineros_activos.share_recibido(miner_pk)
            
            if is_share:
                logger.info(f"Share recibida: {block}")

            else:
                t.mined = True
                state.mined_blocks.blocks.append(block)
                state.previous_hash = block.hash
                
                event = {
                    "type": "NEW_TX"
                }
                publish_seguro(event)
                
                logger.info("Notificando mineros")
                logger.info(f"Workload recibida: {block}")
            
            return {"status": "received"}
    logger.info("HTTP 400 - Transaction not in available tasks")
    raise HTTPException(
        status_code=400,
        detail="Transaction not in available tasks"
    )

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