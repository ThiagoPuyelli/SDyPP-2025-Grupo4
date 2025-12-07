from fastapi import APIRouter, HTTPException, Query
import state
from models import MinedChain, Miner
from utils import is_valid_hash, notify_miners_new_block
import os
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

router = APIRouter()

# a los mineros dar tareas bloque a bloque, solo una. cuando uno termina de minar, el pool deberia notificarles a los demas y dar mas tareas y el bloque minado para que puedan encadenar. 

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
async def submit_result(chain: MinedChain, miner_id: str = Query(..., description="Miner PK")):
    if not state.mineros_activos.validar_minero(Miner(id=miner_id, processing_tier=0, endpoint="")):
        raise HTTPException(
            status_code=403,
            detail="Miner not registered, login with /login first"
        )
    
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

## registro de mineros
@router.post("/login")
async def login(id: str = Query(..., description="Miner PK"), 
                endpoint: str = Query(..., description="Miner endpoint URL"),
                processing_tier: int = Query(..., description="Miner mining speed")):
    
    challenge = os.urandom(32)
    challenge_b64 = base64.b64encode(challenge).decode()

    miner = Miner(
        id=id,
        processing_tier=processing_tier,
        endpoint=endpoint,
        login_challenge=challenge_b64
    )
    if state.mineros_activos.validar_minero(miner):
        logger.info(f"Minero ya registrado: {miner}")
        return {"status": "received"}
    if state.mineros_pendientes_de_registro.validar_minero(miner):
        logger.info(f"Solicitud de login ya registrada: {miner}")
        return {"status": "received"}
    
    state.mineros_pendientes_de_registro.agregar_minero(miner)
    logger.info(f"Minero solicitando login: {miner}")
    return {"status": "received", 
            "challenge": challenge_b64}

@router.post("/verify_login")
async def verify_login(id: str = Query(..., description="Miner PK"), 
                       response: str = Query(..., description="Response to challenge")):
    
    miner = next(
        (m for m in state.mineros_pendientes_de_registro.get_all_miners() if m.id == id),
        None
    )
    if miner is None:
        raise HTTPException(
            status_code=403,
            detail="No login request found for this miner"
        )
    
    try:
        challenge = base64.b64decode(miner.login_challenge)
        signature = base64.b64decode(response)
        public_key = serialization.load_pem_public_key(id.encode())
        public_key.verify(
            signature,
            challenge,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

    except Exception as e:
        raise HTTPException(
            status_code=403,
            detail="Invalid response to challenge (signature check failed)"
        )
    
    state.mineros_pendientes_de_registro.eliminar_minero(miner)
    state.mineros_activos.agregar_minero(miner)
    logger.info(f"Minero registrado exitosamente: {miner}")
    return {"status": "logged_in"}

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