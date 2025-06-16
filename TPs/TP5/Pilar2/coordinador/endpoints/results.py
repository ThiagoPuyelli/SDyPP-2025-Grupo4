from fastapi import APIRouter
from models import MinedChain
from utils import compute_hash, is_valid_hash
import state

router = APIRouter()

@router.post("/results")
async def submit_result(chain: MinedChain):
    blocks = chain.blocks
    if not blocks:
        return {"status": "error", "detail": "Cadena vacía"}

    for i, block in enumerate(blocks):
        block_data = {
            "timestamp": block.timestamp,
            "previous_hash": block.previous_hash,
            "transaction": block.transaction.dict(),
            "nonce": block.nonce,
        }
        computed_hash = compute_hash(block_data)
        if computed_hash != block.hash or not is_valid_hash(block.hash):
            return {"status": "error", "detail": f"Bloque {i} inválido"}

        if i > 0 and block.previous_hash != blocks[i-1].hash:
            return {"status": "error", "detail": f"Encadenamiento inválido en bloque {i}"}

    last_hash = "0" if not state.blockchain else compute_hash({
        "timestamp": state.blockchain[-1]["timestamp"],
        "previous_hash": state.blockchain[-1]["previous_hash"],
        "transaction": state.blockchain[-1]["transaction"],
        "nonce": state.blockchain[-1]["nonce"],
    })

    if blocks[0].previous_hash != last_hash:
        return {"status": "error", "detail": "Bloque inicial no conecta con blockchain"}

    state.received_chains.append(blocks)
    return {"status": "received"}
