from fastapi import APIRouter
from models import MinedChain
from utils import compute_hash, is_valid_hash
from state import blockchain, received_chains, cicle_state, CoordinatorState

router = APIRouter()

@router.post("/results")
async def submit_result(chain: MinedChain):
    if not cicle_state == CoordinatorState.OPEN_TO_RESULTS:
        return {"status": "error", "message": f"endpoint not available, wait for {CoordinatorState.OPEN_TO_RESULTS.name} state"}
    
    blocks = chain.blocks
    if not blocks:
        return {"status": "error", "detail": "Empty chain"}

    for i, block in enumerate(blocks):
        block_data = {
            "timestamp": block.timestamp,
            "previous_hash": block.previous_hash,
            "transaction": block.transaction.dict(),
            "nonce": block.nonce,
        }
        computed_hash = compute_hash(block_data)
        if computed_hash != block.hash or not is_valid_hash(block.hash):
            return {"status": "error", "detail": f"Block {i} is invalid"}

        if i > 0 and block.previous_hash != blocks[i-1].hash:
            return {"status": "error", "detail": f"Invalid chaining at block {i}"}

    last_hash = "0" if not blockchain else compute_hash({
        "timestamp": blockchain[-1]["timestamp"],
        "previous_hash": blockchain[-1]["previous_hash"],
        "transaction": blockchain[-1]["transaction"],
        "nonce": blockchain[-1]["nonce"],
    })

    if blocks[0].previous_hash != last_hash:
        return {"status": "error", "detail": "Initial block does not chain to the block chain"}

    received_chains.append(blocks)
    return {"status": "received"}
