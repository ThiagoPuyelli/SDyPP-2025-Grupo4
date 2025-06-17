from fastapi import APIRouter
from models import MinedChain
from utils import compute_hash, is_valid_hash, create_genesis_block
from state import blockchain, received_chains, cicle_state, CoordinatorState

router = APIRouter()

@router.post("/results")
async def submit_result(chain: MinedChain):
    if not cicle_state == CoordinatorState.OPEN_TO_RESULTS:
        return {"status": "error", "message": f"endpoint not available, wait for {CoordinatorState.OPEN_TO_RESULTS.name} state"}
    
    blocks = chain.blocks
    if not blocks:
        return {"status": "error", "message": "Empty chain"}

    if blockchain.is_empty(): create_genesis_block()
    if blocks[0].previous_hash != blockchain.get_last_block().hash:
        return {"status": "error", "message": "Initial block does not chain to the block chain"}
    
    for i, block in enumerate(blocks):
        block_data = {
            "previous_hash": block.previous_hash,
            "transaction": block.transaction.dict(),
            "nonce": block.nonce,
            "miner_id": block.miner_id,
        }
        computed_hash = compute_hash(block_data)
        if computed_hash != block.hash or not is_valid_hash(block.hash):
            return {"status": "error", "message": f"Block {i} is invalid"}

        if i > 0 and block.previous_hash != blocks[i-1].hash:
            return {"status": "error", "message": f"Invalid chaining at block {i}"}

    received_chains.add_chain(blocks)
    return {"status": "received"}
