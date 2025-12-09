from fastapi import APIRouter, HTTPException, Query
from models import MinedChain
from utils import is_valid_hash
from state import blockchain, received_chains, CoordinatorState
import state
from log_config import logger

router = APIRouter()

@router.post("/results")
async def submit_result(chain: MinedChain, miner_id: str = Query(..., description="Miner PK")):
    try:
        if state.cicle_state != CoordinatorState.OPEN_TO_RESULTS and state.cicle_state != CoordinatorState.GIVING_TASKS:
            raise HTTPException(
                status_code=403,
                detail=f"Endpoint not available, try again"
            )
        
        blocks = chain.blocks
        if not blocks:
            raise HTTPException(
                status_code=400,
                detail="Empty chain"
            )

        if blocks[0].previous_hash != blockchain.get_last_block().hash:
            raise HTTPException(
                status_code=400,
                detail="Initial block does not chain to the blockchain"
            )
        
        for i, block in enumerate(blocks):
            if not is_valid_hash(block, state.current_target_prefix):
                raise HTTPException(
                    status_code=400,
                    detail=f"Block {i} is invalid"
                )

            if i > 0 and block.previous_hash != blocks[i-1].hash:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chaining at block {i}"
                )

        received_chains.add_chain(MinedChain(blocks=blocks))
        logger.info(f"Workload recibida: {blocks}")
        return {"status": "received"}
    
    except HTTPException as e:
        logger.info(f"Workload rechazada - HTTPException: {e.detail}")
        raise