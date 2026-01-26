from fastapi import APIRouter, HTTPException, Query
from models import MinedChain
from utils import is_valid_hash, tx_signature, verify_tx_signature
from state import blockchain, received_chains, CoordinatorState
import state
from log_config import logger
from metrics import record_result

router = APIRouter()

@router.post("/results")
async def submit_result(chain: MinedChain):
    try:
        if state.cicle_state != CoordinatorState.OPEN_TO_RESULTS and state.cicle_state != CoordinatorState.GIVING_TASKS:
            record_result("reject_state")
            raise HTTPException(
                status_code=403,
                detail=f"Endpoint not available, try again"
            )
        
        blocks = chain.blocks
        if not blocks:
            record_result("reject_empty")
            raise HTTPException(
                status_code=400,
                detail="Empty chain"
            )

        if blocks[0].previous_hash != blockchain.get_last_block().hash:
            record_result("reject_prev_hash")
            raise HTTPException(
                status_code=400,
                detail="Initial block does not chain to the blockchain"
            )
        
        for i, block in enumerate(blocks):
            if not is_valid_hash(block, state.persistent_state.get_prefix()):
                record_result("reject_invalid_hash")
                raise HTTPException(
                    status_code=400,
                    detail=f"Block {i} is invalid"
                )

            if i > 0 and block.previous_hash != blocks[i-1].hash:
                record_result("reject_chain_link")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chaining at block {i}"
                )
            
        seen_txs = set()
        active_sigs = {
            tx_signature(t) for t in state.active_transactions.peek_all()
            }

        for block in blocks:
            if not verify_tx_signature(block.transaction):
                record_result("reject_tx_signature")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid transaction signature"
                )
            
            sig = tx_signature(block.transaction)

            if not sig in active_sigs:
                record_result("reject_not_active")
                raise HTTPException(
                    status_code=400,
                    detail="Transaction not currently active"
                )
            if sig in seen_txs:
                record_result("reject_duplicate")
                raise HTTPException(
                    status_code=400,
                    detail="Duplicated transaction in mined chain"
                )
            seen_txs.add(sig)

        received_chains.add_chain(MinedChain(blocks=blocks))
        logger.info(f"Workload recibida: {blocks}")
        record_result("accepted")
        return {"status": "received"}
    
    except HTTPException as e:
        logger.info(f"Workload rechazada - HTTPException: {e.detail}")
        raise
