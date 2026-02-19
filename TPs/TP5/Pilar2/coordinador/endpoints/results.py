from fastapi import APIRouter, HTTPException
from models import MinedChain
from utils import is_valid_hash, tx_signature, verify_tx_signature
from state import blockchain, received_chains, CoordinatorState
import state
from log_config import logger
from metrics import record_result, record_block_validation_duration
from time import perf_counter

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

        prefix = state.persistent_state.get_prefix()
        seen_txs = set()
        active_sigs = {
            tx_signature(t) for t in state.active_transactions.peek_all()
            }

        for i, block in enumerate(blocks):
            block_started = perf_counter()

            if not is_valid_hash(block, prefix):
                record_result("reject_invalid_hash")
                record_block_validation_duration(prefix, "reject_invalid_hash", perf_counter() - block_started)
                raise HTTPException(
                    status_code=400,
                    detail=f"Block {i} is invalid"
                )

            if i > 0 and block.previous_hash != blocks[i-1].hash:
                record_result("reject_chain_link")
                record_block_validation_duration(prefix, "reject_chain_link", perf_counter() - block_started)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chaining at block {i}"
                )

            if not verify_tx_signature(block.transaction):
                record_result("reject_tx_signature")
                record_block_validation_duration(prefix, "reject_tx_signature", perf_counter() - block_started)
                raise HTTPException(
                    status_code=400,
                    detail="Invalid transaction signature"
                )
            
            sig = tx_signature(block.transaction)

            if sig not in active_sigs:
                record_result("reject_not_active")
                record_block_validation_duration(prefix, "reject_not_active", perf_counter() - block_started)
                raise HTTPException(
                    status_code=400,
                    detail="Transaction not currently active"
                )
            if sig in seen_txs:
                record_result("reject_duplicate")
                record_block_validation_duration(prefix, "reject_duplicate", perf_counter() - block_started)
                raise HTTPException(
                    status_code=400,
                    detail="Duplicated transaction in mined chain"
                )
            seen_txs.add(sig)
            record_block_validation_duration(prefix, "accepted", perf_counter() - block_started)

        received_chains.add_chain(MinedChain(blocks=blocks))
        logger.info(f"Workload recibida: {blocks}")
        record_result("accepted")
        return {"status": "received"}
    
    except HTTPException as e:
        logger.info(f"Workload rechazada - HTTPException: {e.detail}")
        raise
