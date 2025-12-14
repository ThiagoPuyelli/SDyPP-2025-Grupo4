from fastapi import APIRouter, HTTPException
from models import ActiveTransaction, Transaction
from state import CoordinatorState
import state
from utils import verify_tx_signature, has_sufficient_funds

router = APIRouter()

@router.post("/tasks")
async def submit_transaction(tx: Transaction):

    if tx.source == tx.target:
        raise HTTPException(
            status_code=400,
            detail="Source and target cannot be the same"
        )
    
    if tx.amount <= 0:
        raise HTTPException(
            status_code=400,
            detail="Amount must be positive"
        )
    
    if tx.source == "0" or not verify_tx_signature(tx):
        raise HTTPException(
            status_code=400,
            detail="Invalid transaction signature"
        )
    
    if not has_sufficient_funds(tx):
        raise HTTPException(
            status_code=400,
            detail="Insufficient funds"
        )

    active_tx = ActiveTransaction(transaction=tx)
    state.pending_transactions.put(active_tx)
    return {"status": "ok"}

@router.get("/tasks")
async def get_task():
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

