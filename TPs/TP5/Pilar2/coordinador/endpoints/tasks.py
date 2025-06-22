from fastapi import APIRouter, HTTPException
from models import ActiveTransaction, Transaction
from state import CoordinatorState, pending_transactions, active_transactions, current_target_prefix, blockchain
import state

router = APIRouter()

@router.post("/tasks")
async def submit_transaction(tx: Transaction):
    active_tx = ActiveTransaction(transaction=tx)
    pending_transactions.put(active_tx)
    return {"status": "ok"}

@router.get("/tasks")
async def get_task():
    if state.cicle_state != CoordinatorState.GIVING_TASKS:
        raise HTTPException(
            status_code=403,
            detail=f"Endpoint not available, wait for {CoordinatorState.GIVING_TASKS.name} state"
        )

    return {
        "previous_hash": blockchain.get_last_block().hash,
        "transaction": active_transactions.peek_all(),
        "target_prefix": current_target_prefix,
    }

