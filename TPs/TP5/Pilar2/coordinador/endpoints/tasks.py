from fastapi import APIRouter
from models import Transaction
from state import cicle_state, CoordinatorState, pending_transactions, active_transactions, current_target_prefix, blockchain
from utils import create_genesis_block, compute_hash

router = APIRouter()

@router.post("/tasks")
async def submit_transaction(tx: Transaction):
    pending_transactions.put(tx)
    return {"status": "ok"}

@router.get("/tasks")
async def get_task():
    if not cicle_state == CoordinatorState.GIVING_TASKS:
        return {"status": "error", "message": f"endpoint not available, wait for {CoordinatorState.GIVING_TASKS.name} state"}

    if blockchain.is_empty:
        create_genesis_block()

    return {
        "previous_hash": blockchain.get_last_block().hash,
        "transaction": active_transactions.peek_all(),
        "target_prefix": current_target_prefix,
    }
