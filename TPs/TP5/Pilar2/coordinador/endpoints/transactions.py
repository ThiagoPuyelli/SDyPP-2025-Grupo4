from fastapi import APIRouter
from models import Transaction
from state import cicle_state, CoordinatorState, pending_transactions, active_transactions, current_target_prefix, blockchain

router = APIRouter()

@router.post("/tasks")
async def submit_transaction(tx: Transaction):
    pending_transactions.append(tx.dict())
    return {"status": "ok"}

@router.get("/tasks")
async def get_task():
    if not cicle_state == CoordinatorState.GIVING_TASKS:
        return {"status": "error", "message": f"endpoint not available, wait for {CoordinatorState.GIVING_TASKS.name} state"}

    from utils import compute_hash

    if not blockchain:
        previous_hash = "0"
    else:
        last_block = blockchain[-1]
        block_data = {
            "timestamp": last_block["timestamp"],
            "previous_hash": last_block["previous_hash"],
            "transaction": last_block["transaction"],
            "nonce": last_block["nonce"]
        }
        previous_hash = compute_hash(block_data)

    if not active_transactions:
        return {"message": "no tasks"}

    return {
        "previous_hash": previous_hash,
        "transaction": active_transactions,
        "target_prefix": current_target_prefix,
    }
