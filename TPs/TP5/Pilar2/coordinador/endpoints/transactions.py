from fastapi import APIRouter
from models import Transaction
import state

router = APIRouter()

@router.post("/tasks")
async def submit_transaction(tx: Transaction):
    state.pending_transactions.append(tx.dict())
    return {"status": "ok"}

@router.get("/tasks")
async def get_task():
    from utils import compute_hash

    if not state.blockchain:
        previous_hash = "0"
    else:
        last_block = state.blockchain[-1]
        block_data = {
            "timestamp": last_block["timestamp"],
            "previous_hash": last_block["previous_hash"],
            "transaction": last_block["transaction"],
            "nonce": last_block["nonce"]
        }
        previous_hash = compute_hash(block_data)

    if not state.active_transactions:
        return {"message": "no tasks"}

    return {
        "previous_hash": previous_hash,
        "transaction": state.active_transactions,
        "target_prefix": state.current_target_prefix,
    }
