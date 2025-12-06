from fastapi import APIRouter #, HTTPException, Query
import state

router = APIRouter()


@router.post("/pool_block_mined")
async def stop_mining():
    state.finalizar_mineria_por_pool = True
    return {"status": "received"}