from fastapi import APIRouter
import state

router = APIRouter()

@router.get("/chain")
def get_chain():
    return state.blockchain

@router.get("/state")
def get_state():
    return state.cicle_state.value
