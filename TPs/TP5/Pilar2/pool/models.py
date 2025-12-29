from pydantic import BaseModel, Extra
from typing import List

class Transaction(BaseModel):
    source: str # pk del remitente
    target: str # pk del receptor
    amount: float # cantidad
    timestamp: str 
    sign: str # firma con la priv key del remitente

class MinedBlock(BaseModel):
    previous_hash: str
    transaction: Transaction
    nonce: int 
    miner_id: str # pk del minero
    hash: str
    class Config:
        extra = Extra.allow

class MinedChain(BaseModel):
    blocks: List[MinedBlock]

class Miner(BaseModel):
    id: str
    share_count: int
    # class Config:
    #     arbitrary_types_allowed = True

class MinersList(BaseModel):
    miners: List[Miner]

class ActiveTransaction(BaseModel):
    transaction: Transaction
    mined: bool = False
