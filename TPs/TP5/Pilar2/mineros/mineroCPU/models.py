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

class ActiveTransaction(BaseModel):
    transaction: Transaction
    ttl: int = 0