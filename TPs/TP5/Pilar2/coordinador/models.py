from pydantic import BaseModel
from typing import List

class Transaction(BaseModel):
    source: str
    target: str
    amount: float
    sign: str

class MinedBlock(BaseModel):
    timestamp: str
    previous_hash: str
    transaction: Transaction
    nonce: int
    miner_id: str
    hash: str
    signature: str

class MinedChain(BaseModel):
    blocks: List[MinedBlock]
