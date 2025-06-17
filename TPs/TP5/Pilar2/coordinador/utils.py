import json
import hashlib
from datetime import datetime, timezone
from config import BLOCK_TARGET_TIME, ACCEPTED_ALGORITHM, INTERVAL_DURATION
from state import blockchain, current_target_prefix 
from typing import Optional
from models import MinedBlock, Transaction

def compute_hash(block_data: dict) -> str:
    block_string = json.dumps(block_data, sort_keys=True).encode()
    return hashlib.new(ACCEPTED_ALGORITHM, block_string).hexdigest()

def is_valid_hash(h: str) -> bool:
    global current_target_prefix
    return h.startswith(current_target_prefix)

def adjust_difficulty():
    # global current_target_prefix
    # if len(blockchain) < 2:
    #     return
    # t1 = datetime.fromisoformat(blockchain[-1]["timestamp"])
    # t0 = datetime.fromisoformat(blockchain[-2]["timestamp"])
    # delta = (t1 - t0).total_seconds()
    # if delta < BLOCK_TARGET_TIME / 2:
    #     current_target_prefix = "00000"
    # elif delta > BLOCK_TARGET_TIME * 2:
    #     current_target_prefix = "000"
    pass

def seconds_until_next_interval(interval_minutes: int = INTERVAL_DURATION // 60) -> float:
    now = datetime.now(timezone.utc)
    minutes = now.hour * 60 + now.minute
    next_minutes = ((minutes // interval_minutes) + 1) * interval_minutes
    delta_minutes = next_minutes - minutes
    delta_seconds = delta_minutes * 60 - now.second - now.microsecond / 1_000_000
    return delta_seconds

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // INTERVAL_DURATION) * INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)

def create_genesis_block() -> Optional[MinedBlock]:
    if blockchain.is_empty():
        genesis_transaction = Transaction(
            source="0" * 64,
            target="0" * 64,
            amount=0,
            timestamp=datetime(2001, 9, 27).isoformat(), 
            sign="0" * 128
        )
        
        genesis_block = MinedBlock(
            previous_hash="0" * 64,
            transaction=genesis_transaction,
            nonce=0,
            miner_id="0" * 64,
            hash="0" * 128
        )
        
        blockchain.append_block(genesis_block)
        return genesis_block
    return None