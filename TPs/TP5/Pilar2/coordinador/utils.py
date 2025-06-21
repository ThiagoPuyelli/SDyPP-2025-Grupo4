import json
import hashlib
from datetime import datetime, timezone
from config import ACCEPTED_ALGORITHM, INTERVAL_DURATION, AWAIT_RESPONSE_DURATION
from state import CoordinatorState, blockchain, current_target_prefix 
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

def get_starting_phase(now) -> CoordinatorState:
    if now is None:
        now = datetime.now(timezone.utc)
    intervalo = get_last_interval_start(now)
    segundos = (now - intervalo).total_seconds()
    if segundos < INTERVAL_DURATION - AWAIT_RESPONSE_DURATION:
        return CoordinatorState.GIVING_TASKS
    else: 
        return CoordinatorState.OPEN_TO_RESULTS

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