import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import config
from state import CoordinatorState, blockchain
from typing import Optional
from models import MinedBlock, Transaction
import state
from log_config import logger
import os

def calcular_md5(texto):
    hash_md5 = hashlib.md5()
    hash_md5.update(texto.encode('utf-8'))
    return hash_md5.hexdigest()

def is_valid_hash(block, prefix):
    t = block.transaction
    cadena_base = f"{block.previous_hash} {t.source} {t.target} {t.amount} {t.timestamp} {t.sign} {block.miner_id}"
    cadena_completa = cadena_base + str(block.nonce)
    hash_calculado = calcular_md5(cadena_completa)
    
    if hash_calculado != block.hash:
        return False
    if not hash_calculado.startswith(prefix):
        return False
    return True

def adjust_difficulty():
    logger.info(f"Ajustando dificultad: {state.current_target_prefix} -> {state.next_target_prefix}")
    state.current_target_prefix = state.next_target_prefix

def ajustar_ceros(cadena, cantidad):
    if cantidad > 0:
        return cadena + '0' * cantidad
    elif cantidad < 0:
        return cadena[:cantidad]
    else:
        return cadena

def get_starting_phase(now) -> CoordinatorState:
    if now is None:
        now = datetime.now(timezone.utc)
    intervalo = get_last_interval_start(now)
    segundos = (now - intervalo).total_seconds()
    if segundos < config.INTERVAL_DURATION - config.AWAIT_RESPONSE_DURATION:
        return CoordinatorState.GIVING_TASKS
    else: 
        return CoordinatorState.OPEN_TO_RESULTS

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // config.INTERVAL_DURATION) * config.INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)

def create_genesis_block() -> Optional[MinedBlock]:
    if blockchain.is_empty():
        genesis_transaction = Transaction(
            source="0" * 1,
            target="0" * 1,
            amount=1000,
            timestamp=datetime.now(timezone.utc).isoformat(), 
            sign="0" * 1
        )
        
        genesis_block = MinedBlock(
            previous_hash="0" * 1,
            transaction=genesis_transaction,
            nonce=0,
            miner_id="0" * 1,
            hash="0" * 1,
            blockchain_config={
                "interval_duration": config.INTERVAL_DURATION,
                "await_response_duration": config.AWAIT_RESPONSE_DURATION,
                "max_mining_attempts": config.MAX_MINING_ATTEMPTS,
                "accepted_algorithm": config.ACCEPTED_ALGORITHM,
            }
        )

        cadena_base = f"{genesis_block.previous_hash} {genesis_transaction.source} {genesis_transaction.target} {genesis_transaction.amount} {genesis_transaction.timestamp} {genesis_transaction.sign} {genesis_block.miner_id}"
        cadena_completa = cadena_base + str(genesis_block.nonce) + f"{genesis_block.blockchain_config['interval_duration']} {genesis_block.blockchain_config['await_response_duration']} {genesis_block.blockchain_config['max_mining_attempts']} {genesis_block.blockchain_config['accepted_algorithm']}"
        hash_calculado = calcular_md5(cadena_completa)
        
        genesis_block.hash = hash_calculado

        blockchain.append_block(genesis_block)
        return genesis_block
    return None
    