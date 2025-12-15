import hashlib
from datetime import datetime, timezone
import config
from state import CoordinatorState, blockchain
from typing import Optional
from models import MinedBlock, Transaction
import state
from log_config import logger
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

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
    ct= state.active_transactions.size()
    if ct == 0:
        logger.info("No hay transacciones activas, no se ajusta la dificultad.")
        return
    no_minadas = (ct - len(state.received_chains.get_all_chains())) / ct   # [0 ; 1] todas minadas --- ninguna minada
    logger.info(f"Porcentaje de transacciones minadas este ciclo: {1-no_minadas}")
    if no_minadas > 0.8 and len(state.current_target_prefix) > 1:
        state.next_target_prefix = ajustar_ceros(state.current_target_prefix, -1)
        logger.info(f"Ajustando dificultad: {state.current_target_prefix} -> {state.next_target_prefix}")
    elif no_minadas < 0.2:
        state.next_target_prefix = ajustar_ceros(state.current_target_prefix, 1)
        logger.info(f"Ajustando dificultad: {state.current_target_prefix} -> {state.next_target_prefix}")
    else:
        logger.info(f"Dificultad se mantiene: {state.current_target_prefix}")
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

        public_key_pem = """-----BEGIN PUBLIC KEY-----
            MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAz9Kyr9Y4iuEt0G6nbP/W
            77CtZW523x4YxCHvmkeHde0X0M7NXATdPBT6gRJGsNQ40qS8ST1ku5k8ajqzyoL8
            IlV3rBFKxUJ9M+68djhZwU/VJyAOP1opj0ZBsM026ByDB+Z4/ifa1uYp9flRfH6u
            X+zh60ovKGgeHLGv36qoSJNgX455W7eWz/SvzqlJU3sy0ajU/2cNBGOEjqv+fTkH
            ymlWzJM/5ikzcrtVC8SXFpJdY/vCDWkZquCPQTRf2hFOb8kqZSYbamoyJdpMwytR
            WgIZ21oFx/p1yIi8f7AJ+7DvPBQLMiGU55P1ZDcG/fRUQxl2og+SuxoRJ2LGxwB2
            PwIDAQAB
            -----END PUBLIC KEY-----"""

        genesis_transaction = Transaction(
            source="0",
            target=public_key_pem,
            amount=10000.0,
            timestamp=datetime.now(timezone.utc).isoformat(), 
            sign="0"
        )
        
        genesis_block = MinedBlock(
            previous_hash="0" * 1,
            transaction=genesis_transaction,
            nonce=0,
            miner_id="0",
            hash="0",
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
    
def tx_signature(tx):
    return (
        tx.source,
        tx.target,
        tx.amount,
        tx.timestamp,
        tx.sign,
    )

def verify_tx_signature(tx) -> bool:
    if tx.source == "0":
        # transacción del sistema / coordinador
        return True

    try:
        public_key = serialization.load_pem_public_key(
            tx.source.encode()
        )
        logger.info("Verifying transaction signature:")
        logger.info(f"{tx.source}|{tx.target}|{tx.amount}|{tx.timestamp}")

        message = f"{tx.source}|{tx.target}|{tx.amount}|{tx.timestamp}".encode()
        logger.info(f"{message}")
        signature = base64.b64decode(tx.sign)

        public_key.verify(
            signature,
            message,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        return True

    except InvalidSignature:
        return False
    except Exception:
        return False
    

def has_sufficient_funds(tx: Transaction) -> bool:
    if tx.source == "0":
        return True

    balance = 0.0

    chain = state.blockchain.get_chain().blocks

    for block in reversed(chain):
        btx = block.transaction

        if btx.target == tx.source:
            balance += btx.amount

        if btx.source == tx.source:
            balance -= btx.amount

    return False
