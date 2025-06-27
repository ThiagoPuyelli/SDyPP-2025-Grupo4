import state
from mineroN.fuerzaBruta import conseguirHash
from log_config import logger
import config
import time
from models import MinedBlock

def minar(data, detener_mineria):
    start = time.perf_counter()
    transactions = data["transaction"]
    previousHash = data["previous_hash"]
    prefix = data["target_prefix"]
    for t in transactions:
        if detener_mineria.is_set(): 
            end = time.perf_counter()
            logger.info(f"TIEMPO TOTAL DE MINADO: {end - start:.2f} segundos")
            return
        numero_encontrado, hash_resultado = conseguirHash(
            prefix, 
            f"{previousHash} {t['source']} {t['target']} {t['amount']} {t['timestamp']} {t['sign']} {config.MINER_ID}", 
            0, 
            1000000000,
            detener_mineria)
        logger.info(f"{numero_encontrado} {hash_resultado}")
        if numero_encontrado:
            nuevo_bloque = MinedBlock(
                previous_hash=previousHash,
                transaction=t,
                nonce=numero_encontrado,
                miner_id=config.MINER_ID,
                hash=hash_resultado
            )
            state.mined_blocks.blocks.append(nuevo_bloque)
            previousHash = hash_resultado
            logger.info(f"Transaccion minada, bloque: {nuevo_bloque}")
    if detener_mineria.is_set():
        logger.info("Se termino de minar sin ser interrumpido")
    end = time.perf_counter()
    logger.info(f"TIEMPO TOTAL DE MINADO: {end - start:.2f} segundos")
        
        