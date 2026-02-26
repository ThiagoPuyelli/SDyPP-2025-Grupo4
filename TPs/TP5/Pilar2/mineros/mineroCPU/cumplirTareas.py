import state
from mineroN.fuerzaBruta import conseguirHash
from log_config import logger
import config
import time
from models import MinedBlock
from utils import calcular_md5

def minar(data, detener_mineria):
    pk_minado = state.pool_id if state.pool_id else config.MINER_ID

    # start = time.perf_counter()
    transactions = data["transaction"]
    state.cant_transacciones_a_minar = len(transactions)
    previousHash = data["previous_hash"]
    prefix = data["target_prefix"]
    nonce_start = data.get("nonce_start", None)
    nonce_end = data.get("nonce_end", None)
    for t in transactions:
        if detener_mineria.is_set(): 
            # end = time.perf_counter()
            # logger.info(f"TIEMPO TOTAL DE MINADO: {end - start:.2f} segundos")
            return
        hash_transaccion = calcular_md5(f"{t['source']} {t['target']} {t['amount']} {t['timestamp']} {t['sign']}")
        hash_pk_minero = calcular_md5(pk_minado)
        numero_encontrado, hash_resultado = conseguirHash(
            prefix, 
            # f"{previousHash} {t['source']} {t['target']} {t['amount']} {t['timestamp']} {t['sign']} {pk_minado}",
            f"{previousHash} {hash_transaccion} {hash_pk_minero}",
            nonce_start if nonce_start is not None else 0, 
            nonce_end if nonce_end is not None else 1000000000000000000, #1000000000,
            detener_mineria)
        # logger.info(f"{numero_encontrado} {hash_resultado}")
        if numero_encontrado:
            nuevo_bloque = MinedBlock(
                previous_hash=previousHash,
                transaction=t,
                nonce=numero_encontrado,
                miner_id=pk_minado,
                hash=hash_resultado
            )
            state.mined_blocks.blocks.append(nuevo_bloque)
            previousHash = hash_resultado
            # logger.info(f"Transaccion minada, bloque: {nuevo_bloque}")
    # if not detener_mineria.is_set():
        # logger.info("Se termino de minar sin ser interrumpido")
    # end = time.perf_counter()
    # logger.info(f"TIEMPO TOTAL DE MINADO: {end - start:.2f} segundos")
        
        