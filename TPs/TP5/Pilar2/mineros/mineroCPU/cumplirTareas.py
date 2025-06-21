import state
from fuerzaBruta import conseguirHash
from log_config import logger
from config import MINER_ID

def minar(data, detener_mineria):
    transactions = data["transaction"]
    previousHash = data["previous_hash"]
    prefix = data["target_prefix"]
    for t in transactions:
        if detener_mineria.is_set(): return
        numero_encontrado, hash_resultado = conseguirHash(prefix, f"{previousHash} {t.source} {t.target} {t.amount} {t.timestamp} {t.sign}", 0, 100000, detener_mineria)

        if numero_encontrado:
            state.mined_blocks.append({
                "previous_hash": previousHash,
                "transaction": t,
                "nonce": numero_encontrado,
                "miner_id": MINER_ID,
                "hash": hash_resultado
            })
            previousHash = hash_resultado
            logger.info(f"Transaccion minada: {t}")
    logger.info("Se termino de minar sin ser interrumpido")

# def cumplirTareas (expireTime: time):
    # response = requests.get(URI + '/tasks')

    # while not response.ok:
    #     print("Reintento de conexión con el coordinador")
    #     time.sleep(3)
    #     response = requests.get(URI + '/tasks')
    
    # print("Tareas obtenidas")
    # data = response.json()
    # print(response.json())
    # transactions = data["transaction"]
    # transactionsComplete = []
    # previousHash = data["previous_hash"]
    # prefix = data["target_prefix"]
    # for t in transactions:
    #     if datetime.now() <= expireTime:
    #         return transactionsComplete
        
    #     numero_encontrado, hash_resultado = conseguirHash(prefix, f"{previousHash} {t.source} {t.target} {t.amount} {t.timestamp} {t.sign}", 0, 100000)
    #     if numero_encontrado is None:
    #         return []
        
    #     transactionsComplete.append({
    #         "previous_hash": previousHash,
    #         "transaction": t,
    #         "nonce": numero_encontrado,
    #         "hash": hash_resultado
    #     })
    #     previousHash = hash_resultado
    #     print("Transaccion completada")
    # return transactionsComplete
        
        