import requests
import time
from datetime import datetime
from utils.fuerzaBruta import conseguirHash
from config import URI

def cumplirTareas (expireTime: time):
    response = requests.get(URI + '/tasks')

    while not response.ok:
        print("Reintento de conexión con el coordinador")
        time.sleep(3)
        response = requests.get(URI + '/tasks')
    
    print("Tareas obtenidas")
    data = response.json()
    print(response.json())
    transactions = data["transaction"]
    transactionsComplete = []
    previousHash = data["previous_hash"]
    prefix = data["target_prefix"]
    for t in transactions:
        if datetime.now() <= expireTime:
            return transactionsComplete
        
        numero_encontrado, hash_resultado = conseguirHash(prefix, f"{previousHash} {t.source} {t.target} {t.amount} {t.timestamp} {t.sign}", 0, 100000)
        if numero_encontrado is None:
            return []
        
        transactionsComplete.append({
            "previous_hash": previousHash,
            "transaction": t,
            "nonce": numero_encontrado,
            "hash": hash_resultado
        })
        print("Transaccion completada")
    return transactionsComplete
        
        