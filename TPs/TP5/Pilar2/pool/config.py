import os

INTERVAL_DURATION = 0
AWAIT_RESPONSE_DURATION = 0
MAX_MINING_ATTEMPTS = 0
ACCEPTED_ALGORITHM = ""
BLOCKCHAIN_PRIZE_AMOUNT = 10 - 1 # 1 es la comision del pool
BASE_REWARD_PERCENTAGE = 0.2  # 20% del premio se reparte equitativamente entre los mineros, sin importar velocidad de minado de cada uno
PRIZE_DECIMALS = 8
MAX_TRANSACTION_AGE_SECONDS = 600  # cantidad de segundos que espero hasta reenviar una transaccion de recompensa al coordinador (lo calculo cuando pido la config)
NONCE_MINER_RANGE = 100_000_000_000  # rango de nonces que doy a cada minero para minar cuando piden tareas


URI = os.getenv("HOST_URI", "http://35.244.137.250")
POOL_ID = os.getenv("POOL_ID", "9002")
POOL_PK = os.getenv("POOL_PK", "pk9002")
PROCESSING_TIER = os.getenv("PROCESSING_TIER", 1)

RABBIT_HOST = os.getenv("RABBIT_HOST", "35.185.40.225")
RABBIT_USER = os.getenv("RABBIT_USER", "blockchain")
RABBIT_PASS = os.getenv("RABBIT_PASS", "blockchain123")