from enum import Enum
from models import MinedChain
from services.subscribers import LocalSubscribers
from typing import Dict
from fastapi import WebSocket
from prize_handler import PrizeHandler, MinadasRepository
import os
import redis

class CoordinatorState(str, Enum):
    # UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    # SELECTING_WINNER = "selecting winner and rewarding"

mined_blocks = MinedChain(blocks=[])

mono_time = None

cant_transacciones_a_minar = 0

mineros_activos = LocalSubscribers()

queue_channel = None
rabbit_connection = None

# guardan la consulta de las tareas disponibles
tareas_disponibles = []
previous_hash = ""
prefix = ""

nonce_start = 0

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    # db=int(get_secret("REDIS_DB", "0")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)
repo_cadenas_minadas = MinadasRepository(redis_client=redis_client)
prize_handler = PrizeHandler(repo_cadenas_minadas)