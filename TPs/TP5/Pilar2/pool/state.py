from enum import Enum
from models import MinedChain
from services.secret_service import get_secret
from services.subscribers import RedisSubscribers
from fastapi import WebSocket
from prize_handler import PrizeHandler
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
    db=int(get_secret("REDIS_DB", "0")),
    password=get_secret("REDIS_PASSWORD"),
    decode_responses=True,
)

mineros_activos = RedisSubscribers(redis_client)

prize_handler = PrizeHandler(redis_client)
