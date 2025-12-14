from enum import Enum
from models import MinedChain
from services.subscribers import LocalSubscribers
from typing import Dict
from fastapi import WebSocket

class CoordinatorState(str, Enum):
    # UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    # SELECTING_WINNER = "selecting winner and rewarding"

mined_blocks = MinedChain(blocks=[])

mono_time = None

cant_transacciones_a_minar = 0

mineros_activos = LocalSubscribers()
conexiones_ws: Dict[str, WebSocket] = {}

queue_channel = None
rabbit_connection = None

# guardan la consulta de las tareas disponibles
tareas_disponibles = []
previous_hash = ""
prefix = ""

nonce_start = 0