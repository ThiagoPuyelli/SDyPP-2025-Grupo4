from enum import Enum
from models import MinedChain

class CoordinatorState(str, Enum):
    # UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    # SELECTING_WINNER = "selecting winner and rewarding"

mined_blocks = MinedChain(blocks=[])

mono_time = None

cant_transacciones_a_minar = 0

finalizar_mineria_por_pool = False

pool_id = None