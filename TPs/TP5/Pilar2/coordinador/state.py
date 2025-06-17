from typing import List
from enum import Enum

class CoordinatorState(str, Enum):
    UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    SELECTING_WINNER = "selecting winner and rewarding"

current_target_prefix = "0000"
cicle_state = CoordinatorState.UNSET
current_phase = None
pending_transactions: List[dict] = []
active_transactions: List[dict] = []
blockchain: List[dict] = []
received_chains: List[List[dict]] = []
