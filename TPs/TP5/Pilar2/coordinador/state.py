from typing import List
from config import CoordinatorState

current_target_prefix = "0000"
cicle_state = CoordinatorState.UNSET
phase_started_at = None
pending_transactions: List[dict] = []
active_transactions: List[dict] = []
blockchain: List[dict] = []
received_chains: List[List[dict]] = []
