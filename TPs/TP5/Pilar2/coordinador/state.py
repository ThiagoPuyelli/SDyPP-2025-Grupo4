from enum import Enum
from services.database_service import LocalBlockchainDatabase, LocalReceivedChainsDatabase
from services.queue_service import LocalTransactionQueue

class CoordinatorState(str, Enum):
    UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    SELECTING_WINNER = "selecting winner and rewarding"

current_target_prefix = "0000"
cicle_state = CoordinatorState.UNSET
current_phase = None
# pending_transactions: List[dict] = []
# active_transactions: List[dict] = []
pending_transactions = LocalTransactionQueue()
active_transactions = LocalTransactionQueue()
# blockchain: List[dict] = []
# received_chains: List[List[dict]] = []
blockchain = LocalBlockchainDatabase()
received_chains = LocalReceivedChainsDatabase()
