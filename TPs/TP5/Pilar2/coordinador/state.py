from enum import Enum
from services.database_service import LocalBlockchainDatabase, LocalReceivedChainsDatabase, RedisBlockchainDatabase, RedisReceivedChainsDatabase, RedisActiveTransactions
from services.queue_service import LocalTransactionQueue, RabbitTransactionQueue
import os
import redis

class CoordinatorState(str, Enum):
    UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    SELECTING_WINNER = "selecting winner and rewarding"

current_target_prefix = "0000"
next_target_prefix = "0000"
cicle_state = CoordinatorState.UNSET

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)

pending_transactions = RabbitTransactionQueue(
    queue_name="pending_transactions",
    host=os.getenv("RABBIT_HOST", "localhost"),
    port=int(os.getenv("RABBIT_PORT", 5672)),
    user=os.getenv("RABBIT_USER", "user"),
    password=os.getenv("RABBIT_PASS", "pass")
)
active_transactions = RedisActiveTransactions(redis_client)
blockchain = RedisBlockchainDatabase(redis_client)
received_chains = RedisReceivedChainsDatabase(redis_client)
