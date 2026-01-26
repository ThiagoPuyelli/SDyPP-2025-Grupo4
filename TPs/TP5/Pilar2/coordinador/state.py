from enum import Enum
from typing import TYPE_CHECKING
from services.secret_service import get_secret
from services.database_service import RedisBlockchainDatabase, RedisReceivedChainsDatabase, RedisTransactions
from services.queue_service import RabbitTransactionQueue
import os
import redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError
from services.persistant_state_service import RedisPersistentState

class CoordinatorState(str, Enum):
    UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    SELECTING_WINNER = "selecting winner and rewarding"

cicle_state = CoordinatorState.UNSET

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(get_secret("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
    retry_on_error=[ConnectionError],
    retry=Retry(ExponentialBackoff(), retries=10)
)

# pending_transactions = RabbitTransactionQueue(
#     queue_name="pending_transactions",
#     host=os.getenv("RABBIT_HOST", "localhost"),
#     port=int(os.getenv("RABBIT_PORT", 5672)),
#     user=get_secret("RABBIT_USER", "user"),
#     password=get_secret("RABBIT_PASS", "pass")
# )

pending_transactions = RedisTransactions(redis_client, key="pending_transactions")
active_transactions = RedisTransactions(redis_client, key="active_transactions")
blockchain = RedisBlockchainDatabase(redis_client)
received_chains = RedisReceivedChainsDatabase(redis_client)
persistent_state = RedisPersistentState(redis_client)