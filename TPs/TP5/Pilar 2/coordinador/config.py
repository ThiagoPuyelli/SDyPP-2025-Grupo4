from enum import Enum

class CoordinatorState(str, Enum):
    UNSET = "server starting, only accepting transactions to queue"
    GIVING_TASKS = "expecting miners to request tasks"
    OPEN_TO_RESULTS = "accepting results from miners"
    SELECTING_WINNER = "selecting winner and rewarding"

INTERVAL_DURATION = 20 * 60
AWAIT_RESPONSE_DURATION = 1 * 60
MAX_MINING_ATTEMPTS = 3
BLOCK_TARGET_TIME = 2 * 60
ACCEPTED_ALGORITHM = "md5"
