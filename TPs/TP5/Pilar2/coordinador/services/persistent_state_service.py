import json
from typing import Optional

import redis

class RedisPersistentState():
    def __init__(self, redis_client: redis.Redis, key: str = "persistent_state"):
        self.r = redis_client
        self.key = key
        self.last_cycle_summary_key = f"{key}:last_cycle_summary"

    def set_prefix(self, prefix: str) -> None:
        self.r.set(self.key, prefix)

    def get_prefix(self) -> str:
        return self.r.get(self.key)
    
    def init_prefix(self, default_prefix: str) -> str:
        created = self.r.setnx(self.key, default_prefix)
        if created:
            return default_prefix
        return self.get_prefix()

    def set_last_cycle_summary(self, summary: dict) -> None:
        self.r.set(self.last_cycle_summary_key, json.dumps(summary))

    def get_last_cycle_summary(self) -> Optional[dict]:
        raw = self.r.get(self.last_cycle_summary_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
