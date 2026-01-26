import redis

class RedisPersistentState():
    def __init__(self, redis_client: redis.Redis, key: str = "persistent_state"):
        self.r = redis_client
        self.key = key

    def set_prefix(self, prefix: str) -> None:
        self.r.set(self.key, prefix)

    def get_prefix(self) -> str:
        return self.r.get(self.key)
    
    def init_prefix(self, default_prefix: str) -> str:
        created = self.r.setnx(self.key, default_prefix)
        if created:
            return default_prefix
        return self.get_prefix()