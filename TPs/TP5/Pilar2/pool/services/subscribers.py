from typing import List
from abc import ABC, abstractmethod
from models import Miner
import redis

class Subscribers(ABC):
    @abstractmethod
    def agregar_minero(self, miner: Miner):
        pass

    @abstractmethod
    def get_all_miners(self) -> List[Miner]:
        pass

    @abstractmethod
    def borrar_todos(self) -> None:
        pass

    @abstractmethod
    def share_recibido(self, miner_id: str) -> None:
        pass

class LocalSubscribers(Subscribers):
    def __init__(self):
        self.mineros: List[Miner] = [] 

    def agregar_minero(self, miner: Miner):
        self.mineros.append(miner)

    def get_all_miners(self) -> List[Miner]:
        return self.mineros
    
    def borrar_todos(self) -> None:
        self.mineros = []

    def share_recibido(self, miner_id: str) -> None:
        for miner in self.mineros:
            if miner.id == miner_id:
                miner.share_count += 1
                return
            
        miner = Miner(id=miner_id, share_count=1)
        self.agregar_minero(miner)

class RedisSubscribers(Subscribers):
    def __init__(self, redis_client: redis.Redis, redis_key: str = "blockchain:subscribers"):
        self.redis = redis_client
        self.redis_key = redis_key

    def agregar_minero(self, miner: Miner):
        # Si ya existe, no pisa el share_count
        self.redis.hsetnx(self.redis_key, miner.id, miner.share_count)

    def get_all_miners(self) -> List[Miner]:
        miners = []
        data = self.redis.hgetall(self.redis_key)

        for miner_id, share_count in data.items():
            miners.append(
                Miner(
                    id=miner_id.decode("utf-8"),
                    share_count=int(share_count)
                )
            )
        return miners

    def borrar_todos(self) -> None:
        self.redis.delete(self.redis_key)

    def share_recibido(self, miner_id: str) -> None:
        # Incremento atómico (si no existe, Redis lo crea con 0 y suma 1)
        self.redis.hincrby(self.redis_key, miner_id, 1)