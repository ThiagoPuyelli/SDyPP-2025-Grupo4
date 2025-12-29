from typing import List
from abc import ABC, abstractmethod
from models import Miner

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

# class RedisSubscribers(Subscribers):
    