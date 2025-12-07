from typing import List
from abc import ABC, abstractmethod
from models import Miner

class Subscribers(ABC):
    @abstractmethod
    def validar_minero(self, miner: Miner) -> bool:
        pass
    
    @abstractmethod
    def agregar_minero(self, miner: Miner):
        pass

    @abstractmethod
    def get_all_miners(self) -> List[Miner]:
        pass

    @abstractmethod
    def eliminar_minero(self, miner: Miner) -> None:
        pass

class LocalSubscribers(Subscribers):
    def __init__(self):
        self.mineros: List[Miner] = []
    
    def validar_minero(self, miner: Miner) -> bool:
        for m in self.mineros:
            if m.id == miner.id:
                return True
        return False

    def agregar_minero(self, miner: Miner):
        self.mineros.append(miner)

    def get_all_miners(self) -> List[Miner]:
        return self.mineros
    
    def eliminar_minero(self, miner: Miner) -> None:
        self.mineros = [m for m in self.mineros if m.id != miner.id]
    
# class RedisSubscribers(Subscribers):
    