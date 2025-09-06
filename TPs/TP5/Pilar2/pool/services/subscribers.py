from typing import List, Union
from abc import ABC, abstractmethod
from models import MinedBlock, MinedChain, Miner
import redis
import json

class Subscribers(ABC):
    @abstractmethod
    def validar_mineros(self):
        pass
    
    @abstractmethod
    def agregar_minero(self):
        pass

class LocalSubscribers(Subscribers):
    def __init__(self):
        self.mineros = List[Miner]
    
    def validar_mineros(self):
        pass

    def agregar_minero(self):
        pass
    
# class RedisSubscribers(Subscribers):
#     def __init__(self, redis_client: redis.Redis):
#         self.r = redis_client
#         self.key_chain = "blockchain:chain"

#     def append_block(self, block: MinedBlock) -> None:
#         self.r.rpush(self.key_chain, block.model_dump_json())
#         self.r.set(f"blockchain:block:{block.hash}", block.model_dump_json())

#     def extend(self, new_blocks: Union[MinedChain, List[MinedBlock]]) -> None:
#         blocks = new_blocks.blocks if isinstance(new_blocks, MinedChain) else new_blocks
#         for block in blocks:
#             self.append_block(block)

#     def get_last_block(self) -> Union[MinedBlock, None]:
#         last = self.r.lindex(self.key_chain, -1)
#         return MinedBlock.model_validate(json.loads(last)) if last else None

#     def get_chain(self) -> MinedChain:
#         raw_blocks = self.r.lrange(self.key_chain, 0, -1)
#         return MinedChain(blocks=[MinedBlock.model_validate(json.loads(b)) for b in raw_blocks])

#     def replace_chain(self, new_chain: MinedChain) -> None:
#         self.clear()
#         self.extend(new_chain)

#     def clear(self) -> None:
#         self.r.delete(self.key_chain)

#     def is_empty(self) -> bool:
#         return self.r.llen(self.key_chain) == 0

#     def get_block(self, block_hash: str) -> Union[MinedBlock, None]:
#         raw = self.r.get(f"blockchain:block:{block_hash}")
#         return MinedBlock.model_validate(json.loads(raw)) if raw else None
    