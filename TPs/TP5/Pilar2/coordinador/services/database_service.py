from typing import List, Optional, Union
from abc import ABC, abstractmethod
from models import ActiveTransaction, MinedBlock, MinedChain, Transaction
import redis
import json
import uuid

class BlockchainDatabase(ABC):
    @abstractmethod
    def append_block(self, block: MinedBlock) -> None:
        pass
    
    @abstractmethod
    def extend(self, new_blocks: Union[MinedChain, List[MinedBlock]]) -> bool:
        pass

    @abstractmethod
    def get_last_block(self) -> Union[MinedBlock, None]:
        pass

    @abstractmethod
    def get_chain(self) -> MinedChain:
        pass
    
    @abstractmethod
    def replace_chain(self, new_chain: MinedChain) -> None:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def get_block(self, block_hash: str) -> Union[MinedBlock, None]:
        """Devuelve un bloque por su hash, o None si no existe"""
        pass

    @abstractmethod
    def get_genesis(self) -> Union[MinedBlock, None]:
        """Devuelve el bloque génesis de la cadena"""
        pass

class ReceivedChainsDatabase(ABC):
    @abstractmethod
    def add_chain(self, chain: MinedChain) -> None:
        pass
    
    @abstractmethod
    def get_all_chains(self) -> List[MinedChain]:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

class ActiveTransactionsModel(ABC):
    @abstractmethod
    def put(self, transaction: ActiveTransaction) -> None:
        """Añade una transacción a la cola"""
        pass

    @abstractmethod
    def get(self) -> Optional[ActiveTransaction]:
        """Obtiene y remueve una transacción de la cola (FIFO)"""
        pass

    @abstractmethod
    def peek_all(self) -> List[Transaction]:
        """Devuelve solo las transacciones base (sin TTL)"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Devuelve el número de transacciones en la cola"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Limpia la cola"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Devuelve True si la cola está vacía"""
        pass

    @abstractmethod
    def get_all_transactions_with_ttl(self) -> List[ActiveTransaction]:
        """Devuelve las transacciones con TTL"""
        pass

class LocalBlockchainDatabase(BlockchainDatabase):
    def __init__(self):
        self._chain = MinedChain(blocks=[])
    
    def append_block(self, block: MinedBlock) -> bool:
        self._chain.blocks.append(block)

    def extend(self, new_blocks: Union[MinedChain, List[MinedBlock]]) -> None:
        blocks = new_blocks.blocks if isinstance(new_blocks, MinedChain) else new_blocks
        for block in blocks:
            self.append_block(block)

    def get_last_block(self) -> Union[MinedBlock, None]:
        return self._chain.blocks[-1] if self._chain.blocks else None

    def get_chain(self) -> MinedChain:
        return self._chain
    
    def replace_chain(self, new_chain: MinedChain) -> None:
        self._chain = new_chain
    
    def clear(self) -> None:
        self._chain = MinedChain(blocks=[])

    def is_empty(self) -> bool:
        return len(self._chain.blocks) == 0
    
    def get_block(self, block_hash: str) -> Union[MinedBlock, None]:
        for block in self._chain.blocks:
            if block.hash == block_hash:
                return block
        return None

class LocalReceivedChainsDatabase(ReceivedChainsDatabase):
    def __init__(self):
        self._chains: List[MinedChain] = []
    
    def add_chain(self, chain: MinedChain) -> None:
        self._chains.append(chain)
    
    def get_all_chains(self) -> List[MinedChain]:
        return self._chains.copy()
    
    def clear(self) -> None:
        self._chains.clear()

    def is_empty(self) -> bool:
        return len(self._chains) == 0
    
class RedisBlockchainDatabase(BlockchainDatabase):
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        self.key_chain = "blockchain:chain"

    def append_block(self, block: MinedBlock) -> None:
        self.r.rpush(self.key_chain, block.model_dump_json())
        self.r.set(f"blockchain:block:{block.hash}", block.model_dump_json())

    def extend(self, new_blocks: Union[MinedChain, List[MinedBlock]]) -> None:
        blocks = new_blocks.blocks if isinstance(new_blocks, MinedChain) else new_blocks
        for block in blocks:
            self.append_block(block)

    def get_last_block(self) -> Union[MinedBlock, None]:
        last = self.r.lindex(self.key_chain, -1)
        return MinedBlock.model_validate(json.loads(last)) if last else None

    def get_chain(self) -> MinedChain:
        raw_blocks = self.r.lrange(self.key_chain, 0, -1)
        return MinedChain(blocks=[MinedBlock.model_validate(json.loads(b)) for b in raw_blocks])

    def replace_chain(self, new_chain: MinedChain) -> None:
        self.clear()
        self.extend(new_chain)

    def clear(self) -> None:
        self.r.delete(self.key_chain)

    def is_empty(self) -> bool:
        return self.r.llen(self.key_chain) == 0

    def get_block(self, block_hash: str) -> Union[MinedBlock, None]:
        raw = self.r.get(f"blockchain:block:{block_hash}")
        return MinedBlock.model_validate(json.loads(raw)) if raw else None
    
    def get_genesis(self) -> Union[MinedBlock, None]:
        raw = self.r.lindex(self.key_chain, 0)
        return MinedBlock.model_validate(json.loads(raw)) if raw else None
    
class RedisReceivedChainsDatabase(ReceivedChainsDatabase):
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        self.key_prefix = "received_chain:"  # Cada cadena se guarda con un UUID único

    def add_chain(self, chain: MinedChain) -> None:
        chain_id = str(uuid.uuid4())
        self.r.set(f"{self.key_prefix}{chain_id}", chain.model_dump_json())

    def get_all_chains(self) -> List[MinedChain]:
        keys = self.r.keys(f"{self.key_prefix}*")
        chains = []
        for key in keys:
            raw = self.r.get(key)
            if raw:
                chains.append(MinedChain.model_validate(json.loads(raw)))
        return chains

    def clear(self) -> None:
        keys = self.r.keys(f"{self.key_prefix}*")
        if keys:
            self.r.delete(*keys)

    def is_empty(self) -> bool:
        return len(self.r.keys(f"{self.key_prefix}*")) == 0
    
class RedisTransactions(ActiveTransactionsModel):
    def __init__(self, redis_client: redis.Redis, key: str):
        self.r = redis_client
        self.key = key

    def put(self, transaction: ActiveTransaction) -> None:
        self.r.rpush(self.key, transaction.model_dump_json())

    def get(self) -> Optional[ActiveTransaction]:
        raw = self.r.lpop(self.key)
        if raw:
            return ActiveTransaction.model_validate(json.loads(raw))
        return None

    def peek_all(self) -> List[Transaction]:
        raw_items = self.r.lrange(self.key, 0, -1)
        return [ActiveTransaction.model_validate(json.loads(item)).transaction for item in raw_items]

    def get_all_transactions_with_ttl(self) -> List[ActiveTransaction]:
        raw_items = self.r.lrange(self.key, 0, -1)
        return [ActiveTransaction.model_validate(json.loads(item)) for item in raw_items]

    def size(self) -> int:
        return self.r.llen(self.key)

    def clear(self) -> None:
        self.r.delete(self.key)

    def is_empty(self) -> bool:
        return self.size() == 0