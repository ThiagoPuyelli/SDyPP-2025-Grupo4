from typing import List, Union
from abc import ABC, abstractmethod
from models import MinedBlock, MinedChain

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

class LocalReceivedChainsDatabase(ReceivedChainsDatabase):
    def __init__(self):
        self._chains: List[MinedChain] = []
    
    def add_chain(self, chain: MinedChain) -> None:
        self._chains.append(chain)
    
    def get_all_chains(self) -> List[MinedChain]:
        return self._chains.copy()
    
    def clear(self) -> None:
        self._chains.clear()