from enum import Enum
import json
from typing import List, Tuple
import redis
import time
import requests
import config
from models import MinedChain, MinedBlock, Transaction, Miner
import base64
from config import BLOCKCHAIN_PRIZE_AMOUNT, BASE_REWARD_PERCENTAGE, PRIZE_DECIMALS
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from log_config import logger
from decimal import Decimal, ROUND_DOWN

def floor_n(value: float, n: int) -> float:
    q = Decimal('1.' + '0' * n)
    return float(
        Decimal(str(value)).quantize(q, rounding=ROUND_DOWN)
    )

class MinadasRepository:

    def __init__(self, redis_client: redis.Redis, redis_key: str = "blockchain:minadas"):
        self.redis = redis_client
        self.redis_key = redis_key

    def add(self, chain: MinedChain, miners: List[Miner]) -> None:
        payload = {
            "chain": chain.model_dump(),
            "miners": [m.model_dump() for m in miners],
        }
        self.redis.rpush(self.redis_key, json.dumps(payload))

    def get_all(self) -> List[Tuple[MinedChain, List[Miner]]]:
        raw_items = self.redis.lrange(self.redis_key, 0, -1)
        result: List[Tuple[MinedChain, List[Miner]]] = []

        for item in raw_items:
            payload = json.loads(item)
            chain = MinedChain(**payload["chain"])
            miners = [Miner(**m) for m in payload.get("miners", [])]
            result.append((chain, miners))

        return result

    def remove(self, chain: MinedChain) -> None:
        raw_items = self.redis.lrange(self.redis_key, 0, -1)

        target_chain = chain.model_dump()

        for item in raw_items:
            payload = json.loads(item)
            if payload.get("chain") == target_chain:
                self.redis.lrem(self.redis_key, 1, item)
                break

    def clear(self) -> None:
        self.redis.delete(self.redis_key)


class PrizeHandler:
    class BlockInChainStatus(Enum):
        NOT_IN_CHAIN = 0
        IN_CHAIN_OTHER_MINER = 1
        IN_CHAIN_POOL = 2

    def __init__(self, minadas_repo: MinadasRepository):
        self.minadas = minadas_repo

    def guardar_cadena_entregada(self, chain: MinedChain, miners: List[Miner]) -> None:
        self.minadas.add(chain, miners)

    def entregar_premios(self) -> None:
        cadenas = self.minadas.get_all()
        try:
            blockchain = self._obtener_blockchain_actual()
        except Exception as e:
            raise e

        for cadena, miners in cadenas:
            state = self._is_block_en_chain(cadena.blocks[0], blockchain)

            if not state == self.BlockInChainStatus.NOT_IN_CHAIN: 
                self.minadas.remove(cadena)
                
            if state == self.BlockInChainStatus.IN_CHAIN_POOL: 
                self._repartir_premio(miners, BLOCKCHAIN_PRIZE_AMOUNT)
                
    
    def _obtener_blockchain_actual(self) -> MinedChain:
        c = 0
        while True:
            try:
                response = requests.get(config.URI + '/chain', timeout=5)
                if response.ok:
                    data = response.json()
                    return MinedChain(
                        blocks=[MinedBlock(**b) for b in data["blocks"]]
                    )
                else:
                    raise Exception("Error al conectar con el coordinador")
            except requests.exceptions.RequestException:
                if c > 3:
                    raise Exception("Error al conectar con el coordinador")
                c += 1
                time.sleep(3)

    def _is_block_en_chain(self, block: MinedBlock, chain: MinedChain) -> BlockInChainStatus:
        for chain_block in chain.blocks:
            if (chain_block.previous_hash == block.previous_hash and
            chain_block.nonce == block.nonce and
            chain_block.miner_id == block.miner_id and
            chain_block.hash == block.hash and
            chain_block.transaction.source == block.transaction.source and
            chain_block.transaction.target == block.transaction.target and
            chain_block.transaction.amount == block.transaction.amount and
            chain_block.transaction.sign == block.transaction.sign):

                if block.miner_id == config.POOL_ID:
                    return self.BlockInChainStatus.IN_CHAIN_POOL
                else: 
                    return self.BlockInChainStatus.IN_CHAIN_OTHER_MINER

        return self.BlockInChainStatus.NOT_IN_CHAIN
                
    def _repartir_premio(self, miners: List[Miner], amount: float) -> None:
        total_shares = sum(miner.share_count for miner in miners)

        for miner in miners:
            prize_amount_raw = float(amount * (((1 - BASE_REWARD_PERCENTAGE) * miner.share_count / total_shares) + 
                                           (BASE_REWARD_PERCENTAGE / len(miners))))
            prize_amount = floor_n(prize_amount_raw, PRIZE_DECIMALS)
            tx = Transaction (
                source= config.POOL_ID,
                target= miner.id,
                amount= prize_amount,
                timestamp= "string",
                sign= self._sign_transaction(config.POOL_PK, config.POOL_ID, miner.id, prize_amount, "string"),
            )
            logger.info(f"Enviando transaccion al coordinador: {json.dumps(tx.model_dump(), ensure_ascii=False)}")
            while True:
                try:
                    response = requests.post(config.URI + '/tasks', json=tx.model_dump(), timeout=5)
                    if response.status_code == 200:
                        break
                    else:
                        pass
                except requests.exceptions.RequestException as e:
                    pass
                time.sleep(3)

    def _sign_transaction(self, 
        source_private_pem: str,
        source_public_pem: str,
        target_public_pem: str,
        amount: float,
        timestamp: str
    ) -> str:

        private_key = serialization.load_pem_private_key(
            source_private_pem.encode(),
            password=None
        )

        message = (
            f"{source_public_pem}|"
            f"{target_public_pem}|"
            f"{float(amount)}|"
            f"{timestamp}"
        ).encode()

        signature = private_key.sign(
            message,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode()