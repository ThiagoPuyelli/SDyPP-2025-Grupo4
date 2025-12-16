import json
from typing import List
import redis
import time
import requests
import config
from models import MinedChain, MinedBlock, Transaction
import base64
from config import BLOCKCHAIN_PRIZE_AMOUNT
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class MinadasRepository:

    def __init__(self, redis_client: redis.Redis, redis_key: str = "blockchain:minadas"):
        self.redis = redis_client
        self.redis_key = redis_key

    def add(self, chain: MinedChain, miners: List[str]) -> None:
        payload = {
            "chain": chain.model_dump(),
            "miners": miners,
        }
        self.redis.rpush(self.redis_key, json.dumps(payload))

    def get_all(self) -> List[tuple[MinedChain, List[str]]]:
        raw_items = self.redis.lrange(self.redis_key, 0, -1)
        result: List[tuple[MinedChain, List[str]]] = []

        for item in raw_items:
            payload = json.loads(item)
            chain = MinedChain(**payload["chain"])
            miners = payload.get("miners", [])
            result.append((chain, miners))

        return result

    def remove(self, chain: MinedChain) -> None:
        raw_items = self.redis.lrange(self.redis_key, 0, -1)

        for item in raw_items:
            payload = json.loads(item)
            if payload.get("chain") == chain.model_dump():
                self.redis.lrem(self.redis_key, 1, item)
                break

    def clear(self) -> None:
        self.redis.delete(self.redis_key)



class PrizeHandler:

    def __init__(self, minadas_repo: MinadasRepository):
        self.minadas = minadas_repo

    def guardar_cadena_entregada(self, chain: MinedChain, miners: List[str]) -> None:
        self.minadas.add(chain, miners)

    def entregar_premios(self) -> None:
        cadenas = self.minadas.get_all()
        try:
            blockchain = self._obtener_blockchain_actual()
        except Exception as e:
            raise e

        for cadena, miners in cadenas:
            state = self._is_block_en_chain_miners(cadena.blocks[0], blockchain, miners)
            
            if state == 2: 
                self._repartir_premio(miners, BLOCKCHAIN_PRIZE_AMOUNT)
            if state == 1 or state == 2: 
                self.minadas.remove(cadena)
    
    def _obtener_blockchain_actual(self) -> MinedChain:
        c = 0
        while True:
            try:
                response = requests.get(config.URI + '/chain', timeout=5)
                if response.ok:
                    data = response.json()
                    return data["blocks"]
                else:
                    raise Exception("Error al conectar con el coordinador")
            except requests.exceptions.RequestException as e:
                if c > 3: raise Exception("Error al conectar con el coordinador")
                c += 1
                time.sleep(3)
    
    def _is_block_en_chain_miners(self, block: MinedBlock, chain: MinedChain, miners: List[str]) -> int:
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
                    return 2
                else: 
                    return 1
        
        return 0
                
    def _repartir_premio(self, miners: List[str], amount: float) -> None:
        amount_per_miner = float(amount / len(miners))
        
        for miner in miners:
            tx = Transaction (
                source= config.POOL_ID,
                target= miner,
                amount= amount_per_miner,
                timestamp= "string",
                sign= self._sign_transaction(config.POOL_PK, config.POOL_ID, miner, amount_per_miner, "string"),
            )
            while True:
                try:
                    response = requests.post(config.URI + '/tasks', json=tx, timeout=5)
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