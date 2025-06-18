# queue_service.py
from typing import Optional, List
from abc import ABC, abstractmethod
from models import ActiveTransaction, Transaction  # Reemplazado Transaction por ActiveTransaction

class BaseQueue(ABC):
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


# Implementación local (para desarrollo/testing)
class LocalTransactionQueue(BaseQueue):
    def __init__(self):
        self._queue: List[ActiveTransaction] = []

    def put(self, transaction: ActiveTransaction) -> None:
        self._queue.append(transaction)

    def get(self) -> Optional[ActiveTransaction]:
        return self._queue.pop(0) if self._queue else None

    def peek_all(self) -> List[Transaction]:
        return [active_tx.transaction for active_tx in self._queue]

    def size(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()

    def is_empty(self) -> bool:
        return len(self._queue) == 0
    
    def get_all_transactions_with_ttl(self) -> List[ActiveTransaction]:
        return self._queue.copy()


# Implementación para RabbitMQ (esqueleto)
class RabbitTransactionQueue(BaseQueue):
    def __init__(self, queue_name: str, connection_params: dict):
        self.queue_name = queue_name
        self.connection_params = connection_params
        # Configurar conexión a RabbitMQ aquí

    def put(self, transaction: ActiveTransaction) -> None:
        # Serializar la transacción y enviar a RabbitMQ
        pass

    def get(self) -> Optional[ActiveTransaction]:
        # Recibir y deserializar transacción desde RabbitMQ
        pass

    def peek_all(self) -> List[Transaction]:
        # Nota: Esto no es natural en RabbitMQ, podrías necesitar un diseño alternativo
        pass

    def size(self) -> int:
        # Obtener tamaño de la cola desde RabbitMQ
        pass

    def clear(self) -> None:
        # Limpiar la cola en RabbitMQ
        pass

    def is_empty(self) -> bool:
        # Devolver True si la cola está vacía
        pass

    def get_all_transactions_with_ttl(self) -> List[ActiveTransaction]:
        pass
