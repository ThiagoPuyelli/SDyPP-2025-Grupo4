# queue_service.py
from typing import Optional, List
from abc import ABC, abstractmethod
from models import ActiveTransaction, Transaction  # Reemplazado Transaction por ActiveTransaction
import pika
import json

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


# Implementación para RabbitMQ
class RabbitTransactionQueue(BaseQueue):
    def __init__(self, queue_name: str, host: str, port: int, user: str, password: str):
        self.queue_name = queue_name
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=host,
                port=port,
                credentials=pika.PlainCredentials(user, password)
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue_name, durable=True)

    def put(self, transaction: ActiveTransaction) -> None:
        body = json.dumps(transaction.model_dump())
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2)
        )

    def get(self) -> Optional[ActiveTransaction]:
        method_frame, header_frame, body = self.channel.basic_get(queue=self.queue_name)
        if body:
            self.channel.basic_ack(method_frame.delivery_tag)
            return ActiveTransaction.model_validate(json.loads(body))
        return None

    def peek_all(self) -> List[Transaction]:
        raise NotImplementedError("RabbitMQ no permite peek_all sin consumir mensajes.")

    def size(self) -> int:
        q = self.channel.queue_declare(queue=self.queue_name, passive=True)
        return q.method.message_count

    def clear(self) -> None:
        self.channel.queue_purge(queue=self.queue_name)

    def is_empty(self) -> bool:
        return self.size() == 0

    def get_all_transactions_with_ttl(self) -> List[ActiveTransaction]:
        raise NotImplementedError("No se puede recuperar todos los elementos sin consumirlos.")
