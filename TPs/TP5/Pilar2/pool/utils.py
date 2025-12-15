from datetime import datetime, timezone
import time
import hashlib
from monotonic import MonotonicTime
from log_config import setup_logger_con_monotonic, logger
import state
from state import CoordinatorState
import state
import config
from models import ActiveTransaction
import requests
import pika.exceptions
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature


def conectar_rabbit():
    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=config.RABBIT_HOST,
                    credentials=pika.PlainCredentials(
                        config.RABBIT_USER,
                        config.RABBIT_PASS
                    ),
                    heartbeat=30,
                    blocked_connection_timeout=30
                )
            )
            channel = connection.channel()
            logger.info("Conectado a RabbitMQ")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"No se pudo conectar a RabbitMQ: {e}, reintentando...")
            time.sleep(5)

def publish_seguro(event):
    try:
        state.queue_channel.basic_publish(
            exchange="blockchain.exchange",
            routing_key="",
            body=json.dumps(event)
        )
    except pika.exceptions.AMQPError as e:
        logger.error(f"Error publicando en RabbitMQ: {e}, reconectando...")
        try:
            state.rabbit_connection.close()
        except Exception:
            pass

        state.rabbit_connection, state.queue_channel = conectar_rabbit()

        # reintento
        state.queue_channel.basic_publish(
            exchange="blockchain.exchange",
            routing_key="",
            body=json.dumps(event)
        )

def get_current_phase(now) -> CoordinatorState:
    if now == None:
        now = datetime.now(timezone.utc)
    intervalo = get_last_interval_start(now)
    segundos = (now - intervalo).total_seconds()
    if segundos < config.INTERVAL_DURATION - config.AWAIT_RESPONSE_DURATION:
        return CoordinatorState.GIVING_TASKS
    else: 
        return CoordinatorState.OPEN_TO_RESULTS

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // config.INTERVAL_DURATION) * config.INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)

def sync_con_coordinador():
    while True:
        try:
            response = requests.get(
                config.URI + '/state',
                timeout=5
            )
            if response.ok:
                data = response.json()
                state.mono_time = MonotonicTime(datetime.fromisoformat(data["server-date-time"]))
                break
            else:
                logger.info(f"Error HTTP {response.status_code}, reintentando...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la conexión con el coordinador: {e}. Reintentando...")

        time.sleep(3)

    # sincronizo el reloj del logger
    logger = setup_logger_con_monotonic(state.mono_time.hora_inicio, state.mono_time.start_monotonic)

def get_tareas():
    # obtengo tareas
    data = None
    try:
        while True:
            try:
                response = requests.get(config.URI + '/tasks', timeout=5)
                if response.ok:
                    break
                else:
                    logger.info(f"Respuesta inválida del coordinador: {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Error al conectar con el coordinador: {e}")
            
            logger.info("Reintento de conexión con el coordinador en 3 segundos...")
            time.sleep(3)
    except Exception as e:
        logger.error(f"Fallo crítico al obtener tareas: {e}")

    data = response.json()
    logger.info(f"Tareas recibidas: {data}")
    
    state.mined_blocks.blocks.clear()
    state.tareas_disponibles = []
    
    for tx in data["transaction"]:
        tr = ActiveTransaction(transaction=tx, mined=False)
        state.tareas_disponibles.append(tr)
    state.cant_transacciones_a_minar = len(state.tareas_disponibles)
    state.previous_hash = data["previous_hash"]
    state.prefix = data["target_prefix"]

def calcular_md5(texto):
    hash_md5 = hashlib.md5()
    hash_md5.update(texto.encode('utf-8'))
    return hash_md5.hexdigest()

def is_valid_hash(block, prefix):
    t = block.transaction
    cadena_base = f"{block.previous_hash} {t.source} {t.target} {t.amount} {t.timestamp} {t.sign} {block.miner_id}"
    cadena_completa = cadena_base + str(block.nonce)
    hash_calculado = calcular_md5(cadena_completa)
    
    if hash_calculado != block.hash:
        return False
    if not hash_calculado.startswith(prefix):
        return False
    return True


MAX_RETRIES = 3          # número máximo de intentos
RETRY_DELAY = 5          # segundos entre intentos

async def notify_single_miner(miner_id: str, message: str) -> bool:
    ws = state.conexiones_ws.get(miner_id)
    if not ws:
        logger.warning(f"No WS connection for {miner_id}")
        return False

    try:
        await ws.send_text(message)
        return True
    except Exception as e:
        logger.error(f"Error sending WS to {miner_id}: {e}")
        state.conexiones_ws.pop(miner_id, None)
        return False

async def notify_miners_new_block():
    to_remove = []

    for miner_id, ws in state.conexiones_ws.items():
        try:
            await ws.send_text("NEW_BLOCK")
        except:
            logger.error(f"Failed to notify {miner_id}")
            to_remove.append(miner_id)

    # remover desconectados
    for miner_id in to_remove:
        state.conexiones_ws.pop(miner_id, None)

def tx_signature(tx):
    return (
        tx.source,
        tx.target,
        tx.amount,
        tx.timestamp,
        tx.sign,
    )

def verify_tx_signature(tx) -> bool:
    if tx.source == "0":
        # transacción del sistema / coordinador
        return True

    try:
        public_key = serialization.load_pem_public_key(
            tx.source.encode()
        )

        message = f"{tx.source}|{tx.target}|{tx.amount}|{tx.timestamp}".encode()

        signature = base64.b64decode(tx.sign)

        public_key.verify(
            signature,
            message,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        return True

    except InvalidSignature:
        return False
    except Exception:
        return False