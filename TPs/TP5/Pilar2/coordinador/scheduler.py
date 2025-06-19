from datetime import datetime, timedelta, timezone
import time
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, MAX_MINING_ATTEMPTS
from state import blockchain, pending_transactions, active_transactions, received_chains, current_phase, CoordinatorState
from utils import adjust_difficulty, get_last_interval_start, seconds_until_next_interval, create_genesis_block
import state
from log_config import logger

def scheduler():
    global current_phase

    create_genesis_block()

    logger.info(f"🚀 Iniciando coordinador... el primer ciclo comienza en {seconds_until_next_interval()}s")
    current_phase = get_last_interval_start()

    while True:
        if state.cicle_state == CoordinatorState.UNSET:
            prox_intervalo = get_last_interval_start()
            if (prox_intervalo > current_phase):
                state.cicle_state = CoordinatorState.GIVING_TASKS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name}")
        
        elif state.cicle_state == CoordinatorState.GIVING_TASKS:
            now = datetime.now(timezone.utc)
            prox_intervalo = current_phase + timedelta(seconds=INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)
            if (now > prox_intervalo):
                state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name}")

        elif state.cicle_state == CoordinatorState.OPEN_TO_RESULTS:
            now = datetime.now(timezone.utc)
            prox_intervalo = get_last_interval_start()
            if (prox_intervalo > current_phase):
                state.cicle_state = CoordinatorState.SELECTING_WINNER
                logger.info(f"[State] {state.cicle_state.name}")
                handle_selecting_winner()
                logger.info("### Fin de ciclo ###\n")
                
                state.cicle_state = CoordinatorState.GIVING_TASKS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name}")

        time.sleep(1)

def handle_selecting_winner():
    best_chain = max(received_chains.get_all_chains(), key=lambda c: len(c.blocks), default=None)

    if best_chain:
        blockchain.extend(best_chain.blocks)
        adjust_difficulty()
        logger.info("✔️ Cadena aceptada")
    else:
        logger.info("⚠️ No se recibió cadena válida")
        return

    mined_signatures = {block.transaction.sign for block in best_chain.blocks}

    # Procesar activas: pasar las no minadas a pending, actualizar TTL o descartarlas
    for active_tx in active_transactions.peek_all():
        tx = active_tx.transaction
        if tx.sign not in mined_signatures:
            active_tx.ttl += 1
            if active_tx.ttl <= MAX_MINING_ATTEMPTS:
                pending_transactions.put(active_tx) # mover a pendiente nuevamente
                logger.info(f"🔁 Reencolando {tx.sign} (TTL {active_tx.ttl})")
            else:
                logger.warning(f"❌ Transacción descartada por TTL: {tx.sign}")
        else:
            logger.info(f"✅ Transacción minada: {tx.sign}")

    # Limpiar las activas antiguas
    active_transactions.clear()

    # Mover todas las pendientes a activas para el próximo ciclo
    while (tx := pending_transactions.get()) is not None:
        active_transactions.put(tx)

    # borrar las pendientes
    pending_transactions.clear()

    received_chains.clear()