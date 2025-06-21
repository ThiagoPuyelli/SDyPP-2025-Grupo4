from datetime import datetime, timedelta, timezone
import time
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, MAX_MINING_ATTEMPTS
from state import CoordinatorState
from utils import adjust_difficulty, get_last_interval_start, create_genesis_block, get_starting_phase
import state
from log_config import logger

def scheduler():

    if state.blockchain.is_empty:
        create_genesis_block()

    state.cicle_state = get_starting_phase()

    # borro bajo ciertas condiciones las received_chains al iniciar el servidor
    if (state.cicle_state == CoordinatorState.GIVING_TASKS and
    not state.received_chains.is_empty):
        handle_selecting_winner()

    state.current_phase = get_last_interval_start()
    
    logger.info(f"🚀 Coordinador Iniciado... ")

    while True:
        if state.cicle_state == CoordinatorState.UNSET:
            prox_intervalo = get_last_interval_start()
            if (prox_intervalo > state.current_phase):
                state.cicle_state = CoordinatorState.GIVING_TASKS
                state.current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name}")
        
        elif state.cicle_state == CoordinatorState.GIVING_TASKS:
            now = datetime.now(timezone.utc)
            prox_intervalo = state.current_phase + timedelta(seconds=INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)
            if (now > prox_intervalo):
                state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
                state.current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name}")

        elif state.cicle_state == CoordinatorState.OPEN_TO_RESULTS:
            now = datetime.now(timezone.utc)
            prox_intervalo = get_last_interval_start() + timedelta(seconds=INTERVAL_DURATION)
            if (now > prox_intervalo):
                handle_selecting_winner()

        logger.info(f"Estado: {state.cicle_state.name}")
        time.sleep(1)

def handle_selecting_winner():
    state.cicle_state = CoordinatorState.SELECTING_WINNER
    logger.info(f"[State] {state.cicle_state.name}")
    best_chain = max(state.received_chains.get_all_chains(), key=lambda c: len(c.blocks), default=None)

    # LO PRIMERO QUE HACEMOS ES BORRAR RECEIVED_CHAINS, necesario para la gestion ante fallos,
    # ya que validaremos esta estructura cuando el servidor inicia
    state.received_chains.clear()

    if best_chain:
        state.blockchain.extend(best_chain.blocks)
        adjust_difficulty()
        logger.info("✔️ Cadena aceptada")

        mined_signatures = {block.transaction.sign for block in best_chain.blocks}

        # Procesar activas: pasar las no minadas a pending, actualizar TTL o descartarlas
        for active_tx in state.active_transactions.peek_all():
            tx = active_tx.transaction
            if tx.sign not in mined_signatures:
                active_tx.ttl += 1
                if active_tx.ttl <= MAX_MINING_ATTEMPTS:
                    state.pending_transactions.put(active_tx) # mover a pendiente nuevamente
                    logger.info(f"🔁 Reencolando {tx.sign} (TTL {active_tx.ttl})")
                else:
                    logger.warning(f"❌ Transacción descartada por TTL: {tx.sign}")
            else:
                logger.info(f"✅ Transacción minada: {tx.sign}")

        # Limpiar las activas antiguas
        state.active_transactions.clear()

        # Mover todas las pendientes a activas para el próximo ciclo
        while (tx := state.pending_transactions.get()) is not None:
            state.active_transactions.put(tx)

        # borrar las pendientes
        state.pending_transactions.clear()
    
    else:
        logger.info("⚠️ No se recibió cadena válida")

    logger.info("### Fin de ciclo ###\n")
    state.cicle_state = CoordinatorState.GIVING_TASKS
    state.current_phase = get_last_interval_start()
    logger.info(f"[State] {state.cicle_state.name}")