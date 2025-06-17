from datetime import datetime, timedelta, timezone
import time
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION
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
                
                state.cicle_state = CoordinatorState.GIVING_TASKS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name}")

        time.sleep(1)

def handle_selecting_winner():
    best_chain = max(received_chains.get_all_chains(), key=len, default=None)
    if best_chain:
        blockchain.extend(best_chain)
        adjust_difficulty()
        logger.info("✔️ Cadena aceptada")
    else:
        logger.info("⚠️ No se recibió cadena válida")


    # TODO implementar logica de ttl
    active_transactions.clear()
    while (tx := pending_transactions.get()) is not None:
        active_transactions.put(tx)
    received_chains.clear()

    logger.info("### Fin de ciclo ###\n")