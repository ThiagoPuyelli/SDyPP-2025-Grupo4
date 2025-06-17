import asyncio
from datetime import datetime, timedelta, timezone
import time
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, CoordinatorState
from state import blockchain, pending_transactions, active_transactions, received_chains, current_phase
from utils import adjust_difficulty
import state
from log_config import logger

def scheduler():
    logger.info("🚀 Iniciando coordinador... esperando próximo ciclo de reloj")
    current_phase = get_last_interval_start()

    while True:
        if state.cicle_state == CoordinatorState.UNSET:
            #cuando llega a la fase nueva:
            prox_intervalo = get_last_interval_start()
            if (prox_intervalo > current_phase):
                state.cicle_state = CoordinatorState.GIVING_TASKS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name} - Comenzando fase a {current_phase}")
        
        elif state.cicle_state == CoordinatorState.GIVING_TASKS:
            now = datetime.now(timezone.utc)
            prox_intervalo = current_phase + timedelta(seconds=INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)
            if (now > prox_intervalo):
                state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name} - Comenzando fase a {current_phase}")

        elif state.cicle_state == CoordinatorState.OPEN_TO_RESULTS:
            now = datetime.now(timezone.utc)
            prox_intervalo = get_last_interval_start()
            if (prox_intervalo > current_phase):
                state.cicle_state = CoordinatorState.SELECTING_WINNER
                logger.info(f"[State] {state.cicle_state.name} - Comenzando fase a {prox_intervalo}")
                handle_selecting_winner()
                
                state.cicle_state = CoordinatorState.GIVING_TASKS
                current_phase = prox_intervalo
                logger.info(f"[State] {state.cicle_state.name} - Comenzando fase a {current_phase}")
        logger.info(f"[State] {state.cicle_state.name} - Comenzando fase a {current_phase}")
        time.sleep(1)

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // INTERVAL_DURATION) * INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)

def handle_selecting_winner():
    best_chain = max(received_chains, key=len, default=None)
    if best_chain:
        blockchain.extend(best_chain)
        adjust_difficulty()
        logger.info("✔️ Cadena aceptada")
    else:
        logger.info("⚠️ No se recibió cadena válida")

    active_transactions.clear()
    active_transactions.extend(pending_transactions)
    pending_transactions.clear()
    received_chains.clear()

    logger.info("### Fin de ciclo ###\n")