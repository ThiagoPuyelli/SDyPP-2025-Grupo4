import asyncio
from datetime import datetime, timedelta
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, CoordinatorState
from state import blockchain, pending_transactions, active_transactions, received_chains
from utils import adjust_difficulty
import state
from log_config import logger


async def coordinator_loop():
    logger.info("🚀 Iniciando coordinador")
    await wait_until_first_interval()

    state.cicle_state = CoordinatorState.GIVING_TASKS
    state.phase_started_at = datetime.utcnow()
    logger.info(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")

    while True:
        now = datetime.utcnow()
        elapsed = (now - state.phase_started_at).total_seconds()

        if state.cicle_state == CoordinatorState.GIVING_TASKS:
            duration = INTERVAL_DURATION - AWAIT_RESPONSE_DURATION
            if elapsed >= duration:
                state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
                state.phase_started_at = now
                logger.info(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")

        elif state.cicle_state == CoordinatorState.OPEN_TO_RESULTS:
            if elapsed >= AWAIT_RESPONSE_DURATION:
                state.cicle_state = CoordinatorState.SELECTING_WINNER
                state.phase_started_at = now
                logger.info(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")
                await handle_selecting_winner_phase()

                # Sin delay: pasamos inmediatamente a GIVING_TASKS
                state.cicle_state = CoordinatorState.GIVING_TASKS
                state.phase_started_at = datetime.utcnow()
                logger.info(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")

        await asyncio.sleep(1)

async def wait_until_first_interval():
    from utils import seconds_until_next_interval
    seconds = seconds_until_next_interval()
    logger.info(f"[Sync] Esperando {seconds:.1f}s hasta el primer ciclo")
    await asyncio.sleep(seconds)

async def handle_selecting_winner_phase():
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

    logger.info("### Fin de ciclo ###")