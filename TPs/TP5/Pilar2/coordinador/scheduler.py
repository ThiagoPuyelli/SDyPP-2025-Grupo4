import asyncio
from datetime import datetime, timedelta
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, CoordinatorState
from state import blockchain, pending_transactions, active_transactions, received_chains
from utils import adjust_difficulty
import state
from log_config import logger


async def coordinator_loop():
    logger.info("🚀 Iniciando coordinador... esperando próximo ciclo de reloj")

    # Espera hasta el inicio del próximo intervalo alineado
    next_start = get_last_interval_start() + timedelta(seconds=INTERVAL_DURATION)
    next_state = CoordinatorState.GIVING_TASKS
    wait = (next_start - datetime.utcnow()).total_seconds()
    logger.info(f"⌛ Esperando {wait:.2f}s hasta el inicio de {next_state.name}")
    await asyncio.sleep(max(0, wait))

    while True:
        # Establecer nuevo estado y tiempo
        state.cicle_state = next_state
        state.phase_started_at = next_start
        logger.info(f"[State] {state.cicle_state.name} - Comenzando fase a {next_start}")

        # Ejecutar lógica si es necesario
        if next_state == CoordinatorState.SELECTING_WINNER:
            await handle_selecting_winner()

        # Calcular próxima fase
        next_start, next_state = get_next_phase_start_and_state()
        wait = (next_start - datetime.utcnow()).total_seconds()
        logger.info(f"🕒 Próxima fase: {next_state.name} comienza en {wait:.2f}s")
        await asyncio.sleep(max(0, wait))

def get_next_phase_start_and_state(now: datetime = None):
    if now is None:
        now = datetime.utcnow()

    interval_start = get_last_interval_start(now)
    giving_tasks_end = interval_start + timedelta(seconds=INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)
    open_to_results_end = interval_start + timedelta(seconds=INTERVAL_DURATION)

    if now < giving_tasks_end:
        next_start = giving_tasks_end
        next_state = CoordinatorState.OPEN_TO_RESULTS
    elif now < open_to_results_end:
        next_start = open_to_results_end
        next_state = CoordinatorState.SELECTING_WINNER
    else:
        next_start = interval_start + timedelta(seconds=INTERVAL_DURATION)
        next_state = CoordinatorState.GIVING_TASKS

    # Evitar que next_start sea igual o anterior a now, para que siempre avance
    if next_start <= now:
        next_start += timedelta(seconds=INTERVAL_DURATION)
        # Estado siguiente al que teníamos antes
        if next_state == CoordinatorState.GIVING_TASKS:
            next_state = CoordinatorState.OPEN_TO_RESULTS
        elif next_state == CoordinatorState.OPEN_TO_RESULTS:
            next_state = CoordinatorState.SELECTING_WINNER
        else:
            next_state = CoordinatorState.GIVING_TASKS

    return next_start, next_state

def get_last_interval_start(now: datetime = None) -> datetime:
    if now is None:
        now = datetime.utcnow()
    total_minutes = now.hour * 60 + now.minute
    interval_minutes = INTERVAL_DURATION // 60
    current_interval = (total_minutes // interval_minutes) * interval_minutes
    hour = current_interval // 60
    minute = current_interval % 60
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

async def wait_until_first_interval():
    from utils import seconds_until_next_interval
    seconds = seconds_until_next_interval()
    logger.info(f"[Sync] Esperando {seconds:.1f}s hasta el primer ciclo")
    await asyncio.sleep(seconds)

async def handle_selecting_winner():
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