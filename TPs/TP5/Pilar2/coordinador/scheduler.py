import time
from config import MAX_MINING_ATTEMPTS
from state import CoordinatorState
from utils import adjust_difficulty, create_genesis_block, get_starting_phase
import state
from log_config import setup_logger_con_monotonic, logger
from monotonic import mono_time

def scheduler():
    global logger

    if state.blockchain.is_empty:
        create_genesis_block()

    logger = setup_logger_con_monotonic(mono_time.hora_inicio, mono_time.start_monotonic)

    state.cicle_state = get_starting_phase(mono_time.get_hora_actual())

    # borro bajo ciertas condiciones las received_chains al iniciar el servidor
    if (state.cicle_state == CoordinatorState.GIVING_TASKS and
    not state.received_chains.is_empty):
        handle_selecting_winner()
        logger.info(f"Primero calculamos el ultimo ganador")
    
    logger.info(f"🚀 Coordinador Iniciado... ")

    while True:

        hora_actual = mono_time.get_hora_actual()
        proximo_estado = get_starting_phase(hora_actual)
        
        if (state.cicle_state == CoordinatorState.GIVING_TASKS and 
        proximo_estado != CoordinatorState.GIVING_TASKS):
            state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
            logger.info(f"[STATE] {state.cicle_state.name}")

        elif (state.cicle_state == CoordinatorState.OPEN_TO_RESULTS and 
        proximo_estado != CoordinatorState.OPEN_TO_RESULTS):
            handle_selecting_winner()
            logger.info("### Fin de ciclo ###\n")
            state.cicle_state = CoordinatorState.GIVING_TASKS
            logger.info(f"[STATE] {state.cicle_state.name}")
        time.sleep(1)

def handle_selecting_winner():
    state.cicle_state = CoordinatorState.SELECTING_WINNER
    logger.info(f"[STATE] {state.cicle_state.name}")

    all_chains = state.received_chains.get_all_chains()
    logger.info(f"Cadenas recibidas: {all_chains}")

    best_chain = max(all_chains, key=lambda c: len(c.blocks), default=None)

    # LO PRIMERO QUE HACEMOS ES BORRAR RECEIVED_CHAINS, necesario para la gestion ante fallos,
        # ya que validaremos esta estructura cuando el servidor inicia
    state.received_chains.clear()
    mined_signatures = {}

    if best_chain:
        state.blockchain.extend(best_chain)
        adjust_difficulty()
        logger.info("✔️ Cadena aceptada")

        mined_signatures = {block.transaction.sign for block in best_chain.blocks}
    
    else:
        logger.info("⚠️ No se recibió cadena válida")


    # Procesar activas: pasar las no minadas a pending, actualizar TTL o descartarlas
    for active_tx in state.active_transactions.get_all_transactions_with_ttl():
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