import time
from datetime import datetime
from models import ActiveTransaction, Transaction
from config import MAX_MINING_ATTEMPTS, PRIZE_AMOUNT
from state import CoordinatorState
from utils import adjust_difficulty, create_genesis_block, get_starting_phase
import state
from log_config import setup_logger_con_monotonic, logger
from monotonic import mono_time
from metrics import update_queue_metrics, record_cycle, record_mining_duration

def scheduler():
    global logger

    if state.blockchain.is_empty:
        logger.info("Creando bloque génesis...")
        create_genesis_block()
    else:
        logger.info("Blockchain ya existente:")
        logger.info(state.blockchain.get_chain())

    logger = setup_logger_con_monotonic(mono_time.hora_inicio, mono_time.start_monotonic)

    state.cicle_state = get_starting_phase(mono_time.get_hora_actual())

    state.persistent_state.init_prefix("0000")

    # borro bajo ciertas condiciones las received_chains al iniciar el servidor
    if (state.cicle_state == CoordinatorState.GIVING_TASKS and
    not state.received_chains.is_empty):
        handle_selecting_winner()
        logger.info(f"Primero calculamos el ultimo ganador")
    
    logger.info(f"🚀 Coordinador Iniciado... ")

    while True:

        hora_actual = mono_time.get_hora_actual()
        proximo_estado = get_starting_phase(hora_actual)
        # Actualizamos gauges en cada iteración
        try:
            pending_size = state.pending_transactions.size()
        except Exception:
            pending_size = 0
        try:
            active_size = state.active_transactions.size()
        except Exception:
            active_size = 0
        try:
            received_size = len(state.received_chains.get_all_chains())
        except Exception:
            received_size = 0
        update_queue_metrics(pending_size, active_size, received_size)
        
        if (state.cicle_state == CoordinatorState.GIVING_TASKS and 
        proximo_estado != CoordinatorState.GIVING_TASKS):
            state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
            logger.info(f"[STATE] {state.cicle_state.name}")

        elif (state.cicle_state == CoordinatorState.OPEN_TO_RESULTS and 
        proximo_estado != CoordinatorState.OPEN_TO_RESULTS):
            if try_acquire_cycle_lock():
                handle_selecting_winner()
                record_cycle("completed")
                logger.info("### Fin de ciclo ###\n")
            else:
                logger.info(f"Otra replica ya está manejando el cierre del ciclo...")
            state.cicle_state = CoordinatorState.GIVING_TASKS
            logger.info(f"[STATE] {state.cicle_state.name}")
        time.sleep(1)

def handle_selecting_winner():
    state.cicle_state = CoordinatorState.SELECTING_WINNER
    logger.info(f"[STATE] {state.cicle_state.name}")
    cycle_closed_at = mono_time.get_hora_actual()
    cycle_prefix = state.persistent_state.get_prefix()

    all_received_chains = state.received_chains.get_all_received()
    all_chains = [entry.chain for entry in all_received_chains]
    logger.info(f"Cadenas recibidas: {all_chains}")

    best_received_chain = max(all_received_chains, key=lambda c: len(c.chain.blocks), default=None)
    best_chain = best_received_chain.chain if best_received_chain else None
    active_before_cycle = state.active_transactions.get_all_transactions_with_ttl()
    cycle_started_at = None
    started_at_values = [tx.mining_started_at for tx in active_before_cycle if tx.mining_started_at]
    if started_at_values:
        try:
            cycle_started_at = datetime.fromisoformat(min(started_at_values))
        except Exception:
            cycle_started_at = None

    # LO PRIMERO QUE HACEMOS ES BORRAR RECEIVED_CHAINS, necesario para la gestion ante fallos,
        # ya que validaremos esta estructura cuando el servidor inicia
    state.received_chains.clear()
    mined_signatures = set()
    mined_blocks = []
    requeued_transactions = []
    discarded_transactions = []
    mined_duration_samples = []
    winner_received_at = None
    winner_elapsed_seconds = None
    winner_avg_seconds_per_task = None

    if best_chain:
        state.blockchain.extend(best_chain)
        logger.info("✔️ Cadena aceptada")
        mined_blocks = best_chain.blocks
        
        # premio al ganador
        active_tx = ActiveTransaction(
            transaction=Transaction(
                source="0", #generado por el coordinador
                target=best_chain.blocks[0].miner_id,
                amount=PRIZE_AMOUNT, #premio fijo
                timestamp=mono_time.get_hora_actual().isoformat(),
                sign="0" #no se firma si es del coordinador
            )
        )
        state.pending_transactions.put(active_tx)
        
        mined_signatures = {block.transaction.sign for block in mined_blocks}
        if best_received_chain:
            winner_received_at = best_received_chain.received_at
            try:
                winner_received_at_dt = datetime.fromisoformat(winner_received_at)
            except Exception:
                winner_received_at_dt = None

            if cycle_started_at:
                if winner_received_at_dt:
                    winner_elapsed_seconds = (winner_received_at_dt - cycle_started_at).total_seconds()
                    if winner_elapsed_seconds < 0:
                        winner_elapsed_seconds = None

                if winner_elapsed_seconds is None:
                    winner_elapsed_seconds = (cycle_closed_at - cycle_started_at).total_seconds()

            if winner_elapsed_seconds and winner_elapsed_seconds > 0 and len(mined_blocks) > 0:
                winner_avg_seconds_per_task = winner_elapsed_seconds / len(mined_blocks)
                record_mining_duration(cycle_prefix, winner_avg_seconds_per_task)
                mined_duration_samples.append(
                    {
                        "mode": "winner_average",
                        "seconds_per_task": winner_avg_seconds_per_task,
                        "elapsed_seconds": winner_elapsed_seconds,
                        "mined_blocks": len(mined_blocks),
                        "prefix": cycle_prefix,
                        "winner_received_at": winner_received_at,
                    }
                )
                logger.info(
                    f"metric=mining_duration_avg_per_winner_task prefix={cycle_prefix} "
                    f"elapsed_seconds={winner_elapsed_seconds:.6f} mined_blocks={len(mined_blocks)} "
                    f"seconds_per_task={winner_avg_seconds_per_task:.6f}"
                )
    
    else:
        logger.info("⚠️ No se recibió cadena válida")


    adjust_difficulty(best_chain)

    # Procesar activas: pasar las no minadas a pending, actualizar TTL o descartarlas
    for active_tx in active_before_cycle:
        tx = active_tx.transaction
        if tx.sign not in mined_signatures:
            active_tx.ttl += 1
            if active_tx.ttl <= MAX_MINING_ATTEMPTS or tx.source == "0": # si es generada por la blockchain no deberia expirar
                state.pending_transactions.put(active_tx) # mover a pendiente nuevamente
                requeued_transactions.append(active_tx)
                logger.info(f"🔁 Reencolando {tx.sign} (TTL {active_tx.ttl})")
            else:
                discarded_transactions.append(active_tx)
                logger.warning(f"❌ Transacción descartada por TTL: {tx.sign}")
        else:
            logger.info(f"✅ Transacción minada: {tx.sign}")

    # Limpiar las activas antiguas
    state.active_transactions.clear()

    # Mover todas las pendientes a activas para el próximo ciclo
    while (tx := state.pending_transactions.get()) is not None:
        tx.mining_started_at = cycle_closed_at.isoformat()
        state.active_transactions.put(tx)

    # borrar las pendientes
    state.pending_transactions.clear()

    current_active = state.active_transactions.get_all_transactions_with_ttl()
    cycle_summary = {
        "completed_at": cycle_closed_at.isoformat(),
        "cycle_prefix": cycle_prefix,
        "received_chains_count": len(all_chains),
        "selected_blocks": [block.model_dump() for block in mined_blocks],
        "winner_received_at": winner_received_at,
        "winner_elapsed_seconds": winner_elapsed_seconds,
        "winner_avg_seconds_per_task": winner_avg_seconds_per_task,
        "active_before_cycle": [tx.model_dump() for tx in active_before_cycle],
        "requeued_transactions": [tx.model_dump() for tx in requeued_transactions],
        "discarded_transactions": [tx.model_dump() for tx in discarded_transactions],
        "mined_duration_samples": mined_duration_samples,
        "current_active_transactions": [tx.model_dump() for tx in current_active],
        "counts": {
            "active_before_cycle": len(active_before_cycle),
            "mined_blocks": len(mined_blocks),
            "requeued_transactions": len(requeued_transactions),
            "discarded_transactions": len(discarded_transactions),
            "mined_duration_samples": len(mined_duration_samples),
            "current_active_transactions": len(current_active),
        },
    }
    state.persistent_state.set_last_cycle_summary(cycle_summary)

def try_acquire_cycle_lock(ttl: int = 45) -> bool:
    return state.redis_client.set(
        "cycle_lock",
        "locked",
        nx=True,
        ex=ttl
    )