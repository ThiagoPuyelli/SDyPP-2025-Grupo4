import asyncio
from datetime import datetime
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, CoordinatorState
from state import blockchain, pending_transactions, active_transactions, received_chains
from utils import seconds_until_next_interval, adjust_difficulty
import state

async def periodic_block_generation():
    while True:
        seconds_to_next = seconds_until_next_interval()
        print(f"Syncing... waiting {seconds_to_next:.1f} seconds until next interval")
        await asyncio.sleep(seconds_to_next)

        state.cicle_state = CoordinatorState.GIVING_TASKS
        state.phase_started_at = datetime.utcnow()
        print(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")

        await asyncio.sleep(INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)

        state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
        state.phase_started_at = datetime.utcnow()
        print(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")
        await asyncio.sleep(AWAIT_RESPONSE_DURATION)

        state.cicle_state = CoordinatorState.SELECTING_WINNER
        state.phase_started_at = datetime.utcnow()
        best_chain = max(received_chains, key=len, default=None)
        if best_chain:
            blockchain.extend(best_chain)
            adjust_difficulty()
            print("✔️ Cadena aceptada")
        else:
            print("⚠️ No se recibió cadena válida")

        active_transactions.clear()
        active_transactions.extend(pending_transactions)
        pending_transactions.clear()
        received_chains.clear()

        print(f"[State] {state.cicle_state.name} - {state.cicle_state.value}")
        print(f"### Fin de ciclo ###\n")
