import asyncio
from config import INTERVAL_DURATION, AWAIT_RESPONSE_DURATION, CoordinatorState
from state import blockchain, pending_transactions, active_transactions, received_chains
from utils import seconds_until_next_interval, adjust_difficulty
import state

async def periodic_block_generation():
    while True:
        seconds_to_next = seconds_until_next_interval(INTERVAL_DURATION // 60)
        print(f"Sincronizando... esperando {seconds_to_next:.1f} segundos para el próximo intervalo")
        await asyncio.sleep(seconds_to_next)

        state.cicle_state = CoordinatorState.GIVING_TASKS
        print(f"[Estado] {state.cicle_state.value} - Entregando tareas")

        await asyncio.sleep(INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)

        state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
        print(f"[Estado] {state.cicle_state.value} - Esperando resultados")
        await asyncio.sleep(AWAIT_RESPONSE_DURATION)

        state.cicle_state = CoordinatorState.SELECTING_WINNER
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

        print(f"[Estado] {state.cicle_state.value} - Fin de ciclo\n")
