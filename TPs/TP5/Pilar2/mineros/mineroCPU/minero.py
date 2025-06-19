from cumplirTareas import cumplirTareas
import time
import requests
from datetime import datetime, timezone, timedelta
from config import URI, BLOCK_TARGET_TIME, INTERVAL_DURATION, AWAIT_RESPONSE_DURATION
from dateutil.parser import isoparse

def get_last_interval_start(lastPhase: datetime = None) -> datetime:
    if lastPhase is None:
        lastPhase = datetime.now(timezone.utc)
    
    total_seconds = (lastPhase.hour * 3600) + (lastPhase.minute * 60) + lastPhase.second
    current_interval = (total_seconds // INTERVAL_DURATION) * INTERVAL_DURATION
    
    hour = current_interval // 3600
    minute = (current_interval % 3600) // 60
    second = 0  # Opcional: resetear segundos
    
    return lastPhase.replace(hour=hour, minute=minute, second=second, microsecond=0)

def iniciar ():
    transactions = []

    response = requests.get(URI + '/state')
    
    while not response.ok:
        print("Reintento de conexión con el coordinador")
        time.sleep(3)
        response = requests.get(URI + '/state')
        
    data = response.json()
    state = data["state"]
    
    
    current_phase = get_last_interval_start()
    operate = False
    while True:
        if state == "GIVING_TASKS":
            if not operate:
                print("Resolviendo tareas")
                transactions.extend(cumplirTareas(nextCycle))
            now = datetime.now(timezone.utc)
            prox_intervalo = current_phase + timedelta(seconds=INTERVAL_DURATION - AWAIT_RESPONSE_DURATION)
            if (now > prox_intervalo):
                state.cicle_state = CoordinatorState.OPEN_TO_RESULTS
                current_phase = prox_intervalo
            
        elif state == "OPEN_TO_RESULTS":
            if len(transactions) > 0:
                requests.post(URI + "/results", transactions)
                transactions = []
        
        if nextCycle <= datetime.now(timezone.utc):
            if state == "GIVING_TASKS":
                nextCycle = datetime.now(timezone.utc) + timedelta(seconds=BLOCK_TARGET_TIME) 
            elif state == "OPEN_TO_RESULTS":
                nextCycle = datetime.now(timezone.utc) + timedelta(seconds=INTERVAL_DURATION)
                
        delay = (nextCycle - datetime.now(timezone.utc)).total_seconds()
        print("DELAY: ", delay)
        if delay > 0:
            state = "OPEN_TO_RESULTS" if state == "GIVING_TASKS" else "GIVING_TASKS" 
            time.sleep(delay)

iniciar()


            

