from threading import Thread
from fastapi import FastAPI
from minero import iniciar
from endpoints import endpoints

app = FastAPI(title="Blockchain Miner")

# Incluir routers
app.include_router(endpoints.router)

@app.on_event("startup")
def iniciar_coordinador():
    thread = Thread(target=iniciar, daemon=True)
    thread.start()
